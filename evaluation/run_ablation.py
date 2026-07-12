#!/usr/bin/env python3
"""Step 10 ablation: compare chunk size / strategy (and optional embedding model).

Runs chunk → embed → FAISS → evaluate for each config, writes:
  - evaluation/ablation_results.json
  - prints a markdown table for README

Relevance for recursive configs uses covered article units (not recursive chunk IDs),
so results are comparable to article-aware ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from glob import glob
from typing import Any, Dict, List, Sequence, Set

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

from app.chunking import (  # noqa: E402
    DEFAULT_ARTICLE_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    chunk_by_article,
    chunk_recursive,
    load_processed_records,
    summarize_chunks,
)
from app.retrieval import VectorRetriever  # noqa: E402
from scripts.build_index import (  # noqa: E402
    build_faiss_index,
    compute_embeddings_with_cache,
    write_jsonl,
)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PART_SUFFIX = re.compile(r"_p\d+$")


def normalize_unit_id(chunk_id: str) -> str:
    """Map article part IDs (…_p0) back to the parent article unit."""
    return PART_SUFFIX.sub("", chunk_id)


def units_for_chunk(chunk: Dict[str, Any]) -> Set[str]:
    """Article units covered by a retrieved chunk (strategy-aware)."""
    units: Set[str] = set()
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        units.add(chunk_id)
        units.add(normalize_unit_id(chunk_id))
    parent = chunk.get("parent_article_id")
    if parent:
        units.add(parent)
        units.add(normalize_unit_id(parent))
    for covered in chunk.get("covered_article_ids") or []:
        units.add(covered)
        units.add(normalize_unit_id(covered))
    return units


def expected_units(expected_ids: Sequence[str]) -> Set[str]:
    units: Set[str] = set()
    for eid in expected_ids:
        units.add(eid)
        units.add(normalize_unit_id(eid))
    return units


def evaluate_config(
    ground_truth: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    retriever: VectorRetriever,
    top_k: int = 10,
) -> Dict[str, float]:
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    latencies_ms: List[float] = []

    if not retriever.is_loaded:
        retriever.load()

    for item in ground_truth:
        question = item["question"]
        expected_ids: List[str] = list(item["expected_chunk_ids"])

        t0 = time.perf_counter()
        retrieved, _, _ = retriever.retrieve(question, top_k=top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Enrich retrieved Chunk objects with metadata from the chunk store.
        retrieved_meta: List[Dict[str, Any]] = []
        for chunk in retrieved:
            meta = dict(chunks_by_id.get(chunk.chunk_id, {}))
            meta["chunk_id"] = chunk.chunk_id
            retrieved_meta.append(meta)

        expected_norm = {normalize_unit_id(e) for e in expected_ids}

        # Credit each gold unit at most once (keeps recall/nDCG well-defined).
        remaining = set(expected_norm)
        covered_in_top5: Set[str] = set()
        graded_gains: List[float] = []
        first_hit_rank = None
        for rank, meta in enumerate(retrieved_meta):
            hit_units = {normalize_unit_id(u) for u in units_for_chunk(meta)} & remaining
            # Exact chunk_id match also counts (article-aware indexes).
            cid = meta.get("chunk_id")
            if cid in expected_ids:
                hit_units.add(normalize_unit_id(cid))
            if hit_units:
                if first_hit_rank is None:
                    first_hit_rank = rank
                if rank < 10:
                    graded_gains.append(1.0)
                if rank < 5:
                    covered_in_top5 |= hit_units
                remaining -= hit_units
            elif rank < 10:
                graded_gains.append(0.0)

        while len(graded_gains) < min(10, len(retrieved_meta)):
            graded_gains.append(0.0)

        recall = len(covered_in_top5) / len(expected_norm) if expected_norm else 0.0
        mrr = 0.0 if first_hit_rank is None else 1.0 / (first_hit_rank + 1)
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(graded_gains[:10]))
        ideal_n = min(len(expected_norm), 10)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

    p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    return {
        "mean_recall_at_5": float(np.mean(recalls)),
        "mean_mrr": float(np.mean(mrrs)),
        "mean_ndcg_at_10": float(np.mean(ndcgs)),
        "latency_p95_ms": p95,
        "n_queries": len(ground_truth),
    }


def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_and_evaluate(
    name: str,
    chunks: List[Dict[str, Any]],
    model_name: str,
    ground_truth: List[Dict[str, Any]],
    ablation_root: str,
    cache_dir: str,
) -> Dict[str, Any]:
    run_dir = os.path.join(ablation_root, name)
    chunks_path = os.path.join(run_dir, "chunks.jsonl")
    index_dir = os.path.join(run_dir, "index")
    os.makedirs(run_dir, exist_ok=True)

    write_jsonl(chunks_path, chunks)
    print(f"\n=== {name} ===")
    print(f"Chunks: {summarize_chunks(chunks)}")

    t_embed = time.perf_counter()
    embeddings = compute_embeddings_with_cache(
        chunks=chunks,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    embed_s = time.perf_counter() - t_embed
    build_faiss_index(embeddings, index_dir)

    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    retriever = VectorRetriever(
        index_dir=index_dir,
        chunks_path=chunks_path,
        model_name=model_name,
    )
    metrics = evaluate_config(ground_truth, chunks_by_id, retriever)
    result = {
        "name": name,
        "model": model_name,
        "n_chunks": len(chunks),
        "embed_seconds": round(embed_s, 2),
        **metrics,
    }
    print(
        f"Recall@5={result['mean_recall_at_5']:.4f}  "
        f"MRR={result['mean_mrr']:.4f}  "
        f"nDCG@10={result['mean_ndcg_at_10']:.4f}  "
        f"p95={result['latency_p95_ms']:.1f}ms"
    )
    return result


def default_configs(model_name: str) -> List[Dict[str, Any]]:
    # Largest / production first so a crash on tiny chunks still leaves usable results.
    configs: List[Dict[str, Any]] = [
        {
            "name": "article_aware",
            "strategy": "article",
            "max_chars": DEFAULT_ARTICLE_MAX_CHARS,
            "model": model_name,
            "details": f"Article/point-aware, max_chars={DEFAULT_ARTICLE_MAX_CHARS}",
        }
    ]
    for size in (1600, 800, 400, 200):
        configs.append(
            {
                "name": f"recursive_{size}",
                "strategy": "recursive",
                "chunk_size": size,
                "model": model_name,
                "details": f"Recursive splitter, chunk_size={size}",
            }
        )
    return configs


def materialize_chunks(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    overlap: int,
) -> List[Dict[str, Any]]:
    if config["strategy"] == "recursive":
        return chunk_recursive(
            records,
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=overlap,
        )
    return chunk_by_article(
        records,
        max_chars=int(config.get("max_chars", DEFAULT_ARTICLE_MAX_CHARS)),
        chunk_overlap=overlap,
    )


def markdown_table(results: Sequence[Dict[str, Any]], configs: Sequence[Dict[str, Any]]) -> str:
    details_by_name = {c["name"]: c.get("details", c["name"]) for c in configs}
    lines = [
        "| Config | Details | Recall@5 | MRR | nDCG@10 | Latency p95 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    best_recall = max(r["mean_recall_at_5"] for r in results) if results else 0.0
    for row in results:
        recall = row["mean_recall_at_5"]
        recall_s = f"**{recall:.2f}**" if recall == best_recall else f"{recall:.2f}"
        lines.append(
            f"| {row['name']} | {details_by_name.get(row['name'], '')} | "
            f"{recall_s} | {row['mean_mrr']:.2f} | {row['mean_ndcg_at_10']:.2f} | "
            f"{row['latency_p95_ms']:.0f}ms |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablation sweep.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--ground-truth", default="evaluation/ground_truth.jsonl")
    parser.add_argument("--ablation-root", default="data/ablation")
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("EMBEDDING_CACHE_DIR", "data/index/embedding_cache"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL),
    )
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--out-json",
        default="evaluation/ablation_results.json",
    )
    parser.add_argument(
        "--compare-model",
        default="",
        help="Optional second embedding model evaluated on article-aware chunks only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run configs even if present in --out-json.",
    )
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.processed_dir, "*.jsonl")))
    if not paths:
        print(f"No processed JSONL in {args.processed_dir}. Run make ingest first.")
        sys.exit(1)

    print(f"Loading processed records from {len(paths)} files...")
    records = load_processed_records(paths)
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(records)} records, {len(ground_truth)} ground-truth questions.")

    configs = default_configs(args.model)
    if args.compare_model:
        configs.append(
            {
                "name": "article_alt_model",
                "strategy": "article",
                "max_chars": DEFAULT_ARTICLE_MAX_CHARS,
                "model": args.compare_model,
                "details": f"Article-aware + {args.compare_model.split('/')[-1]}",
            }
        )

    results: List[Dict[str, Any]] = []
    existing: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r", encoding="utf-8") as handle:
                existing = {
                    row["name"]: row
                    for row in json.load(handle).get("configs", [])
                    if "name" in row
                }
            print(f"Resuming; found {len(existing)} prior result(s) in {args.out_json}")
        except Exception:
            existing = {}

    for config in configs:
        if config["name"] in existing and not args.force:
            print(f"Skipping {config['name']} (already in results). Use --force to redo.")
            results.append(existing[config["name"]])
            continue
        chunks = materialize_chunks(records, config, overlap=args.chunk_overlap)
        result = build_and_evaluate(
            name=config["name"],
            chunks=chunks,
            model_name=config["model"],
            ground_truth=ground_truth,
            ablation_root=args.ablation_root,
            cache_dir=args.cache_dir,
        )
        result["details"] = config.get("details", "")
        result["strategy"] = config["strategy"]
        results.append(result)
        # Persist after each config so crashes still leave a partial table.
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump({"configs": results}, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("ABLATION TABLE")
    print("=" * 60)
    print(markdown_table(results, configs))
    print(f"\nWrote {args.out_json}")

    best = max(results, key=lambda r: r["mean_recall_at_5"])
    worst = min(results, key=lambda r: r["mean_recall_at_5"])
    print(
        f"\nTakeaway: {best['name']} leads at Recall@5={best['mean_recall_at_5']:.2f}; "
        f"{worst['name']} is lowest at {worst['mean_recall_at_5']:.2f}."
    )


if __name__ == "__main__":
    main()
