#!/usr/bin/env python3
"""Evaluate retrieval quality on the labeled ground-truth set.

By default uses article-unit matching (same as ablation): `_pN` suffixes and
`parent_article_id` / `covered_article_ids` count toward the gold unit.
Pass `--exact` for strict chunk_id equality.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Set

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.retrieval import VectorRetriever

load_dotenv()

PART_SUFFIX = re.compile(r"_p\d+$")


def normalize_unit_id(chunk_id: str) -> str:
    return PART_SUFFIX.sub("", chunk_id)


def calculate_recall_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    top_k_retrieved = set(retrieved_ids[:k])
    intersection = top_k_retrieved.intersection(expected_ids)
    if not expected_ids:
        return 0.0
    return len(intersection) / len(expected_ids)


def calculate_mrr(retrieved_ids: List[str], expected_ids: Set[str]) -> float:
    for idx, r_id in enumerate(retrieved_ids):
        if r_id in expected_ids:
            return 1.0 / (idx + 1)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, r_id in enumerate(retrieved_ids[:k]):
        if r_id in expected_ids:
            dcg += 1.0 / np.log2(idx + 2)

    idcg = 0.0
    ideal_relevant_count = min(len(expected_ids), k)
    for idx in range(ideal_relevant_count):
        idcg += 1.0 / np.log2(idx + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def units_for_chunk_row(chunk_row: Dict[str, Any], chunk_id: str) -> Set[str]:
    units: Set[str] = {chunk_id, normalize_unit_id(chunk_id)}
    parent = chunk_row.get("parent_article_id")
    if parent:
        units.add(parent)
        units.add(normalize_unit_id(parent))
    for covered in chunk_row.get("covered_article_ids") or []:
        units.add(covered)
        units.add(normalize_unit_id(covered))
    return units


def run_evaluation(
    ground_truth_path: str,
    retriever: VectorRetriever,
    *,
    exact_ids: bool = False,
) -> Dict[str, float]:
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at {ground_truth_path}")

    queries: List[Dict[str, Any]] = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    chunks_by_id = {c.get("chunk_id"): c for c in retriever.chunks if c.get("chunk_id")}

    recalls_at_5: List[float] = []
    mrrs: List[float] = []
    ndcgs_at_10: List[float] = []

    print(f"Running retrieval evaluation on {len(queries)} queries (exact_ids={exact_ids})...")

    if not retriever.is_loaded:
        print("Retriever is not loaded. Attempting to load...")
        retriever.load()

    for q_idx, item in enumerate(queries):
        question = item["question"]
        expected_ids = set(item["expected_chunk_ids"])

        try:
            retrieved_chunks, _, _ = retriever.retrieve(question, top_k=10)
        except Exception as e:
            print(f"Error retrieving for question '{question}': {str(e)}")
            retrieved_chunks = []

        if exact_ids:
            retrieved_ids = [c.chunk_id for c in retrieved_chunks]
            expected_for_metrics = expected_ids
            r5 = calculate_recall_at_k(retrieved_ids, expected_for_metrics, k=5)
            mrr = calculate_mrr(retrieved_ids, expected_for_metrics)
            ndcg10 = calculate_ndcg_at_k(retrieved_ids, expected_for_metrics, k=10)
        else:
            expected_norm = {normalize_unit_id(e) for e in expected_ids}
            remaining = set(expected_norm)
            covered_in_top5: Set[str] = set()
            first_hit_rank = None
            for rank, chunk in enumerate(retrieved_chunks):
                row = chunks_by_id.get(chunk.chunk_id, {"chunk_id": chunk.chunk_id})
                hit_units = {normalize_unit_id(u) for u in units_for_chunk_row(row, chunk.chunk_id)} & remaining
                if hit_units:
                    if first_hit_rank is None:
                        first_hit_rank = rank
                    if rank < 5:
                        covered_in_top5 |= hit_units
                    remaining -= hit_units

            r5 = len(covered_in_top5) / len(expected_norm) if expected_norm else 0.0
            mrr = 0.0 if first_hit_rank is None else 1.0 / (first_hit_rank + 1)
            ndcg_ids: List[str] = []
            rem = set(expected_norm)
            for chunk in retrieved_chunks[:10]:
                row = chunks_by_id.get(chunk.chunk_id, {"chunk_id": chunk.chunk_id})
                hits = {normalize_unit_id(u) for u in units_for_chunk_row(row, chunk.chunk_id)} & rem
                if hits:
                    uid = next(iter(hits))
                    ndcg_ids.append(uid)
                    rem -= hits
                else:
                    ndcg_ids.append(f"__miss_{len(ndcg_ids)}")
            ndcg10 = calculate_ndcg_at_k(ndcg_ids, expected_norm, k=10)

        recalls_at_5.append(r5)
        mrrs.append(mrr)
        ndcgs_at_10.append(ndcg10)

        print(f"Query {q_idx + 1:02d} | Recall@5: {r5:.2f} | MRR: {mrr:.2f} | nDCG@10: {ndcg10:.2f}")

    mean_recall = float(np.mean(recalls_at_5))
    mean_mrr = float(np.mean(mrrs))
    mean_ndcg = float(np.mean(ndcgs_at_10))

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 40)
    print(f"Mean Recall@5:  {mean_recall:.4f} (Target >= 0.75)")
    print(f"Mean MRR:       {mean_mrr:.4f}")
    print(f"Mean nDCG@10:   {mean_ndcg:.4f}")
    print("=" * 40)

    return {
        "mean_recall_at_5": mean_recall,
        "mean_mrr": mean_mrr,
        "mean_ndcg_at_10": mean_ndcg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", action="store_true", help="Strict chunk_id matching")
    args = parser.parse_args()

    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
    CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    GROUND_TRUTH_PATH = "evaluation/ground_truth.jsonl"

    retriever = VectorRetriever(
        index_dir=INDEX_DIR,
        chunks_path=CHUNKS_PATH,
        model_name=EMBEDDING_MODEL_NAME,
    )

    run_evaluation(GROUND_TRUTH_PATH, retriever, exact_ids=args.exact)
