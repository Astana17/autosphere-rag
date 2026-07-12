#!/usr/bin/env python3
"""Measure retrieval-cache effect on p50/p95 latency (50 GT questions × 2 passes)."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

from app.retrieval import VectorRetriever  # noqa: E402


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(sorted(values), dtype=np.float64)
    idx = min(int(len(arr) * p), len(arr) - 1)
    return float(arr[idx])


def load_questions(path: str, limit: int = 50) -> List[str]:
    questions: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            questions.append(json.loads(line)["question"])
            if len(questions) >= limit:
                break
    # Repeat GT set to reach ~50 queries if needed.
    while questions and len(questions) < limit:
        questions.extend(questions[: limit - len(questions)])
    return questions[:limit]


def time_pass(retriever: VectorRetriever, questions: List[str], use_cache: bool) -> List[float]:
    latencies: List[float] = []
    for question in questions:
        t0 = time.perf_counter()
        retriever.retrieve(question, top_k=5, use_cache=use_cache)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def main() -> None:
    gt_path = os.path.join(ROOT, "evaluation", "ground_truth.jsonl")
    index_dir = os.getenv("INDEX_DIR", os.path.join(ROOT, "data", "index"))
    chunks_path = os.getenv("CHUNKS_PATH", os.path.join(ROOT, "data", "chunks.jsonl"))
    model = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    out_path = os.path.join(ROOT, "evaluation", "cache_latency.json")

    questions = load_questions(gt_path, limit=50)
    if not questions:
        print("No questions found in ground truth.")
        sys.exit(1)

    retriever = VectorRetriever(index_dir=index_dir, chunks_path=chunks_path, model_name=model)
    retriever.load()
    if not retriever.is_loaded:
        print("Retriever not loaded. Run `make index` first.")
        sys.exit(1)

    retriever.clear_cache()
    cold = time_pass(retriever, questions, use_cache=False)
    retriever.clear_cache()
    # Prime cache on first warm-prep pass, then measure hits.
    _ = time_pass(retriever, questions, use_cache=True)
    warm = time_pass(retriever, questions, use_cache=True)

    results = {
        "n_questions": len(questions),
        "without_cache_p50_ms": round(percentile(cold, 0.50), 2),
        "without_cache_p95_ms": round(percentile(cold, 0.95), 2),
        "with_cache_p50_ms": round(percentile(warm, 0.50), 2),
        "with_cache_p95_ms": round(percentile(warm, 0.95), 2),
        "speedup_p95": round(percentile(cold, 0.95) / max(percentile(warm, 0.95), 1e-6), 2),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
