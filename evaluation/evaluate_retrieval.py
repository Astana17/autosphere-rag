#!/usr/bin/env python3
import os
import json
import numpy as np
from typing import List, Dict, Any, Set
from app.retrieval import VectorRetriever
from dotenv import load_dotenv

load_dotenv()

def calculate_recall_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    """
    Computes recall@k: fraction of expected chunks that are retrieved in top-k.
    """
    top_k_retrieved = set(retrieved_ids[:k])
    intersection = top_k_retrieved.intersection(expected_ids)
    if not expected_ids:
        return 0.0
    return len(intersection) / len(expected_ids)

def calculate_mrr(retrieved_ids: List[str], expected_ids: Set[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant chunk.
    """
    for idx, r_id in enumerate(retrieved_ids):
        if r_id in expected_ids:
            return 1.0 / (idx + 1)
    return 0.0

def calculate_ndcg_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    """
    Computes normalized Discounted Cumulative Gain (nDCG@k) with binary relevance.
    """
    dcg = 0.0
    for idx, r_id in enumerate(retrieved_ids[:k]):
        if r_id in expected_ids:
            dcg += 1.0 / np.log2(idx + 2)

    # Calculate Ideal DCG (IDCG)
    idcg = 0.0
    ideal_relevant_count = min(len(expected_ids), k)
    for idx in range(ideal_relevant_count):
        idcg += 1.0 / np.log2(idx + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def run_evaluation(ground_truth_path: str, retriever: VectorRetriever) -> Dict[str, float]:
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at {ground_truth_path}")

    queries = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    recalls_at_5 = []
    mrrs = []
    ndcgs_at_10 = []

    print(f"Running retrieval evaluation on {len(queries)} queries...")

    # Ensure retriever is loaded
    if not retriever.is_loaded:
        print("Retriever is not loaded. Attempting to load...")
        retriever.load()

    for q_idx, item in enumerate(queries):
        question = item["question"]
        expected_ids = set(item["expected_chunk_ids"])

        try:
            # Retrieve top 10 for assessment
            retrieved_chunks, _, _ = retriever.retrieve(question, top_k=10)
            retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        except Exception as e:
            print(f"Error retrieving for question '{question}': {str(e)}")
            retrieved_ids = []

        r5 = calculate_recall_at_k(retrieved_ids, expected_ids, k=5)
        mrr = calculate_mrr(retrieved_ids, expected_ids)
        ndcg10 = calculate_ndcg_at_k(retrieved_ids, expected_ids, k=10)

        recalls_at_5.append(r5)
        mrrs.append(mrr)
        ndcgs_at_10.append(ndcg10)

        print(f"Query {q_idx+1:02d} | Recall@5: {r5:.2f} | MRR: {mrr:.2f} | nDCG@10: {ndcg10:.2f}")

    mean_recall = float(np.mean(recalls_at_5))
    mean_mrr = float(np.mean(mrrs))
    mean_ndcg = float(np.mean(ndcgs_at_10))

    print("\n" + "="*40)
    print("EVALUATION RESULTS SUMMARY")
    print("="*40)
    print(f"Mean Recall@5:  {mean_recall:.4f} (Target >= 0.75)")
    print(f"Mean MRR:       {mean_mrr:.4f}")
    print(f"Mean nDCG@10:   {mean_ndcg:.4f}")
    print("="*40)

    return {
        "mean_recall_at_5": mean_recall,
        "mean_mrr": mean_mrr,
        "mean_ndcg_at_10": mean_ndcg
    }

if __name__ == "__main__":
    INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
    CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    GROUND_TRUTH_PATH = "evaluation/ground_truth.jsonl"

    retriever = VectorRetriever(
        index_dir=INDEX_DIR,
        chunks_path=CHUNKS_PATH,
        model_name=EMBEDDING_MODEL_NAME
    )

    run_evaluation(GROUND_TRUTH_PATH, retriever)
