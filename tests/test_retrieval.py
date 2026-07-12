import os
import pytest
from evaluation.evaluate_retrieval import calculate_recall_at_k, calculate_mrr, calculate_ndcg_at_k

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "index", "index.faiss")
CHUNKS_PATH = os.path.join(ROOT, "data", "chunks.jsonl")
GT_PATH = os.path.join(ROOT, "evaluation", "ground_truth.jsonl")


def test_recall_at_k():
    expected_ids = {"chunk_A", "chunk_B"}

    retrieved_1 = ["chunk_A", "chunk_C", "chunk_B", "chunk_D"]
    assert calculate_recall_at_k(retrieved_1, expected_ids, k=5) == 1.0

    retrieved_2 = ["chunk_A", "chunk_C", "chunk_E", "chunk_B"]
    assert calculate_recall_at_k(retrieved_2, expected_ids, k=3) == 0.5

    assert calculate_recall_at_k(retrieved_2, expected_ids, k=2) == 0.5
    assert calculate_recall_at_k(["chunk_F", "chunk_G"], expected_ids, k=5) == 0.0


def test_mean_reciprocal_rank():
    expected_ids = {"chunk_A"}

    assert calculate_mrr(["chunk_A", "chunk_B"], expected_ids) == 1.0
    assert calculate_mrr(["chunk_B", "chunk_C", "chunk_A"], expected_ids) == 1.0 / 3.0
    assert calculate_mrr(["chunk_B", "chunk_C"], expected_ids) == 0.0


def test_ndcg_at_k():
    expected_ids = {"chunk_A", "chunk_B"}

    retrieved_perfect = ["chunk_A", "chunk_B"]
    assert pytest.approx(calculate_ndcg_at_k(retrieved_perfect, expected_ids, k=5), 0.01) == 1.0

    retrieved_partial = ["chunk_A", "chunk_C", "chunk_D"]
    assert pytest.approx(calculate_ndcg_at_k(retrieved_partial, expected_ids, k=5), 0.01) == 0.613


@pytest.mark.skipif(
    not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(GT_PATH)),
    reason="FAISS index / chunks / ground truth not built locally",
)
def test_live_retrieval_recall_at_5_meets_threshold():
    """Defense gate: recall@5 on the labeled set must be >= 0.75.

    Runs in a subprocess to isolate native FAISS/torch crashes from the pytest process.
    """
    import subprocess
    import sys

    script = f"""
import json, os, sys
sys.path.insert(0, {ROOT!r})
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from app.retrieval import VectorRetriever
from evaluation.evaluate_retrieval import run_evaluation
retriever = VectorRetriever(
    index_dir={os.path.join(ROOT, "data", "index")!r},
    chunks_path={CHUNKS_PATH!r},
    model_name=os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
)
retriever.load()
assert retriever.is_loaded, "retriever failed to load"
results = run_evaluation({GT_PATH!r}, retriever)
print(json.dumps(results))
assert results["mean_recall_at_5"] >= 0.75, results
"""
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.fail(f"live retrieval eval failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")
