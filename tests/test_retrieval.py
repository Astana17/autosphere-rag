import pytest
from evaluation.evaluate_retrieval import calculate_recall_at_k, calculate_mrr, calculate_ndcg_at_k

def test_recall_at_k():
    expected_ids = {"chunk_A", "chunk_B"}

    # Check perfect recall (both expected are retrieved in top-k)
    retrieved_1 = ["chunk_A", "chunk_C", "chunk_B", "chunk_D"]
    assert calculate_recall_at_k(retrieved_1, expected_ids, k=5) == 1.0

    # Check partial recall (only chunk_A is in top-3)
    retrieved_2 = ["chunk_A", "chunk_C", "chunk_E", "chunk_B"]
    assert calculate_recall_at_k(retrieved_2, expected_ids, k=3) == 0.5

    # Check zero recall (none in top-2)
    assert calculate_recall_at_k(retrieved_2, expected_ids, k=2) == 0.5
    assert calculate_recall_at_k(["chunk_F", "chunk_G"], expected_ids, k=5) == 0.0

def test_mean_reciprocal_rank():
    expected_ids = {"chunk_A"}

    # Chunk A is at rank 1
    assert calculate_mrr(["chunk_A", "chunk_B"], expected_ids) == 1.0

    # Chunk A is at rank 3
    assert calculate_mrr(["chunk_B", "chunk_C", "chunk_A"], expected_ids) == 1.0 / 3.0

    # Not found
    assert calculate_mrr(["chunk_B", "chunk_C"], expected_ids) == 0.0

def test_ndcg_at_k():
    expected_ids = {"chunk_A", "chunk_B"}

    # Perfect retrieval in order: DCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309. IDCG = 1.6309
    retrieved_perfect = ["chunk_A", "chunk_B"]
    assert pytest.approx(calculate_ndcg_at_k(retrieved_perfect, expected_ids, k=5), 0.01) == 1.0

    # Missing one expected chunk: DCG = 1/log2(2) = 1.0. IDCG = 1.6309. NDCG = 1.0 / 1.6309 = 0.613
    retrieved_partial = ["chunk_A", "chunk_C", "chunk_D"]
    assert pytest.approx(calculate_ndcg_at_k(retrieved_partial, expected_ids, k=5), 0.01) == 0.613
