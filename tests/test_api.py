import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import httpx

from app.main import app, retriever
from app.reliability import ServiceTimeoutError
from app.schemas import AnswerResponse

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_not_loaded():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = False
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert body["error"] == "service_not_ready"
    finally:
        retriever.is_loaded = original_loaded


def test_ready_endpoint_loaded():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["index_loaded"] is True
        assert body["cache_ready"] is True
    finally:
        retriever.is_loaded = original_loaded


def test_ask_endpoint_validation_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_ask_endpoint_validation_too_long_question():
    long_question = "a" * 501
    response = client.post("/ask", json={"question": long_question})
    assert response.status_code == 422


def test_ask_endpoint_success_mocked():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True

    mock_answer = AnswerResponse(
        answer="В населенных пунктах разрешено не более 60 км/ч.",
        sources=[],
        confidence=0.9,
        used_context=True,
        degraded=False,
    )

    with patch.object(retriever, "retrieve", return_value=([], False, 0.85)):
        with patch("app.main.retry_with_backoff", new=AsyncMock(return_value=mock_answer)):
            try:
                response = client.post(
                    "/ask",
                    json={
                        "question": "Какая максимальная скорость в населенных пунктах?",
                        "top_k": 3,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert "sources" in data
                assert "confidence" in data
                assert "used_context" in data
                assert data["degraded"] is False
            finally:
                retriever.is_loaded = original_loaded


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    for key in (
        "p50_latency_ms",
        "p95_latency_ms",
        "error_rate",
        "cache_hit_rate",
        "total_tokens_consumed",
        "mean_retrieval_score",
    ):
        assert key in data


def test_ask_llm_timeout_returns_504():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True

    async def raise_timeout(*args, **kwargs):
        raise ServiceTimeoutError("timed out")

    with patch.object(retriever, "retrieve", return_value=([], False, 0.1)):
        with patch("app.main.retry_with_backoff", side_effect=raise_timeout):
            try:
                response = client.post(
                    "/ask",
                    json={"question": "Какой штраф за телефон за рулем?"},
                )
                assert response.status_code == 504
                detail = response.json()["detail"]
                assert detail["error"] == "llm_timeout"
                assert "request_id" in detail
            finally:
                retriever.is_loaded = original_loaded


def test_ask_http_error_graceful_degradation():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True

    async def raise_http(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    with patch.object(retriever, "retrieve", return_value=([], False, 0.1)):
        with patch("app.main.retry_with_backoff", side_effect=raise_http):
            try:
                response = client.post(
                    "/ask",
                    json={"question": "Какой штраф за телефон за рулем?"},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["degraded"] is True
                assert data["used_context"] is False
                assert data["sources"] == []
            finally:
                retriever.is_loaded = original_loaded
