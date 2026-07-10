import pytest
from fastapi.testclient import TestClient
from app.main import app, retriever

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_endpoint_not_loaded():
    # Force retriever state to not loaded
    original_loaded = retriever.is_loaded
    retriever.is_loaded = False
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert "ready" not in response.json()
    finally:
        retriever.is_loaded = original_loaded

def test_ready_endpoint_loaded():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True}
    finally:
        retriever.is_loaded = original_loaded

def test_ask_endpoint_validation_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    # Verify that the custom handler caught it and returned structured detail
    assert "detail" in response.json()

def test_ask_endpoint_validation_too_long_question():
    long_question = "a" * 501
    response = client.post("/ask", json={"question": long_question})
    assert response.status_code == 422

def test_ask_endpoint_success_mocked():
    original_loaded = retriever.is_loaded
    retriever.is_loaded = True
    try:
        response = client.post("/ask", json={"question": "Каковы нормы рабочего времени?"})
        # If no LLM is running locally, it will hit our graceful fallback or mock retriever response
        # and return 200. Let's verify status code is 200.
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "used_context" in data
    finally:
        retriever.is_loaded = original_loaded
