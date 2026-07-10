import pytest
from unittest.mock import AsyncMock, patch
from app.generation import answer, SYSTEM_PROMPT, PROMPT_VERSION
from app.schemas import Chunk, AnswerResponse

# A mock list of chunks
mock_chunks = [
    Chunk(chunk_id="chunk_1", text="Трудовой договор заключается в письменной форме.", source_file="labor_code.txt", page=10),
    Chunk(chunk_id="chunk_2", text="Испытательный срок не может превышать три месяца.", source_file="labor_code.txt", page=12)
]

@pytest.mark.asyncio
async def test_prompt_version():
    assert PROMPT_VERSION == "rag_v1"

@pytest.mark.asyncio
async def test_system_prompt_contents():
    # Assert critical guidelines are inside the system prompt to check for regressions
    assert "ONLY using the provided text contexts" in SYSTEM_PROMPT
    assert "I don't know from the provided context" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_regression_success_grounded(mock_post):
    # Configure mock LLM response returning standard JSON string conforming to schema
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": '{"answer": "Письменная форма обязательна для трудового договора.", "sources": [{"source_file": "labor_code.txt", "page": 10, "chunk_id": "chunk_1"}], "confidence": 0.95, "used_context": true}'
        }
    }
    mock_post.return_value = mock_response

    response: AnswerResponse = await answer("Какова форма трудового договора?", mock_chunks)
    
    assert response.used_context is True
    assert "Письменная форма" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].source_file == "labor_code.txt"
    assert response.confidence == 0.95

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_regression_unrelated_query_refusal(mock_post):
    # If chunks do not contain the answer, LLM should return the exact refusal string
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": '{"answer": "I don\'t know from the provided context", "sources": [], "confidence": 0.0, "used_context": false}'
        }
    }
    mock_post.return_value = mock_response

    response: AnswerResponse = await answer("Какая погода сегодня в Астане?", mock_chunks)
    
    assert response.used_context is False
    assert response.answer == "I don't know from the provided context"
    assert len(response.sources) == 0
    assert response.confidence == 0.0

@pytest.mark.asyncio
async def test_regression_no_context_refusal():
    # If list of retrieved chunks is empty, generator must refuse immediately without calling LLM
    response: AnswerResponse = await answer("Каков испытательный срок?", [])
    assert response.used_context is False
    assert response.answer == "I don't know from the provided context"
    assert len(response.sources) == 0
