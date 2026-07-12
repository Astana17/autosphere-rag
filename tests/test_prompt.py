import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.generation import answer, SYSTEM_PROMPT, PROMPT_VERSION, REFUSAL_ANSWER, build_user_prompt
from app.schemas import Chunk, AnswerResponse


@pytest.fixture(autouse=True)
def llm_test_env(monkeypatch):
    """Unit tests mock the Ollama HTTP shape; keep them independent of .env."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL_NAME", "llama3.2:3b")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("LLM_API_KEY", raising=False)


mock_chunks = [
    Chunk(
        chunk_id="road_traffic_rules_73",
        text="В населенных пунктах движение транспортных средств разрешается со скоростью не более 60 км/ч.",
        source_file="road_traffic_rules.docx",
        page=None,
        section_title="Пункт 73",
    ),
    Chunk(
        chunk_id="code_of_administrative_offences_591",
        text="Пользование водителем телефоном во время управления транспортным средством влечет штраф.",
        source_file="code_of_administrative_offences.docx",
        section_title="Статья 591",
    ),
]


def _mock_ollama_client(content: str):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": content}}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_prompt_version():
    assert PROMPT_VERSION == "rag_v1"


@pytest.mark.asyncio
async def test_system_prompt_contents():
    assert "ONLY using the provided text contexts" in SYSTEM_PROMPT
    assert REFUSAL_ANSWER in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_build_user_prompt_includes_chunk_ids():
    prompt = build_user_prompt("Какая скорость?", mock_chunks)
    assert "road_traffic_rules_73" in prompt
    assert "Какая скорость?" in prompt


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_regression_success_grounded_speed_limit(mock_client_cls):
    mock_client_cls.return_value = _mock_ollama_client(
        '{"answer": "В населенных пунктах разрешено не более 60 км/ч.", '
        '"sources": [{"source_file": "road_traffic_rules.docx", "chunk_id": "road_traffic_rules_73"}], '
        '"confidence": 0.95, "used_context": true}'
    )
    response: AnswerResponse = await answer(
        "Какая максимальная скорость разрешена в населенных пунктах?",
        mock_chunks,
    )
    assert response.used_context is True
    assert "60" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "road_traffic_rules_73"
    assert response.confidence == 0.95


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_regression_success_grounded_phone_fine(mock_client_cls):
    mock_client_cls.return_value = _mock_ollama_client(
        '{"answer": "Пользование телефоном за рулем влечет административный штраф.", '
        '"sources": [{"source_file": "code_of_administrative_offences.docx", '
        '"chunk_id": "code_of_administrative_offences_591"}], '
        '"confidence": 0.9, "used_context": true}'
    )
    response = await answer(
        "Какой штраф предусмотрен за пользование водителем телефоном?",
        mock_chunks,
    )
    assert response.used_context is True
    assert "штраф" in response.answer.lower()
    assert response.sources[0].chunk_id == "code_of_administrative_offences_591"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_regression_unrelated_query_refusal(mock_client_cls):
    mock_client_cls.return_value = _mock_ollama_client(
        f'{{"answer": "{REFUSAL_ANSWER}", "sources": [], "confidence": 0.0, "used_context": false}}'
    )
    response = await answer("Какая погода сегодня в Астане?", mock_chunks)
    assert response.used_context is False
    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []
    assert response.confidence == 0.0


@pytest.mark.asyncio
async def test_regression_no_context_refusal():
    response = await answer("Каков испытательный срок?", [])
    assert response.used_context is False
    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_regression_malformed_json_falls_back(mock_client_cls):
    mock_client_cls.return_value = _mock_ollama_client("this is not json at all")
    response = await answer("Какая максимальная скорость?", mock_chunks)
    assert response.answer == REFUSAL_ANSWER
    assert response.degraded is True
    assert response.used_context is False
