import os
from typing import Any, Dict, List

import httpx

from app.schemas import AnswerResponse, SourceReference


def _ragflow_url() -> str:
    base_url = os.getenv("RAGFLOW_BASE_URL", "").rstrip("/")
    chat_id = os.getenv("RAGFLOW_CHAT_ID", "").strip()
    if not base_url:
        raise ValueError("RAGFLOW_BASE_URL is not configured.")
    if not chat_id:
        raise ValueError("RAGFLOW_CHAT_ID is not configured.")
    return f"{base_url}/api/v1/openai/{chat_id}/chat/completions"


def _source_from_reference_chunk(chunk: Dict[str, Any]) -> SourceReference:
    document_name = chunk.get("document_name") or chunk.get("doc_name") or "ragflow"
    chunk_id = chunk.get("id") or chunk.get("chunk_id")
    metadata = chunk.get("document_metadata") or {}
    section_title = metadata.get("section_title") or metadata.get("title")
    return SourceReference(
        source_file=document_name,
        page=None,
        section_title=section_title,
        chunk_id=chunk_id,
        excerpt=(chunk.get("content") or "")[:700],
    )


def _extract_sources(message: Dict[str, Any]) -> List[SourceReference]:
    reference = message.get("reference") or {}
    chunks = reference.get("chunks") or {}
    if isinstance(chunks, dict):
        return [_source_from_reference_chunk(chunk) for chunk in chunks.values()]
    if isinstance(chunks, list):
        return [_source_from_reference_chunk(chunk) for chunk in chunks]
    return []


async def ask_ragflow(question: str) -> AnswerResponse:
    api_key = os.getenv("RAGFLOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("RAGFLOW_API_KEY is not configured.")

    payload = {
        "model": os.getenv("RAGFLOW_MODEL", "model"),
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "extra_body": {
            "reference": True,
            "reference_metadata": {"include": True},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=float(os.getenv("RAGFLOW_TIMEOUT_SECONDS", "45"))) as client:
        response = await client.post(_ragflow_url(), headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    message = data["choices"][0]["message"]
    answer = message.get("content") or "I don't know from the provided context"
    sources = _extract_sources(message)
    return AnswerResponse(
        answer=answer,
        sources=sources,
        confidence=0.9 if sources else 0.6,
        used_context=bool(sources),
    )
