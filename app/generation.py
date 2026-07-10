import os
import json
import httpx
import inspect
from json import JSONDecodeError
from typing import List
from pydantic import ValidationError
from app.schemas import Chunk, AnswerResponse, SourceReference
from app.observability import logger

PROMPT_VERSION = "rag_v1"

SYSTEM_PROMPT = """You are a highly precise, fact-grounded Q&A assistant for Kazakhstan road traffic rules and related road legislation.
Answer in Russian unless the user clearly asks for another language.
Your task is to answer the user's question ONLY using the provided text contexts.

Constraints:
1. Ground every statement in your answer in the provided contexts.
2. If the answer cannot be found in the contexts, you MUST return:
   - "answer": "I don't know from the provided context"
   - "sources": []
   - "confidence": 0.0
   - "used_context": false
3. You must respond in a valid, strict JSON format matching this Pydantic schema:
   {
     "answer": "string (the final grounded answer, or refusal statement)",
     "sources": [
       {"source_file": "string", "page": integer_or_null, "section_title": "string_or_null", "chunk_id": "string"}
     ],
     "confidence": float (between 0.0 and 1.0 indicating your confidence),
     "used_context": boolean (true if answer is from context, false if context was insufficient)
   }
4. Do not include any preambles, markdown formatting (like ```json), or explanations outside of the JSON. Return only the raw JSON.
5. If a context contains a numbered rule, article, penalty, or exception that directly answers the question, use it. Do not refuse just because other retrieved contexts are unrelated.
"""

async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value

def extract_llm_content(response_data: dict) -> str:
    if "choices" in response_data:
        return response_data["choices"][0]["message"]["content"]
    if "message" in response_data:
        return response_data["message"]["content"]
    raise KeyError("choices/message")

def chat_completions_url(base_url: str) -> str:
    normalized_url = base_url.rstrip("/")
    if normalized_url.endswith("/chat/completions"):
        return normalized_url
    return f"{normalized_url}/chat/completions"

def build_user_prompt(question: str, retrieved_chunks: List[Chunk]) -> str:
    contexts_str = ""
    for idx, chunk in enumerate(retrieved_chunks):
        contexts_str += f"--- CONTEXT {idx+1} (ID: {chunk.chunk_id}, File: {chunk.source_file}, Page: {chunk.page}, Section: {chunk.section_title}) ---\n"
        contexts_str += f"{chunk.text}\n\n"

    return f"Contexts:\n{contexts_str}\nQuestion: {question}"

async def answer(question: str, retrieved_chunks: List[Chunk]) -> AnswerResponse:
    """
    Generates an answer grounded in the retrieved chunks.
    Connects to the configured provider (Ollama or OpenAI) or falls back to mock responses.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    api_key = os.getenv("LLM_API_KEY", "")

    user_content = build_user_prompt(question, retrieved_chunks)

    logger.info(
        f"Invoking LLM provider: {provider} (model: {model_name})",
        extra={
            "extra_info": {
                "prompt_version": PROMPT_VERSION,
                "model_name": model_name,
                "provider": provider,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks]
            }
        }
    )

    # If no chunks are retrieved, immediately return refusal
    if not retrieved_chunks:
        return AnswerResponse(
            answer="I don't know from the provided context",
            sources=[],
            confidence=0.0,
            used_context=False
        )

    # Implementation for Ollama or OpenAI
    try:
        if provider == "openai":
            # Call OpenAI Chat Completions API
            url = chat_completions_url(base_url)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                await maybe_await(response.raise_for_status())
                res_data = await maybe_await(response.json())
                content = extract_llm_content(res_data)

        elif provider == "ollama":
            # Call Ollama API
            url = f"{base_url.rstrip('/')}/api/chat"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                },
                "format": "json"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)
                await maybe_await(response.raise_for_status())
                res_data = await maybe_await(response.json())
                content = extract_llm_content(res_data)

        elif provider == "alemplus":
            url = chat_completions_url(base_url)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                await maybe_await(response.raise_for_status())
                res_data = await maybe_await(response.json())
                content = extract_llm_content(res_data)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Parse output JSON
        data = json.loads(content.strip())

        # Convert sources dict list to SourceReference schema objects
        sources_list = []
        for src in data.get("sources", []):
            sources_list.append(
                SourceReference(
                    source_file=src.get("source_file", "unknown"),
                    page=src.get("page"),
                    section_title=src.get("section_title"),
                    chunk_id=src.get("chunk_id"),
                    excerpt=src.get("excerpt")
                )
            )

        return AnswerResponse(
            answer=data.get("answer", "I don't know from the provided context"),
            sources=sources_list,
            confidence=float(data.get("confidence", 0.0)),
            used_context=bool(data.get("used_context", False))
        )

    except (JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as e:
        logger.error(
            f"LLM returned malformed content: {str(e)}. Falling back to default response.",
            extra={"extra_info": {"error": str(e)}}
        )
        return AnswerResponse(
            answer="Извините, не удалось сформировать корректный ответ из контекста.",
            sources=[],
            confidence=0.0,
            used_context=False
        )
