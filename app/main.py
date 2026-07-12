import time
import uuid
import os
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

load_dotenv()

from app.schemas import AskRequest, AnswerResponse
from app.retrieval import VectorRetriever
from app.generation import answer, PROMPT_VERSION
from app.observability import logger, request_id_var, metrics_tracker
from app.reliability import (
    run_with_timeout,
    retry_with_backoff,
    ServiceTimeoutError,
    DatabaseConnectionError,
)

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

retriever = VectorRetriever(
    index_dir=INDEX_DIR,
    chunks_path=CHUNKS_PATH,
    model_name=EMBEDDING_MODEL_NAME,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    retriever.load()
    yield
    logger.info("Shutting down FastAPI application...")


app = FastAPI(
    title="Production RAG API",
    description="A reliable, observed, and production-grade RAG QA service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)

    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    request_id_var.reset(token)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation failed: {exc.errors()}",
        extra={"extra_info": {"errors": exc.errors(), "body": str(exc.body)}},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Input validation failed."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception occurred: {str(exc)}",
        exc_info=True,
        extra={"extra_info": {"error_type": type(exc).__name__}},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
            "request_id": request_id_var.get(),
        },
    )


@app.get("/health")
def health():
    """Liveness probe — always 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — 200 only when index, chunks, and model are loaded."""
    if retriever.is_loaded:
        return {
            "ready": True,
            "index_loaded": True,
            "chunks_loaded": bool(retriever.chunks),
            "cache_ready": True,
        }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "ready": False,
            "error": "service_not_ready",
            "message": "Retrieval models or indexes are not fully loaded.",
            "request_id": request_id_var.get(),
        },
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask(request_payload: AskRequest):
    """Retrieve, generate with timeout/retry, return structured answer."""
    if not retriever.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_not_ready",
                "message": "The vector search engine is not ready.",
                "request_id": request_id_var.get(),
            },
        )

    start_time = time.time()
    is_error = False
    cache_hit = False
    tokens_used = 0
    top_score = 0.0
    stage_latencies: dict = {}
    retrieved_chunks = []

    try:
        t_start_retrieval = time.time()
        retrieved_chunks, cache_hit, top_score = retriever.retrieve(
            query=request_payload.question,
            top_k=request_payload.top_k or 5,
            filters=request_payload.filters,
        )
        stage_latencies["embedding_ms"] = round(retriever.last_embed_ms, 2)
        stage_latencies["retrieval_ms"] = round((time.time() - t_start_retrieval) * 1000.0, 2)
        stage_latencies["faiss_search_ms"] = round(retriever.last_search_ms, 2)

        t_start_generation = time.time()
        try:

            async def generate_call():
                return await run_with_timeout(
                    async_func=answer,
                    timeout_sec=LLM_TIMEOUT,
                    question=request_payload.question,
                    retrieved_chunks=retrieved_chunks,
                )

            response_data: AnswerResponse = await retry_with_backoff(
                async_func=generate_call,
                exceptions_to_retry=(httpx.HTTPError, ServiceTimeoutError),
                max_retries=MAX_RETRIES,
                initial_delay=1.0,
                backoff_factor=2.0,
            )
        except ServiceTimeoutError:
            is_error = True
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "error": "llm_timeout",
                    "message": "LLM generation timed out after retries.",
                    "request_id": request_id_var.get(),
                },
            )
        except httpx.HTTPError as e:
            logger.error(f"Generation stage failed after retries: {str(e)}")
            # Graceful degradation — honest fallback instead of crashing.
            response_data = AnswerResponse(
                answer="I couldn't search the documents right now due to backend failures.",
                sources=[],
                confidence=0.0,
                used_context=False,
                degraded=True,
            )

        stage_latencies["generation_ms"] = round((time.time() - t_start_generation) * 1000.0, 2)

        prompt_words = len(request_payload.question.split())
        for chunk in retrieved_chunks:
            prompt_words += len(chunk.text.split())
        completion_words = len(response_data.answer.split())
        tokens_used = int((prompt_words + completion_words) * 1.3)

        total_latency_ms = (time.time() - start_time) * 1000.0
        stage_latencies["total_ms"] = round(total_latency_ms, 2)

        logger.info(
            f"Processed query: '{request_payload.question}'",
            extra={
                "extra_info": {
                    "question": request_payload.question,
                    "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
                    "prompt_version": PROMPT_VERSION,
                    "model_name": os.getenv("LLM_MODEL_NAME", "llama3.2:3b"),
                    "latency_ms_by_stage": stage_latencies,
                    "total_latency_ms": round(total_latency_ms, 2),
                    "token_usage": tokens_used,
                    "answer_length": len(response_data.answer),
                    "cache_hit_bool": cache_hit,
                    "error_bool": False,
                    "degraded": response_data.degraded,
                }
            },
        )

        metrics_tracker.record_request(
            latency_ms=total_latency_ms,
            is_error=False,
            is_cache_hit=cache_hit,
            tokens_used=tokens_used,
            top_retrieval_score=top_score,
        )

        return response_data

    except HTTPException:
        is_error = True
        raise
    except DatabaseConnectionError as dbe:
        is_error = True
        logger.error(f"Database connection error: {str(dbe)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "vector_db_unreachable",
                "message": "The vector search service is currently unreachable.",
                "request_id": request_id_var.get(),
            },
        )
    except Exception as e:
        is_error = True
        logger.error(f"Failed to process request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An error occurred processing your request.",
                "request_id": request_id_var.get(),
            },
        )
    finally:
        if is_error:
            total_latency_ms = (time.time() - start_time) * 1000.0
            metrics_tracker.record_request(
                latency_ms=total_latency_ms,
                is_error=True,
                is_cache_hit=False,
                tokens_used=0,
                top_retrieval_score=0.0,
            )


@app.get("/metrics")
def metrics():
    """JSON snapshot of p50/p95 latency, error rate, cache hit rate, tokens, retrieval score."""
    return metrics_tracker.get_snapshot()
