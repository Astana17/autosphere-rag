import time
import uuid
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.schemas import AskRequest, AnswerResponse
from app.retrieval import VectorRetriever
from app.generation import answer, PROMPT_VERSION
from app.observability import logger, request_id_var, metrics_tracker
from app.reliability import run_with_timeout, retry_with_backoff, ServiceTimeoutError, DatabaseConnectionError

# Configure retriever settings from environment
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "10.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

retriever = VectorRetriever(
    index_dir=INDEX_DIR,
    chunks_path=CHUNKS_PATH,
    model_name=EMBEDDING_MODEL_NAME
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load vector db retriever components
    logger.info("Starting up FastAPI application...")
    retriever.load()
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Production RAG API",
    description="A reliable, observed, and production-grade RAG QA service.",
    version="1.0.0",
    lifespan=lifespan
)

# Request ID and Timing Middleware
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    # Retrieve request_id from headers or generate new uuid
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # Store request_id in context variable
    token = request_id_var.set(request_id)
    
    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    
    request_id_var.reset(token)
    return response

# Custom Exception Handler to prevent 500 stacktrace leaks
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation failed: {exc.errors()}",
        extra={"extra_info": {"errors": exc.errors(), "body": str(exc.body)}}
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Input validation failed."}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception occurred: {str(exc)}",
        exc_info=True,
        extra={"extra_info": {"error_type": type(exc).__name__}}
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
            "request_id": request_id_var.get()
        }
    )

@app.get("/health")
def health():
    """
    Liveness probe. Always returns HTTP 200 status code if process is alive.
    """
    return {"status": "ok"}

@app.get("/ready")
def ready():
    """
    Readiness probe. Returns HTTP 200 only if retrieval components are loaded.
    Otherwise returns HTTP 503.
    """
    if retriever.is_loaded:
        return {"ready": True}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval models or indexes are not fully loaded."
        )

@app.post("/ask", response_model=AnswerResponse)
async def ask(request_payload: AskRequest):
    """
    Main endpoint. Accepts a question, retrieves relevant documents,
    queries the LLM (with retries and timeouts), and returns structured answers.
    """
    if not retriever.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector search engine is not ready."
        )
        
    start_time = time.time()
    is_error = False
    cache_hit = False
    tokens_used = 0
    top_score = 0.0
    stage_latencies = {}
    
    try:
        # Stage 1: Retrieval
        t_start_retrieval = time.time()
        retrieved_chunks, cache_hit, top_score = retriever.retrieve(
            query=request_payload.question,
            top_k=request_payload.top_k or 5
        )
        stage_latencies["retrieval_ms"] = (time.time() - t_start_retrieval) * 1000.0
        
        # Stage 2: Generation with exponential retry and timeout
        t_start_generation = time.time()
        try:
            # We combine exponential retry and timeout boundaries
            async def generate_call():
                return await run_with_timeout(
                    async_func=answer,
                    timeout_sec=LLM_TIMEOUT,
                    question=request_payload.question,
                    retrieved_chunks=retrieved_chunks
                )

            # Retry on connection or timeout exceptions
            response_data: AnswerResponse = await retry_with_backoff(
                async_func=generate_call,
                exceptions_to_retry=(httpx.HTTPError, ServiceTimeoutError),
                max_retries=MAX_RETRIES,
                initial_delay=1.0,
                backoff_factor=2.0
            )
        except ServiceTimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM generation timed out. Please try again."
            )
        except Exception as e:
            logger.error(f"Generation stage failed: {str(e)}")
            # Graceful degradation fallback rather than crashing hard
            response_data = AnswerResponse(
                answer="I couldn't search the documents right now due to backend failures.",
                sources=[],
                confidence=0.0,
                used_context=False
            )
            
        stage_latencies["generation_ms"] = (time.time() - t_start_generation) * 1000.0
        
        # Estimate tokens (simplistic heuristic: 1 word ~ 1.3 tokens)
        prompt_words = len(request_payload.question.split())
        for chunk in retrieved_chunks:
            prompt_words += len(chunk.text.split())
        completion_words = len(response_data.answer.split())
        tokens_used = int((prompt_words + completion_words) * 1.3)
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000.0
        
        # Log structured request summary
        logger.info(
            f"Processed query: '{request_payload.question}'",
            extra={
                "extra_info": {
                    "question": request_payload.question,
                    "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
                    "prompt_version": PROMPT_VERSION,
                    "model_name": os.getenv("LLM_MODEL_NAME", "llama3.2:3b"),
                    "latency_ms_by_stage": stage_latencies,
                    "total_latency_ms": total_latency_ms,
                    "token_usage": tokens_used,
                    "answer_length": len(response_data.answer),
                    "cache_hit_bool": cache_hit,
                    "error_bool": False
                }
            }
        )
        
        # Record metrics
        metrics_tracker.record_request(
            latency_ms=total_latency_ms,
            is_error=False,
            is_cache_hit=cache_hit,
            tokens_used=tokens_used,
            top_retrieval_score=top_score
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
            detail="The vector search service is currently unreachable."
        )
    except Exception as e:
        is_error = True
        logger.error(f"Failed to process request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your request."
        )
    finally:
        if is_error:
            total_latency_ms = (time.time() - start_time) * 1000.0
            metrics_tracker.record_request(
                latency_ms=total_latency_ms,
                is_error=True,
                is_cache_hit=False,
                tokens_used=0,
                top_retrieval_score=0.0
            )

@app.get("/metrics")
def metrics():
    """
    Returns a snapshot of the operational RAG performance metrics.
    """
    return metrics_tracker.get_snapshot()
