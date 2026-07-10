import time
import uuid
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

from app.schemas import AskRequest, AnswerResponse, SourceReference
from app.retrieval import VectorRetriever
from app.generation import answer, PROMPT_VERSION
from app.observability import logger, request_id_var, metrics_tracker
from app.reliability import run_with_timeout, retry_with_backoff, ServiceTimeoutError, DatabaseConnectionError
from app.ragflow_client import ask_ragflow

load_dotenv()

# Configure retriever settings from environment
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "10.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RAG_PROVIDER = os.getenv("RAG_PROVIDER", "local").lower()

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

INDEX_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AutoSphere RAG</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #16202a;
      --muted: #66737f;
      --line: #d9e0e7;
      --accent: #0d7c66;
      --accent-dark: #075f4e;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      font-weight: 760;
      letter-spacing: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #98a2b3;
    }
    .dot.ok { background: var(--accent); }
    .dot.bad { background: var(--danger); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    label {
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 650;
    }
    textarea {
      width: 100%;
      min-height: 142px;
      resize: vertical;
      border: 1px solid #c7d1db;
      border-radius: 8px;
      padding: 13px 14px;
      color: var(--text);
      font: inherit;
      line-height: 1.45;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(13, 124, 102, 0.16);
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 12px;
    }
    .topk {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    input[type="number"] {
      width: 72px;
      border: 1px solid #c7d1db;
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: wait;
      background: #8aa9a1;
    }
    .answer {
      margin-top: 16px;
      min-height: 180px;
      white-space: pre-wrap;
      line-height: 1.58;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong { font-size: 15px; }
    .sources {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 10px;
    }
    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
      overflow-wrap: anywhere;
    }
    .source strong {
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }
    .source small {
      color: var(--muted);
      line-height: 1.35;
    }
    .excerpt {
      margin-top: 8px;
      color: var(--text);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .error {
      color: var(--danger);
      font-weight: 650;
    }
    .muted { color: var(--muted); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .meta { grid-template-columns: 1fr; }
      .toolbar { align-items: stretch; flex-direction: column; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>AutoSphere RAG</h1>
      <div class="status"><span id="readyDot" class="dot"></span><span id="readyText">Проверяю готовность</span></div>
    </header>
    <main>
      <section class="panel">
        <form id="askForm">
          <label for="question">Вопрос по правилам дорожного движения</label>
          <textarea id="question" maxlength="500" placeholder="Например: Какая максимальная скорость разрешена в населенном пункте?"></textarea>
          <div class="toolbar">
            <div class="topk">
              <span>Источников</span>
              <input id="topK" type="number" min="5" max="20" value="5" />
            </div>
            <button id="askButton" type="submit">Спросить</button>
          </div>
        </form>
        <div id="answer" class="answer muted">Ответ появится здесь.</div>
        <div class="meta">
          <div class="metric"><span>Confidence</span><strong id="confidence">-</strong></div>
          <div class="metric"><span>Контекст</span><strong id="usedContext">-</strong></div>
          <div class="metric"><span>Request ID</span><strong id="requestId">-</strong></div>
        </div>
      </section>
      <aside class="panel">
        <label>Источники</label>
        <div id="sources" class="sources muted">Пока нет источников.</div>
      </aside>
    </main>
  </div>
  <script>
    const readyDot = document.getElementById("readyDot");
    const readyText = document.getElementById("readyText");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");
    const askButton = document.getElementById("askButton");
    const confidenceEl = document.getElementById("confidence");
    const usedContextEl = document.getElementById("usedContext");
    const requestIdEl = document.getElementById("requestId");

    async function checkReady() {
      try {
        const response = await fetch("/ready");
        if (!response.ok) throw new Error("not ready");
        readyDot.className = "dot ok";
        readyText.textContent = "Система готова";
      } catch {
        readyDot.className = "dot bad";
        readyText.textContent = "Индекс не загружен";
      }
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function renderSources(sources) {
      if (!sources || sources.length === 0) {
        sourcesEl.className = "sources muted";
        sourcesEl.textContent = "Источники не вернулись.";
        return;
      }
      sourcesEl.className = "sources";
      sourcesEl.innerHTML = sources.map((source) => `
        <div class="source">
          <strong>${escapeHtml(source.source_file || "unknown")}</strong>
          <small>${escapeHtml(source.section_title || "Без раздела")}<br>${escapeHtml(source.chunk_id || "")}</small>
          ${source.excerpt ? `<div class="excerpt">${escapeHtml(source.excerpt)}</div>` : ""}
        </div>
      `).join("");
    }

    document.getElementById("askForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = document.getElementById("question").value.trim();
      const topK = Math.max(5, Number(document.getElementById("topK").value || 5));
      if (question.length < 3) {
        answerEl.className = "answer error";
        answerEl.textContent = "Введите вопрос длиной хотя бы 3 символа.";
        return;
      }
      askButton.disabled = true;
      answerEl.className = "answer muted";
      answerEl.textContent = "Ищу по документам и формирую ответ...";
      sourcesEl.className = "sources muted";
      sourcesEl.textContent = "Загрузка источников...";
      confidenceEl.textContent = "-";
      usedContextEl.textContent = "-";
      requestIdEl.textContent = "-";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question, top_k: topK})
        });
        const requestId = response.headers.get("X-Request-ID") || "-";
        requestIdEl.textContent = requestId;
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || "Ошибка запроса");
        }
        answerEl.className = "answer";
        answerEl.textContent = data.answer;
        confidenceEl.textContent = Number(data.confidence || 0).toFixed(2);
        usedContextEl.textContent = data.used_context ? "Да" : "Нет";
        renderSources(data.sources);
      } catch (error) {
        answerEl.className = "answer error";
        answerEl.textContent = error.message;
        sourcesEl.className = "sources muted";
        sourcesEl.textContent = "Источники не загружены.";
      } finally {
        askButton.disabled = false;
      }
    });

    checkReady();
  </script>
</body>
</html>
"""

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

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

def fallback_from_retrieved_chunks(retrieved_chunks) -> AnswerResponse:
    useful_chunks = [chunk for chunk in retrieved_chunks if len(chunk.text.strip()) > 80]
    if not useful_chunks:
        useful_chunks = retrieved_chunks

    best_chunk = useful_chunks[0]
    excerpt = best_chunk.text.strip()
    if len(excerpt) > 900:
        excerpt = excerpt[:900].rsplit(" ", 1)[0] + "..."

    return AnswerResponse(
        answer=f"По найденному фрагменту документа:\n\n{excerpt}",
        sources=[
            SourceReference(
                source_file=chunk.source_file,
                page=chunk.page,
                section_title=chunk.section_title,
                chunk_id=chunk.chunk_id,
                excerpt=chunk.text.strip()[:700]
            )
            for chunk in useful_chunks[:3]
        ],
        confidence=0.55,
        used_context=True
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
    if RAG_PROVIDER == "ragflow":
        start_time = time.time()
        try:
            response_data = await run_with_timeout(
                async_func=ask_ragflow,
                timeout_sec=float(os.getenv("RAGFLOW_TIMEOUT_SECONDS", "45")),
                question=request_payload.question,
            )
            total_latency_ms = (time.time() - start_time) * 1000.0
            metrics_tracker.record_request(
                latency_ms=total_latency_ms,
                is_error=False,
                is_cache_hit=False,
                tokens_used=int(len(request_payload.question.split()) * 1.3 + len(response_data.answer.split()) * 1.3),
                top_retrieval_score=response_data.confidence,
            )
            logger.info(
                f"Processed RAGFlow query: '{request_payload.question}'",
                extra={
                    "extra_info": {
                        "question": request_payload.question,
                        "retrieved_chunk_ids": [source.chunk_id for source in response_data.sources],
                        "prompt_version": "ragflow",
                        "model_name": os.getenv("RAGFLOW_MODEL", "model"),
                        "latency_ms_by_stage": {"ragflow_ms": total_latency_ms},
                        "total_latency_ms": total_latency_ms,
                        "token_usage": 0,
                        "answer_length": len(response_data.answer),
                        "cache_hit_bool": False,
                        "error_bool": False,
                    }
                },
            )
            return response_data
        except Exception as exc:
            metrics_tracker.record_request(
                latency_ms=(time.time() - start_time) * 1000.0,
                is_error=True,
                is_cache_hit=False,
                tokens_used=0,
                top_retrieval_score=0.0,
            )
            logger.error(f"RAGFlow request failed: {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="RAGFlow request failed. Check RAGFLOW_BASE_URL, RAGFLOW_CHAT_ID and RAGFLOW_API_KEY.",
            )

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
            response_data = fallback_from_retrieved_chunks(retrieved_chunks)

        if (
            retrieved_chunks
            and not response_data.used_context
            and response_data.answer.strip() == "I don't know from the provided context"
        ):
            response_data = fallback_from_retrieved_chunks(retrieved_chunks)

        source_ids_with_excerpts = {source.chunk_id for source in response_data.sources if source.excerpt}
        for source in response_data.sources:
            if source.excerpt:
                continue
            for chunk in retrieved_chunks:
                if chunk.chunk_id == source.chunk_id:
                    source.excerpt = chunk.text.strip()[:700]
                    source_ids_with_excerpts.add(source.chunk_id)
                    break

        if response_data.used_context and not response_data.sources and retrieved_chunks:
            response_data.sources = fallback_from_retrieved_chunks(retrieved_chunks).sources

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
                    "model_name": os.getenv("LLM_MODEL_NAME", "alemplus-gemma4"),
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
