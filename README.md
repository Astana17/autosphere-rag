# Production-Grade Retrieval-Augmented Generation (RAG) System

## 1. What This Project Does
This project implements a production-grade, highly reliable, and observed Retrieval-Augmented Generation (RAG) system. The service reads documents from a legal/financial corpus, splits and indexes them using a vector database (FAISS), and leverages an LLM to generate precise, grounded answers backed by source citations. Built using FastAPI, the service includes structured logging, latency & error instrumentation, custom caching layers, and reliability wrappers (timeouts/exponential backoffs) to operate robustly under production workloads.

## 2. Corpus and Licensing
- **Corpus Name:** [e.g. SEC 10-K filings / Kazakh Labor & Tax Code]
- **Source Link:** [e.g. adilet.zan.kz / SEC EDGAR]
- **Licensing and Permission:** [Explain permission/license to use this corpus, e.g., "Publicly available legal codes under the Government of Kazakhstan Open Data policy." or "Public SEC corporate filings available under fair use/public domain."]

## 3. Quickstart
To run the service locally from a fresh clone, execute the following commands:

```bash
# 1. Initialize environment and install dependencies
make install

# 2. Ingest documents and build the vector index
make index

# 3. Start the FastAPI web service
make serve
```

### Sample Query Example
Once the service is running, you can ask questions using `curl`:

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Каков минимальный размер отпускных согласно Трудовому кодексу?", "top_k": 5}'
```

#### Sample Response:
```json
{
  "answer": "Согласно статье 103 Трудового кодекса РК, минимальный размер оплачиваемого ежегодного трудового отпуска...",
  "sources": [
    {
      "source_file": "kazakh_labor_code.txt",
      "page": 45,
      "section_title": "Статья 103",
      "chunk_id": "chunk_421"
    }
  ],
  "confidence": 0.94,
  "used_context": true
}
```

## 4. Architecture Diagram
The architecture consists of offline data preparation and online query serving layers.

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion [Offline Pipeline]
        RawDocs[Raw Documents] --> Clean[Text Normalizer]
        Clean --> Chunk[Document Chunkers]
        Chunk --> Embed[Embedding Model]
        Embed --> Cache[(Embedding Cache)]
        Embed --> FAISS[(FAISS Index Store)]
    end

    %% Web Service
    subgraph Runtime [FastAPI Service]
        Client[Client Request] --> API[FastAPI Endpoints]
        API --> Observability[JSON Logger / Metrics Middleware]
        API --> Retrieval[Retriever Module]
        Retrieval --> CacheHit{Cache Hit?}
        CacheHit -- Yes --> GetCached[Return Cache]
        CacheHit -- No --> Search[FAISS Vector Search]
        Search --> GenPrompt[Prompt Builder]
        GenPrompt --> LLM[LLM Generation Client]
        LLM --> Retry[Reliability: Retry / Fallback]
        Retry --> Output[Validated JSON Output]
    end

    %% Storage connections
    Search -.-> FAISS
```

## 5. Retrieval Quality
The pipeline has been evaluated on a manually-labeled ground-truth set of 15 questions
(`evaluation/ground_truth.jsonl`). Ablation scores below use **article-unit matching**
(so recursive chunk IDs remain comparable to article-aware labels). Exact `chunk_id`
matching via `evaluation/evaluate_retrieval.py` on the production article index scores
**Recall@5 = 0.70 / MRR = 0.66 / nDCG@10 = 0.65**.

| Model / Configuration | Chunk Strategy | Recall@5 | MRR | nDCG@10 |
| :--- | :--- | :---: | :---: | :---: |
| Recursive baseline (MiniLM, size=800) | Recursive splitter | 0.83 | 0.71 | 0.76 |
| **Current production** (MiniLM) | **Article/point-aware** | **0.97** | **0.85** | **0.86** |

*Target boundary:* `recall@5 >= 0.75` on the labeled set (article-aware unit match hits **0.97**).

## 6. Ablation Results
Chunk-size / strategy sweep with fixed embedding model
`paraphrase-multilingual-MiniLM-L12-v2`. Reproduce with `make ablation`
(writes `evaluation/ablation_results.json`).

| Config | Details | Recall@5 | MRR | nDCG@10 | Latency p95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| recursive_200 | Recursive splitter, chunk_size=200 | 0.67 | 0.62 | 0.64 | 14ms |
| recursive_400 | Recursive splitter, chunk_size=400 | 0.83 | 0.74 | 0.76 | 13ms |
| recursive_800 | Recursive splitter, chunk_size=800 | 0.83 | 0.71 | 0.76 | 13ms |
| recursive_1600 | Recursive splitter, chunk_size=1600 | 0.73 | 0.64 | 0.67 | 31ms |
| **article_aware** | **Article/point-aware, max_chars=1500** | **0.97** | **0.85** | **0.86** | **13ms** |

**Takeaway:** Article/point-aware chunking wins (Recall@5 **0.97**) because the corpus and
ground-truth labels are already organized as `Статья`/`Пункт` units. Fixed-size recursive
chunks peak at 400–800 chars (0.83); 200-char slices cut definitions mid-sentence (0.67),
while 1600-char windows mix unrelated articles and dilute the embedding (0.73).

## 7. Latency and Cost Budget
- **Latency Performance (without cache):**
  - p50: ~1.2s (largely LLM generation bound)
  - p95: ~2.4s
- **Latency Performance (with cache):**
  - p50: ~15ms (Cache Hit)
  - p95: ~35ms
- **Estimated Cost:**
  - Standard compute: Free (using local Ollama inference)
  - Commercial API cost (if using OpenAI GPT-4o-mini): ~$0.005 per 1,000 requests.

## 8. What Happens When It Breaks (Failure Modes)
The system is built to degrade gracefully when individual parts fail:

1. **LLM API/Ollama Downtime (Connection Error / Timeout)**
   - **Behavior:** The API attempts up to 3 retries using exponential backoff (1s → 2s → 4s) in `app/reliability.py`. If all fail, the client receives an HTTP 504 response with error code `llm_service_unavailable` and a clear description containing the request ID.
2. **Vector Database / FAISS Corrupt or Missing Index**
   - **Behavior:** The readiness endpoint `/ready` monitors the loaded state of the FAISS index. If the files are missing on startup, `/ready` returns `{"ready": false}` and the service rejects incoming requests with HTTP 503 `service_not_ready` instead of crashing.
3. **Invalid Prompt Format / Pydantic Parse Failure**
   - **Behavior:** If the LLM generates a malformed response that cannot be parsed into our strict JSON schemas, our validation layer catches the exception, reports it in structured log outputs, and falls back to a standardized refusal response: `{"answer": "Извините, не удалось сформировать корректный ответ из контекста.", "sources": [], "confidence": 0.0}` with HTTP 200 and a degraded flag.
