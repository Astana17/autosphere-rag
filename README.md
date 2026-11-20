# AutoSphere

AutoSphere is a team-built document question-answering prototype for Russian-language road traffic and vehicle insurance documents from Kazakhstan. It uses multilingual embeddings and FAISS to retrieve relevant sections, then returns an answer and source references through FastAPI.

## How it works

1. The ingestion scripts extract articles and metadata from DOCX files.
2. The indexing pipeline chunks the articles, embeds them with multilingual MiniLM, and saves a FAISS index.
3. `POST /ask` retrieves relevant passages and passes them to the configured language model. The response includes source filenames, section titles, and chunk IDs when available.

The API exposes `/health`, `/ready`, and `/metrics` for basic service diagnostics. See the [chunking decision record](docs/adr/001-chunking.md) for the tradeoffs between fixed-size chunks and article boundaries.

## Run locally

Use Python and Make. For answer generation, start a local Ollama server and pull the default model with `ollama pull llama3.2:3b`. Building the index also downloads the embedding model on its first run.

```bash
make install
make index
make serve
```

`make index` uses the checked-in records under [data/processed](data/processed). Run `make ingest` first if you want to regenerate those records from the bundled DOCX files.

Once `GET /ready` returns HTTP 200, open `http://localhost:8000/docs` or send a question from the domain covered by the corpus:

```bash
curl --fail-with-body http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Какие документы должен иметь при себе водитель?", "top_k":5}'
```

The default language model endpoint is `http://localhost:11434`. To use a different endpoint or model, set `LLM_BASE_URL` or `LLM_MODEL_NAME` in the server environment.

## Corpus and evaluation

The bundled records cover four documents. Each processed record includes the document version and a source URL when available. Consult the original publications for current wording.

| Document | Source recorded in the corpus |
| --- | --- |
| Road traffic rules | [Adilet](https://adilet.zan.kz/rus/docs/P2300000534) |
| Code of Administrative Offences | [Adilet](https://adilet.zan.kz/rus/docs/K1400000235) |
| Mandatory vehicle liability insurance law | [Adilet](https://adilet.zan.kz/rus/docs/Z030000446_) |
| Road traffic law | [Adilet](https://adilet.zan.kz/rus/docs/Z1400000194) |

The [evaluation script](evaluation/evaluate_retrieval.py) reports Recall@5, MRR, and nDCG@10. The checked-in [example labels](evaluation/ground_truth.jsonl) refer to a sample corpus. To benchmark the bundled road traffic corpus, update the labels with expected chunk IDs from the rebuilt index and run `venv/bin/python -m evaluation.evaluate_retrieval` from the repository root.

## Tests

```bash
make test
```

The test suite covers request handling, chunking, retrieval, prompt construction, and retry behavior.
