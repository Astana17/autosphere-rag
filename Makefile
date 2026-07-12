.PHONY: install index chunk ingest serve serve-dev test ablation measure-cache smoke clean help

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make install       - Create venv and install dependencies"
	@echo "  make ingest        - Extract and process raw DOCX files to JSONL"
	@echo "  make chunk         - Run recursive + article chunking -> data/chunks*.jsonl"
	@echo "  make index         - Embed chunks and build FAISS index"
	@echo "  make serve         - Start the FastAPI web service (stable, no reload)"
	@echo "  make serve-dev     - Start with --reload (may hang on first /ask with PyTorch)"
	@echo "  make test          - Run unit tests with pytest"
	@echo "  make ablation      - Chunk-size / strategy ablation -> evaluation/ablation_results.json"
	@echo "  make measure-cache - Retrieval cache p50/p95 -> evaluation/cache_latency.json"
	@echo "  make smoke         - 100 GT questions against running server (SMOKE_BASE_URL)"
	@echo "  make clean         - Remove virtual environment and temporary files"

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest_docs.py

chunk:
	$(PYTHON) scripts/chunk_docs.py

index:
	$(PYTHON) scripts/build_index.py

SERVE_ENV = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

serve:
	$(SERVE_ENV) $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

serve-dev:
	$(SERVE_ENV) $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 $(PYTHON) -m pytest tests/ -q

ablation:
	EMBEDDING_DEVICE=cpu TOKENIZERS_PARALLELISM=false $(PYTHON) evaluation/run_ablation.py

measure-cache:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EMBEDDING_DEVICE=cpu TOKENIZERS_PARALLELISM=false $(PYTHON) scripts/measure_cache.py

smoke:
	$(PYTHON) scripts/smoke_test.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
