.PHONY: install index chunk ingest serve test clean help

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make install  - Create venv and install dependencies"
	@echo "  make ingest   - Extract and process raw DOCX files to JSONL"
	@echo "  make chunk    - Run recursive + article chunking -> data/chunks*.jsonl"
	@echo "  make index    - Embed chunks and build FAISS index"
	@echo "  make serve    - Start the FastAPI web service"
	@echo "  make test     - Run unit tests with pytest"
	@echo "  make clean    - Remove virtual environment and temporary files"

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

serve:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
