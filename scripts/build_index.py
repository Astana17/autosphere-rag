#!/usr/bin/env python3
"""Build embeddings + FAISS index from persisted chunks (or chunk first)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

from app.chunking import (  # noqa: E402
    DEFAULT_ARTICLE_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PREPROCESSING_VERSION,
    chunk_by_article,
    load_processed_records,
    summarize_chunks,
)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def ensure_chunks(
    chunks_path: str,
    processed_dir: str,
    max_chars: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """Load existing chunks.jsonl or build article-aware chunks from processed docs."""
    if os.path.exists(chunks_path) and os.path.getsize(chunks_path) > 0:
        print(f"Loading existing chunks from {chunks_path}...")
        return load_chunks(chunks_path)

    paths = sorted(glob(os.path.join(processed_dir, "*.jsonl")))
    if not paths:
        print(f"No processed JSONL in {processed_dir}. Run `make ingest` first.")
        sys.exit(1)

    print(f"Building article-aware chunks from {len(paths)} processed files...")
    records = load_processed_records(paths)
    chunks = chunk_by_article(records, max_chars=max_chars, chunk_overlap=chunk_overlap)
    write_jsonl(chunks_path, chunks)
    print(f"Wrote {len(chunks)} chunks to {chunks_path}: {summarize_chunks(chunks)}")
    return chunks


def load_embedding_cache(cache_path: str) -> Dict[str, List[float]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}
    return {}


def save_embedding_cache(cache: Dict[str, List[float]], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)


def compute_embeddings_with_cache(
    chunks: List[Dict[str, Any]],
    model_name: str,
    cache_dir: str,
    batch_size: int = 64,
) -> np.ndarray:
    cache_path = os.path.join(cache_dir, "cache.json")
    cache = load_embedding_cache(cache_path)

    print("Loading embedding model for index creation...")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    if SentenceTransformer is not None:
        model = SentenceTransformer(model_name, device=device)
    else:
        model = None
    if model is None:
        print("Mock Mode: SentenceTransformer not available. Generating random vectors.")

    hashes: List[str] = []
    texts: List[str] = []
    for chunk in chunks:
        chunk_text = chunk["text"]
        raw_key = f"{chunk_text}||{model_name}||{PREPROCESSING_VERSION}"
        chunk_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        hashes.append(chunk_hash)
        texts.append(chunk_text)

    missing_indices = [i for i, h in enumerate(hashes) if h not in cache]
    print(
        f"Embedding cache: {len(hashes) - len(missing_indices)} hits, "
        f"{len(missing_indices)} misses (device={device})"
    )

    if missing_indices:
        if model is not None:
            missing_texts = [texts[i] for i in missing_indices]
            for start in range(0, len(missing_texts), batch_size):
                batch = missing_texts[start : start + batch_size]
                vectors = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                for offset, vector in enumerate(vectors):
                    cache[hashes[missing_indices[start + offset]]] = vector.tolist()
                done = min(start + batch_size, len(missing_texts))
                if done % (batch_size * 5) == 0 or done == len(missing_texts):
                    print(f"Computed embeddings for {done}/{len(missing_texts)} new chunks...")
                # Checkpoint large caches so a crash does not lose the whole run.
                if done % (batch_size * 20) == 0:
                    save_embedding_cache(cache, cache_path)
        else:
            for idx in missing_indices:
                cache[hashes[idx]] = np.random.randn(384).tolist()

        save_embedding_cache(cache, cache_path)
        print("Saved updated embedding cache to disk.")

    embeddings = np.array([cache[h] for h in hashes], dtype=np.float32)
    return embeddings


def build_faiss_index(embeddings: np.ndarray, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss_file = os.path.join(index_dir, "index.faiss")
    dimension = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP index with dimension: {dimension}...")

    if faiss is not None:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, faiss_file)
        print(f"Saved FAISS index to {faiss_file}")
    else:
        print("FAISS package not installed. Skipping index binary write.")
        with open(faiss_file, "w", encoding="utf-8") as handle:
            handle.write("FAISS index mock file content")


def main() -> None:
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
    chunks_path = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
    index_dir = os.getenv("INDEX_DIR", "data/index")
    embedding_cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "data/index/embedding_cache")
    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    max_chars = int(os.getenv("ARTICLE_MAX_CHARS", str(DEFAULT_ARTICLE_MAX_CHARS)))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
    _ = DEFAULT_CHUNK_SIZE  # documented default for recursive ablation

    print("Step 1: Ensuring chunks are available...")
    chunks = ensure_chunks(
        chunks_path=chunks_path,
        processed_dir=processed_dir,
        max_chars=max_chars,
        chunk_overlap=chunk_overlap,
    )
    print(f"Using {len(chunks)} chunks: {summarize_chunks(chunks)}")

    print("Step 2: Computing embeddings...")
    embeddings = compute_embeddings_with_cache(
        chunks=chunks,
        model_name=model_name,
        cache_dir=embedding_cache_dir,
    )
    print(f"Computed embeddings matrix shape: {embeddings.shape}")

    print("Step 3: Storing and indexing...")
    build_faiss_index(embeddings, index_dir)
    print("Pipeline index build completed successfully!")


if __name__ == "__main__":
    main()
