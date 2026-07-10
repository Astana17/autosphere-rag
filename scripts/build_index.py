#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import numpy as np
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# We try importing FAISS, SentenceTransformer, Langchain.
# The code is written defensively to run or guide the user.
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    faiss = None
    SentenceTransformer = None
    RecursiveCharacterTextSplitter = None

PREPROCESSING_VERSION = "prep_v1"

def load_raw_documents(data_dir: str) -> List[Dict[str, Any]]:
    """
    Loads raw text/markdown/PDF documents from the raw directory.
    Returns list of dicts: [{'source_file': '...', 'text': '...'}]
    """
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        # Create a tiny dummy file to make sure it doesn't fail
        dummy_path = os.path.join(data_dir, "sample.txt")
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write("Трудовой кодекс Республики Казахстан. Статья 1. Основные определения. Трудовые отношения - это отношения между работником и работодателем.")
        print(f"Created a sample raw file at {dummy_path} for demonstration.")

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith((".txt", ".md", ".html")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({
                    "source_file": file_name,
                    "text": content
                })
            except Exception as e:
                print(f"Failed to read {file_name}: {str(e)}")
    return documents

def clean_and_normalize(text: str) -> str:
    """
    Cleans raw document text: normalizes spaces, strips page headers/footers placeholders, etc.
    """
    # Simple normalization skeleton
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Strip header/footer patterns if matched (simple heuristic)
        if "Page " in line or "Стр. " in line:
            continue
        cleaned_lines.append(line.strip())
    
    normalized = " ".join(cleaned_lines)
    # Collapse multiple spaces
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()

def chunk_documents_recursive(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Strategy 1: Recursive Character Chunking.
    """
    chunks = []
    if RecursiveCharacterTextSplitter is None:
        print("Warning: langchain-text-splitters is not installed. Using simple fallback chunker.")
        # Simple character-based fallback splitter
        for doc in documents:
            text = clean_and_normalize(doc["text"])
            doc_len = len(text)
            start = 0
            idx = 0
            while start < doc_len:
                end = start + chunk_size
                chunk_text = text[start:end]
                chunks.append({
                    "chunk_id": f"{doc['source_file']}_rec_{idx}",
                    "text": chunk_text,
                    "source_file": doc["source_file"],
                    "char_start": start,
                    "char_end": start + len(chunk_text),
                    "document_version": PREPROCESSING_VERSION
                })
                start += (chunk_size - chunk_overlap)
                idx += 1
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    for doc in documents:
        text = clean_and_normalize(doc["text"])
        splits = splitter.split_text(text)
        for idx, split_text in enumerate(splits):
            # Try finding boundaries
            char_start = text.find(split_text)
            char_end = char_start + len(split_text) if char_start != -1 else None
            chunks.append({
                "chunk_id": f"{doc['source_file']}_rec_{idx}",
                "text": split_text,
                "source_file": doc["source_file"],
                "char_start": char_start if char_start != -1 else None,
                "char_end": char_end,
                "document_version": PREPROCESSING_VERSION
            })
    return chunks

def chunk_documents_by_article(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strategy 2: Section-aware chunker (custom article boundaries).
    Highly optimized for legal texts (Kazakh Labor Code / Tax Code).
    """
    chunks = []
    import re
    # Matches patterns like "Статья 123." or "Статья 12-1."
    article_pattern = re.compile(r'(Статья\s+\d+(?:-\d+)?\.)')
    
    for doc in documents:
        text = clean_and_normalize(doc["text"])
        parts = article_pattern.split(text)
        
        # parts will look like: ["intro text", "Статья 1.", "article 1 text", "Статья 2.", "article 2 text"]
        if len(parts) <= 1:
            # Fallback if no articles found
            chunks.append({
                "chunk_id": f"{doc['source_file']}_sec_0",
                "text": text,
                "source_file": doc["source_file"],
                "section_title": "Full Document",
                "char_start": 0,
                "char_end": len(text),
                "document_version": PREPROCESSING_VERSION
            })
            continue

        # Ingest header text first if present
        if parts[0].strip():
            chunks.append({
                "chunk_id": f"{doc['source_file']}_sec_header",
                "text": parts[0].strip(),
                "source_file": doc["source_file"],
                "section_title": "Introductory Provisions",
                "char_start": 0,
                "char_end": len(parts[0]),
                "document_version": PREPROCESSING_VERSION
            })
            
        char_idx = len(parts[0])
        idx = 1
        for i in range(1, len(parts), 2):
            article_title = parts[i].strip()
            article_content = parts[i+1].strip() if i+1 < len(parts) else ""
            full_article_text = f"{article_title} {article_content}".strip()
            
            char_start = char_idx
            char_end = char_start + len(full_article_text)
            char_idx = char_end
            
            chunks.append({
                "chunk_id": f"{doc['source_file']}_sec_{idx}",
                "text": full_article_text,
                "source_file": doc["source_file"],
                "section_title": article_title.rstrip("."),
                "char_start": char_start,
                "char_end": char_end,
                "document_version": PREPROCESSING_VERSION
            })
            idx += 1
            
    return chunks

def load_embedding_cache(cache_path: str) -> Dict[str, List[float]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_embedding_cache(cache: Dict[str, List[float]], cache_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

def compute_embeddings_with_cache(
    chunks: List[Dict[str, Any]],
    model_name: str,
    cache_dir: str
) -> np.ndarray:
    """
    Computes embeddings using the SentenceTransformer model.
    Utilizes an on-disk cache to prevent re-computing embeddings for unchanged chunks.
    """
    cache_path = os.path.join(cache_dir, "cache.json")
    cache = load_embedding_cache(cache_path)
    
    print("Loading embedding model for index creation...")
    if SentenceTransformer is not None:
        model = SentenceTransformer(model_name)
    else:
        print("Mock Mode: SentenceTransformer not available. Generating random vectors.")
        model = None

    embeddings = []
    cache_updated = False
    
    for idx, chunk in enumerate(chunks):
        # Cache key: hash(text + model_name + preprocessing_version)
        chunk_text = chunk["text"]
        raw_key = f"{chunk_text}||{model_name}||{PREPROCESSING_VERSION}"
        chunk_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        
        if chunk_hash in cache:
            embeddings.append(cache[chunk_hash])
        else:
            if model is not None:
                vector = model.encode([chunk_text])[0].tolist()
            else:
                # Random embedding mock for 384 dimensions
                vector = np.random.randn(384).tolist()
            cache[chunk_hash] = vector
            embeddings.append(vector)
            cache_updated = True
            
        if (idx + 1) % 100 == 0:
            print(f"Computed embeddings for {idx + 1}/{len(chunks)} chunks...")
            
    if cache_updated:
        save_embedding_cache(cache, cache_path)
        print("Saved updated embedding cache to disk.")
        
    return np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray, index_dir: str):
    """
    Indexes vectors into FAISS (in-memory L2/InnerProduct index) and persists to disk.
    """
    os.makedirs(index_dir, exist_ok=True)
    faiss_file = os.path.join(index_dir, "index.faiss")
    
    dimension = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP index with dimension: {dimension}...")
    
    if faiss is not None:
        # L2 norm vectors for Cosine Similarity using Inner Product Index Flat IP
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, faiss_file)
        print(f"Saved FAISS index to {faiss_file}")
    else:
        # Mock index saving
        print("FAISS package not installed. Skipping index binary write.")
        with open(faiss_file, "w") as f:
            f.write("FAISS index mock file content")

def main():
    raw_dir = "data/raw"
    chunks_path = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
    index_dir = os.getenv("INDEX_DIR", "data/index")
    embedding_cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "data/index/embedding_cache")
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Step 1: Loading raw documents...")
    docs = load_raw_documents(raw_dir)
    if not docs:
        print("No source documents found in raw directory. Exiting.")
        sys.exit(0)

    print("Step 2: Splitting documents into chunks...")
    # Choose default chunking strategy (Article boundary custom strategy is default)
    # Can be swapped for Recursive Character splitter sweep in ablation study
    chunks = chunk_documents_by_article(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"Step 3: Persisting chunk metadata to {chunks_path}...")
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Step 4: Computing embeddings...")
    embeddings = compute_embeddings_with_cache(
        chunks=chunks,
        model_name=model_name,
        cache_dir=embedding_cache_dir
    )
    print(f"Computed embeddings matrix shape: {embeddings.shape}")

    print("Step 5: Storing and indexing...")
    build_faiss_index(embeddings, index_dir)
    print("Pipeline index build completed successfully!")

if __name__ == "__main__":
    main()
