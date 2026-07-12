#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import numpy as np
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from typing import List, Dict, Any, Optional

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

DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def read_docx_text(file_path: str) -> str:
    """
    Extracts paragraph text from a .docx file using the Office Open XML package.
    This keeps setup lightweight and avoids needing Word or LibreOffice.
    """
    with ZipFile(file_path) as docx_zip:
        document_xml = docx_zip.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.findall(".//w:p", DOCX_NAMESPACE):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", DOCX_NAMESPACE)]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)

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

    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith((".txt", ".md", ".html", ".docx")):
            try:
                if file_name.endswith(".docx"):
                    content = read_docx_text(file_path)
                else:
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
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.replace("\xa0", " ").strip()
        # Strip header/footer patterns if matched (simple heuristic)
        if "Page " in line or "Стр. " in line:
            continue
        if line:
            cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r'[ \t]+', ' ', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
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

# Matches the start of a defined-term entry in a glossary-style section, e.g.
# "27) дорожное движение - совокупность общественных отношений..." or
# "13) дорожное движение – совокупность...". Both ПДД (Пункт 2) and the road traffic
# law (Статья 1) list dozens of unrelated defined terms as "N) термин - определение;"
# in one continuous block of text.
GLOSSARY_ENTRY_PATTERN = re.compile(r'^(?P<num>\d+(?:-\d+)?)\)\s*(?P<term>[^\n]*?)\s*[–-]\s', re.MULTILINE)
GLOSSARY_MIN_ENTRIES = 5  # below this, it's more likely an ordinary numbered list than a glossary
# Guards against false positives: an ordinary article can also enumerate "N) ..." items
# (e.g. Статья 73 lists licence subcategories "D1E", "CE1", ...) without being a glossary
# of unrelated terms. Only the two sections that open with this defining phrase actually
# are one — matching on it avoids exploding unrelated enumerated lists into "terms".
GLOSSARY_INTRO_PATTERN = re.compile(r'используются следующие основные понятия', re.IGNORECASE)


def split_glossary_by_term(section: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    If a section is a glossary (opens with "...используются следующие основные понятия:"
    followed by many "N) term - definition;" entries), split it into one chunk per defined
    term instead of chopping it at an arbitrary character offset. A query about one specific
    term (e.g. "уступить дорогу") then retrieves a small, precise chunk instead of competing
    against ~a dozen sibling chunks that each pack 15-20 unrelated definitions together.
    Returns None if the section doesn't look like a glossary, so the caller falls back to
    length-based splitting.
    """
    text = section["text"]
    if not GLOSSARY_INTRO_PATTERN.search(text):
        return None

    matches = list(GLOSSARY_ENTRY_PATTERN.finditer(text))
    if len(matches) < GLOSSARY_MIN_ENTRIES:
        return None

    chunks = []
    base_start = section.get("char_start") or 0
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entry_text = text[start:end].strip()
        if not entry_text:
            continue
        term = match.group("term").strip()
        chunk = dict(section)
        chunk["chunk_id"] = f"{section['chunk_id']}_term_{i}"
        chunk["text"] = entry_text
        chunk["section_title"] = f"{section.get('section_title', '')} — {term}"[:160]
        chunk["char_start"] = base_start + start
        chunk["char_end"] = base_start + end
        # A term entry is normally short, but the last one absorbs everything up to the
        # section end -- if the glossary list doesn't cleanly end the section, cap it like
        # any other long chunk instead of letting it balloon unbounded.
        chunks.extend(split_long_section(chunk))
    return chunks


def split_long_section(section: Dict[str, Any], chunk_size: int = 1600, overlap: int = 150) -> List[Dict[str, Any]]:
    text = section["text"]
    if len(text) <= chunk_size:
        return [section]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk = dict(section)
            chunk["chunk_id"] = f"{section['chunk_id']}_part_{idx}"
            chunk["text"] = chunk_text
            chunk["char_start"] = (section.get("char_start") or 0) + start
            chunk["char_end"] = (section.get("char_start") or 0) + end
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
        idx += 1
    return chunks

def chunk_documents_by_article(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strategy 2: Section-aware chunker (custom article boundaries).
    Highly optimized for legal texts (Kazakh Labor Code / Tax Code).
    """
    chunks = []
    # Structural headers (РАЗДЕЛ/Глава/Статья) always start a new section, regardless
    # of document type. Bare numbered items ("73. ...") only start a new section when
    # they are top-level clauses (e.g. the ПДД itself, which has no "Статья" wrapper) -
    # inside a "Статья N" block they are sub-clauses of that article and must stay merged,
    # otherwise every numbered sentence of a KoAP/law article gets split into its own chunk.
    header_pattern = re.compile(
        r'^(?P<title>(?:РАЗДЕЛ\s+\d+\.?.*|Глава\s+\d+\.?.*|Статья\s+\d+(?:-\d+)?\.?.*))$',
        re.IGNORECASE
    )
    bare_item_pattern = re.compile(r'^(?P<title>\d+\.\s+.*)$')

    for doc in documents:
        text = clean_and_normalize(doc["text"])
        lines = text.splitlines()
        current_lines = []
        current_title = "Introductory Provisions"
        current_start = 0
        section_idx = 0
        cursor = 0

        def flush_section():
            nonlocal section_idx, current_lines, current_title, current_start
            section_text = "\n".join(current_lines).strip()
            if not section_text:
                return
            base_chunk = {
                "chunk_id": f"{doc['source_file']}_sec_{section_idx}",
                "text": section_text,
                "source_file": doc["source_file"],
                "section_title": current_title,
                "char_start": current_start,
                "char_end": current_start + len(section_text),
                "document_version": PREPROCESSING_VERSION
            }
            glossary_chunks = split_glossary_by_term(base_chunk)
            chunks.extend(glossary_chunks if glossary_chunks is not None else split_long_section(base_chunk))
            section_idx += 1

        for line in lines:
            stripped = line.strip()
            inside_article = current_title.lower().startswith("статья")
            match = header_pattern.match(stripped)
            if not match and not inside_article:
                match = bare_item_pattern.match(stripped)

            if match and current_lines:
                flush_section()
                current_lines = [stripped]
                current_title = match.group("title")[:160]
                current_start = text.find(stripped, cursor)
            else:
                if match:
                    current_title = match.group("title")[:160]
                    current_start = text.find(stripped, cursor)
                current_lines.append(stripped)
            cursor += len(line) + 1

        flush_section()

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
