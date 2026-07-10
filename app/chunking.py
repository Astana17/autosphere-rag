"""Chunking strategies for the legal-document RAG corpus.

Strategy 1 — recursive: fixed-size RecursiveCharacterTextSplitter baseline.
Strategy 2 — article: one chunk per ingested Статья/Пункт, with size-capped
sub-splits that preserve parent article metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

PREPROCESSING_VERSION = "prep_v1"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_ARTICLE_MAX_CHARS = 1500

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only without the dep
    RecursiveCharacterTextSplitter = None


def _slug(value: str) -> str:
    """Make a stable, filesystem-friendly fragment for chunk IDs."""
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^\w\-]+", "", value, flags=re.UNICODE)
    return value or "unit"


def _section_title(record: Dict[str, Any]) -> str:
    article = (record.get("article") or "").strip()
    title = (record.get("title") or "").strip()
    if article and title and title != article:
        return f"{article}. {title}"
    return article or title or (record.get("chapter") or "Untitled")


def _base_metadata(record: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    return {
        "strategy": strategy,
        "source_file": record.get("source_file"),
        "document_id": record.get("document_id"),
        "document_title": record.get("document_title"),
        "chapter": record.get("chapter"),
        "article": record.get("article"),
        "section_title": _section_title(record),
        "article_id": record.get("article_id"),
        "source_url": record.get("source_url"),
        "language": record.get("language"),
        "jurisdiction": record.get("jurisdiction"),
        "document_version": record.get("document_version") or PREPROCESSING_VERSION,
        "page": record.get("page"),
    }


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    if not text:
        return []
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        return [part for part in splitter.split_text(text) if part.strip()]

    # Fallback when langchain-text-splitters is unavailable.
    chunks: List[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _char_offsets(full_text: str, piece: str, search_from: int = 0) -> tuple[int, int]:
    idx = full_text.find(piece, search_from)
    if idx < 0:
        idx = full_text.find(piece)
    if idx < 0:
        return 0, len(piece)
    return idx, idx + len(piece)


def load_processed_records(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Load ingested article/point records from one or more JSONL files."""
    import json

    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def group_records_by_document(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        key = record.get("document_id") or record.get("source_file") or "unknown"
        grouped.setdefault(key, []).append(record)
    return grouped


def chunk_recursive(
    records: Sequence[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Baseline: concatenate each document, then recursively split by size."""
    chunks: List[Dict[str, Any]] = []
    for document_id, doc_records in group_records_by_document(records).items():
        full_text = "\n\n".join(
            (rec.get("text") or "").strip() for rec in doc_records if (rec.get("text") or "").strip()
        )
        if not full_text:
            continue

        # Use the first record as the document-level metadata carrier.
        prototype = dict(doc_records[0])
        prototype["article"] = None
        prototype["title"] = None
        prototype["article_id"] = document_id
        prototype["chapter"] = None

        parts = _split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        cursor = 0
        for idx, part in enumerate(parts):
            char_start, char_end = _char_offsets(full_text, part, search_from=cursor)
            cursor = max(char_end - chunk_overlap, char_start + 1)
            meta = _base_metadata(prototype, strategy="recursive")
            meta.update(
                {
                    "chunk_id": f"{document_id}_rec_{idx}",
                    "text": part,
                    "char_start": char_start,
                    "char_end": char_end,
                    "section_title": f"{prototype.get('document_title') or document_id} [rec {idx}]",
                    "parent_article_id": None,
                    "part_index": idx,
                    "part_count": len(parts),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
            chunks.append(meta)
    return chunks


def chunk_by_article(
    records: Sequence[Dict[str, Any]],
    max_chars: int = DEFAULT_ARTICLE_MAX_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Primary: one chunk per Статья/Пункт; sub-split oversized units."""
    chunks: List[Dict[str, Any]] = []
    for record in records:
        text = (record.get("text") or "").strip()
        if not text:
            continue

        article_id = record.get("article_id") or _slug(
            f"{record.get('document_id', 'doc')}_{record.get('article', 'unit')}"
        )
        section = _section_title(record)

        if len(text) <= max_chars:
            meta = _base_metadata(record, strategy="article")
            meta.update(
                {
                    "chunk_id": article_id,
                    "text": text,
                    "char_start": 0,
                    "char_end": len(text),
                    "parent_article_id": article_id,
                    "part_index": 0,
                    "part_count": 1,
                    "max_chars": max_chars,
                }
            )
            chunks.append(meta)
            continue

        # Keep the article header visible in every sub-chunk for retrieval/citation.
        header = section
        body = text
        if text.startswith(header):
            body = text[len(header) :].lstrip(" .\n")
        elif record.get("article") and text.startswith(str(record["article"])):
            header = str(record["article"])
            body = text[len(header) :].lstrip(" .\n")

        # Budget for the header prefix so total length stays near max_chars.
        prefix = f"{header}\n" if header else ""
        body_budget = max(max_chars - len(prefix), 200)
        parts = _split_text(body, chunk_size=body_budget, chunk_overlap=chunk_overlap)
        if not parts:
            parts = [body]

        cursor = 0
        for idx, part in enumerate(parts):
            # Offsets refer to the original article text.
            search_text = part
            char_start, char_end = _char_offsets(text, search_text, search_from=cursor)
            if char_start == 0 and search_text not in text:
                # Part may be body-only; locate inside body region.
                char_start, char_end = _char_offsets(text, part, search_from=cursor)
            cursor = max(char_end - chunk_overlap, char_start + 1)

            chunk_text = f"{prefix}{part}".strip() if prefix and not part.startswith(header) else part
            meta = _base_metadata(record, strategy="article")
            meta.update(
                {
                    "chunk_id": f"{article_id}_p{idx}",
                    "text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "parent_article_id": article_id,
                    "part_index": idx,
                    "part_count": len(parts),
                    "max_chars": max_chars,
                }
            )
            chunks.append(meta)
    return chunks


def run_chunking(
    records: Sequence[Dict[str, Any]],
    strategy: str = "article",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chars: int = DEFAULT_ARTICLE_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Dispatch helper used by scripts and notebooks."""
    strategy = strategy.lower().strip()
    if strategy in {"recursive", "rec", "baseline"}:
        return chunk_recursive(
            records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strategy in {"article", "section", "article_aware"}:
        return chunk_by_article(
            records,
            max_chars=max_chars,
            chunk_overlap=chunk_overlap,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def summarize_chunks(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact stats for notebooks / CLI output."""
    if not chunks:
        return {"count": 0}
    lengths = [len(c.get("text") or "") for c in chunks]
    strategies = sorted({c.get("strategy") for c in chunks})
    return {
        "count": len(chunks),
        "strategies": strategies,
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "documents": sorted({c.get("document_id") for c in chunks if c.get("document_id")}),
    }
