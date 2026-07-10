#!/usr/bin/env python3
"""Run both chunking strategies on ingested JSONL and persist chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List

# Allow `python scripts/chunk_docs.py` from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.chunking import (  # noqa: E402
    DEFAULT_ARTICLE_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_by_article,
    chunk_recursive,
    load_processed_records,
    summarize_chunks,
)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk processed legal documents.")
    parser.add_argument(
        "--processed-dir",
        default=os.getenv("PROCESSED_DIR", "data/processed"),
        help="Directory with ingested *.jsonl article records",
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("CHUNKS_DIR", "data"),
        help="Directory for chunks_*.jsonl outputs",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_ARTICLE_MAX_CHARS)
    parser.add_argument(
        "--default-strategy",
        choices=("article", "recursive"),
        default="article",
        help="Which strategy is copied to data/chunks.jsonl",
    )
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.processed_dir, "*.jsonl")))
    if not paths:
        print(f"No processed JSONL files found in {args.processed_dir}")
        sys.exit(1)

    print(f"Loading {len(paths)} processed files...")
    records = load_processed_records(paths)
    print(f"Loaded {len(records)} article/point records.")

    recursive_chunks = chunk_recursive(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    article_chunks = chunk_by_article(
        records,
        max_chars=args.max_chars,
        chunk_overlap=args.chunk_overlap,
    )

    recursive_path = os.path.join(args.out_dir, "chunks_recursive.jsonl")
    article_path = os.path.join(args.out_dir, "chunks_article.jsonl")
    default_path = os.path.join(args.out_dir, "chunks.jsonl")

    write_jsonl(recursive_path, recursive_chunks)
    write_jsonl(article_path, article_chunks)

    default_chunks = article_chunks if args.default_strategy == "article" else recursive_chunks
    write_jsonl(default_path, default_chunks)

    print("\nChunking complete:")
    print(f"  recursive -> {recursive_path}: {summarize_chunks(recursive_chunks)}")
    print(f"  article   -> {article_path}: {summarize_chunks(article_chunks)}")
    print(f"  default   -> {default_path} (strategy={args.default_strategy})")


if __name__ == "__main__":
    main()
