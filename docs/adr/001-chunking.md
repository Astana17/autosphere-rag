# ADR 001: Chunking Strategy for Legal Document RAG System

- **Status:** Approved
- **Date:** 2026-07-10
- **Author:** Founding ML Engineer

## Context

Our RAG system answers employee questions grounded in structured Kazakh legal texts
(road traffic law/rules, administrative offences code, compulsory motor insurance law).

These documents have clear semantic units: `Статья N` for laws/codes and `Пункт N` for
traffic rules. Arbitrary fixed-length splits cut mid-article, which hurts retrieval and
makes citations unreliable.

Step 3 already persists article/point records in `data/processed/*.jsonl`. Chunking must
reuse those boundaries rather than re-parsing DOCX with fragile regex.

## Options Considered

### Option 1: Naive Fixed-Size Recursive Character Splitter
`RecursiveCharacterTextSplitter` over each full document (concatenated processed records),
chunk size 200–1600 with overlap.
- **Pros:** Required RAID baseline; domain-agnostic; easy size ablation.
- **Cons:** Cuts articles mid-list; mixes unrelated articles in one chunk; weaker citations.

### Option 2: Article/Point-Aware Splitter (with size cap)
One chunk per ingested `Статья` / `Пункт`. Units longer than 1,500 characters are
sub-split with the recursive splitter while retaining `parent_article_id` and a header
prefix on every part.
- **Pros:** Aligns with legal units and ground-truth labels; preserves citations; works for
  both `Статья` and `Пункт` via ingest metadata.
- **Cons:** Variable chunk lengths; oversized articles still need a secondary split.

## Decision

Implement **both** strategies (RAID requirement):

| Output | Strategy | Role |
|--------|----------|------|
| `data/chunks_recursive.jsonl` | Option 1 | Baseline + chunk-size ablation |
| `data/chunks_article.jsonl` | Option 2 | Primary retrieval index |
| `data/chunks.jsonl` | Option 2 (default copy) | Used by `scripts/build_index.py` |

Code lives in `app/chunking.py`; CLI entrypoint is `make chunk` / `scripts/chunk_docs.py`.

## Consequences

- Default index uses article-aware chunks, so ground-truth can key off `article_id`.
- Large articles (definitions, long offence lists) become `_p0`, `_p1`, … parts without
  losing parent metadata.
- Ablation compares recursive sizes `{200,400,800,1600}` against article-aware on the
  same 15-question set; expect article-aware to win on recall and citation quality.
- Ingest format changes (new heading styles) are handled in `scripts/ingest_docs.py`, not
  in the chunker.
