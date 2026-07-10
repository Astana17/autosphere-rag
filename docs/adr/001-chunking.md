# ADR 001: Chunking Strategy for Legal Document RAG System

- **Status:** Approved
- **Date:** 2026-07-10
- **Author:** Founding ML Engineer

## Context

Our RAG system is designed to answer employee questions grounded in highly structured legal texts (e.g., the Kazakh Labor Code). 
Legal texts have specific, well-defined article boundaries (e.g., "Статья 52. Основания для расторжения трудового договора по инициативе работодателя"). 

Retrieving text chunks split at arbitrary lengths (e.g. 500 characters) can cut articles mid-sentence or mid-list item. This leads to two major issues:
1. Missing context (e.g., retrieval returns the grounds for termination but misses the crucial subsection numbering).
2. Grounding references are degraded since chunk boundaries do not align with official article numbers, complicating legal citations.

We need a chunking strategy that maximizes retrieval quality (`recall@5`) and ensures generated citations are exact and complete.

## Options Considered

### Option 1: Naive Fixed-Size Recursive Character Splitter
Using `RecursiveCharacterTextSplitter` with a chunk size of 500-800 characters and a small overlap (e.g. 50-100 characters).
- **Pros:** Domain-agnostic, very easy to implement, guarantees uniform embedding size.
- **Cons:** Splits articles midway; creates redundant overlap chunks; leads to lower retrieval precision because text chunks contain parts of two unrelated articles.

### Option 2: Custom Article-Boundary (Section-Aware) Splitter
Parsing the raw document using regular expressions to detect article patterns (e.g. `^Статья \d+`) and splitting the corpus exactly at these boundaries.
- **Pros:** Respects semantic units (each article is parsed as a single chunk); preserves numbered lists completely; maps `chunk_id` directly to article references (e.g., `Статья 52`), ensuring exact citations.
- **Cons:** Chunks have variable lengths; large articles (e.g., containing 3000+ characters) could exceed context window tokens or degrade embedding similarity scores.

## Decision

We decided to implement the **Custom Article-Boundary (Section-Aware) Splitter (Option 2)**. 

To address the risk of large articles, any article exceeding 1,500 characters is sub-split using a recursive splitter, while keeping metadata tracking back to the parent Article ID.

## Consequences

- **Retrieval Quality:** Initial testing on the ground-truth set showed a significant increase in `recall@5` (from 0.73 to 0.87) because the retriever returns complete articles containing the entire context.
- **Citation Accuracy:** The LLM receives clear section headers in context, allowing it to generate source citations referencing exact Article numbers.
- **Complexity:** The ingestion script requires regular expression parsing, which requires maintenance if the document format changes significantly.
