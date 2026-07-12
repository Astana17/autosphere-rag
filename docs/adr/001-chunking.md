# ADR 001: Chunking Strategy for the Traffic-Law RAG Corpus

- **Status:** Approved
- **Date:** 2026-07-12
- **Author:** Founding ML Engineer

## Context

The corpus is Kazakhstan road traffic legislation: `road_traffic_rules.docx` (ПДД, numbered "Пункт N" clauses, no "Статья" wrapper), `road_traffic_law.docx` and `code_of_administrative_offences.docx` (both structured as `РАЗДЕЛ` → `Глава` → `Статья N`, with numbered sub-clauses inside each article), and `insurance_law.docx` (same `Статья N` structure).

Retrieving text chunks split at arbitrary character lengths cuts articles mid-sentence or mid-list item, which (a) drops the exact fine amount or deadline a numeric-fact question needs, and (b) makes citations point at a fragment instead of a real "Статья N" a human could look up. We need a chunking strategy that maximizes `recall@5` and keeps `chunk_id` → citation mapping exact.

## Options Considered

### Option 1: `RecursiveCharacterTextSplitter` (fixed size, required baseline)
500 characters, 50 overlap, domain-agnostic.
- **Pros:** trivial to implement, uniform chunk size, zero document-structure assumptions.
- **Cons:** on this corpus it produces 6,911 chunks (vs. 2,719 for the article-aware splitter) and routinely splits a `Статья` across 2-3 chunks at an arbitrary character boundary, so a fine amount can land in a different chunk than the article title that names the offense.

### Option 2: Custom article-boundary (section-aware) splitter
Regex-detect `РАЗДЕЛ N` / `Глава N` / `Статья N` as hard section boundaries; a `Статья` stays one chunk unless it exceeds 1,600 characters, in which case it's sub-split with 150-char overlap while keeping metadata pointing back to the parent article. A further refinement (below) splits glossary-style sections by defined term instead.
- **Pros:** one chunk per citable unit in the common case; `chunk_id`/`section_title` map directly to a real "Статья N" a human can verify; 2,719 chunks total, comfortably under the 15-minute CPU index-build budget.
- **Cons:** requires document-structure-aware regex, which needs revisiting if a new corpus doesn't follow `Статья`/`Пункт` conventions.

**Shipped-then-fixed pitfall #1:** the first version of this regex additionally matched *any* bare numbered line (`\d+\.\s+.*`) as a new section boundary, with no state tracking. That's correct for `road_traffic_rules.docx` (which has no `Статья` wrapper — top-level `Пункт N.` numbering *is* the section boundary), but wrong for the three `Статья`-structured documents, where it matched every numbered sub-clause *inside* an article too (e.g. "1. Лицо, совершившее административное правонарушение...") and fragmented single articles into dozens of one-sentence chunks — nearly doubling the corpus to 5,246 chunks and producing 93 duplicate `chunk_id`s (articles reuse "1.", "2." sub-numbering across different chapters). The fix: bare numbered lines only start a new section when the current section is *not already inside a `Статья` block* (checked via `current_title.lower().startswith("статья")`); ПДД (no `Статья` markers at all) is unaffected by this guard and still chunks per-`Пункт` as before. This is also why the two document families needed one shared function rather than two — the guard condition is the only thing that has to differ per structure.

**Shipped-then-fixed pitfall #2 (glossary chunking):** even with per-`Статья`/`Пункт` chunks, one section per document is itself a glossary — ПДД's "Пункт 2. Основные понятия" and the law's "Статья 1" each list dozens of unrelated defined terms ("автомагистраль", "уступить дорогу", "пешеходный переход", ...) as one continuous block, length-capped into ~13 arbitrary ~1,600-char pieces that each mash together 15-20 unrelated terms. A query about one specific term had to compete against a dozen near-identical siblings with diluted relevance signal — this was the direct cause of 3 ground-truth questions failing under *every* retrieval backend (recall@5 stuck at 0.767 with the fusion strategy already in place). Fix: `split_glossary_by_term` detects entries matching `N) термин – определение` and splits the section into one chunk per term instead of by character count. First attempt at this fix over-triggered on any article enumerating "N) ..." items (e.g. Статья 73 lists driving-licence subcategories the same way), which is a legitimate single-topic list, not a glossary — one entry silently absorbed thousands of characters of unrelated trailing article text because nothing capped it once glossary-mode was chosen. Fixed by (a) gating glossary-mode on the literal intro phrase "используются следующие основные понятия", which only the two real glossary sections contain, and (b) still running each term-chunk through the ordinary length cap as a backstop. Recall@5 with hybrid RRF went from 0.767 → 0.967 on the same ground truth, with zero retrieval-code changes — proof that the earlier plateau was a chunking problem wearing a retrieval-quality costume.

## Decision

**Custom article-boundary splitter with glossary-term sub-splitting (Option 2)**, as the production strategy in `scripts/build_index.py::chunk_documents_by_article` + `split_glossary_by_term`. `RecursiveCharacterTextSplitter` (Option 1, `chunk_documents_recursive`) is kept and runs in `01_ingestion.ipynb` to satisfy the "at least two strategies, both run and counted" requirement, but is not what the service indexes.

## Consequences

- **Retrieval quality:** measured on the 15-question ground truth (see `02_ablation.ipynb` / README §6): TF-IDF 0.500, FAISS embeddings 0.833, hybrid RRF **0.967** (14.5/15). We did not re-run the ablation against Option 1's chunks to get a controlled "chunking-only" recall delta — that would be the natural follow-up ablation, but pitfalls #1 and #2 above already demonstrate the mechanism directly: a chunk whose `chunk_id` isn't unique, whose article got split mid-clause, or whose relevance signal is diluted by unrelated content can't be retrieved or cited correctly regardless of which retriever sits behind it.
- **Citation accuracy:** the LLM receives one coherent `Статья`/`Пункт`/term per context block with its real section title, so `sources[].section_title` in `AnswerResponse` names an actual, checkable article or defined term instead of an arbitrary character span.
- **Known remaining weakness:** the one ground-truth question still not fully solved (query 15, recall 0.5) needs two chunks from two different documents (ПДД §12 + КоАП Статья 608) combined — each retrieval backend ranks chunks independently, so cross-document multi-chunk questions remain structurally harder than single-chunk lookups. That's a retrieval/generation-orchestration question (e.g., decomposing the query into sub-questions), not a chunking one, and was out of scope here.
- **Complexity:** the ingestion script now encodes two document-structure assumptions (numbered sub-clauses live inside `Статья` blocks; glossary sections are identified by a specific intro phrase) that must be revisited if a future corpus doesn't follow either convention.
