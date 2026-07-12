# ADR 002: Retrieval Cache Invalidation Strategy

- **Status:** Approved
- **Date:** 2026-07-13
- **Author:** Founding ML Engineer

## Context

The online path embeds each query and searches FAISS on every `/ask`. Repeated
identical questions (internal tools, eval loops, demos) waste CPU on embedding.
We already have an on-disk **embedding cache** for index builds; we also need a
cheap online layer and a documented invalidation story (RAID §7).

A full **response cache** keyed only by `hash(question)` would also cache LLM
output, but that couples invalidation to prompt version, model name, and index
content, and risks serving stale grounded answers after a re-index.

## Options Considered

### Option 1: Response cache (`hash(question)` → AnswerResponse)
- **Pros:** Largest latency win (skips retrieval + LLM).
- **Cons:** Must invalidate on index rebuild, prompt version bump, and model
  change; stale legal answers are high risk for this corpus.

### Option 2: In-memory retrieval cache
Key: `hash(query || top_k || index_version || model_name || filters)`.
- **Pros:** Skips re-embedding and FAISS search; still runs generation so answers
  stay tied to the current prompt; trivial to implement; hit rate is measurable.
- **Cons:** Process-local (lost on restart); unbounded growth unless capped.

### Option 3: No online cache (embedding disk cache only)
- **Pros:** Simplest.
- **Cons:** Fails RAID “measure cache effect on p95” for the serving path.

## Decision

Use **Option 2** (retrieval cache) plus the existing on-disk embedding cache for
index builds. Do **not** ship a response cache for this prototype.

**Invalidation rules:**
1. **Process restart** — cache is empty (in-memory dict).
2. **Index rebuild** — bump `index_version` (default `"v1"`) or restart the
   service after `make index`; cache keys include `index_version` + model name.
3. **Soft size cap** — `RETRIEVAL_CACHE_MAX_ENTRIES` (default 2048); oldest
   insertion is evicted when full.
4. **Manual** — `retriever.clear_cache()` (used by `scripts/measure_cache.py`).

Measure impact with `make measure-cache` (50 GT questions, cold then warm pass).

## Consequences

- p95 retrieval latency drops sharply on repeated queries (see README §7).
- End-to-end `/ask` latency remains LLM-bound on cache hits unless a response
  cache is added later with stricter invalidation.
- Operators must restart (or bump `index_version`) after rebuilding FAISS so
  cached chunk lists cannot point at a previous index layout.
