import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.schemas import Chunk
from app.observability import logger
from app.reliability import DatabaseConnectionError

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

# Soft cap so the in-memory retrieval cache cannot grow unboundedly.
MAX_RETRIEVAL_CACHE_ENTRIES = int(os.getenv("RETRIEVAL_CACHE_MAX_ENTRIES", "2048"))


class VectorRetriever:
    def __init__(self, index_dir: str, chunks_path: str, model_name: str):
        self.index_dir = index_dir
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []
        self.model: Optional[Any] = None
        self.is_loaded = False
        self.last_embed_ms = 0.0
        self.last_search_ms = 0.0

        # In-memory retrieval cache: hash -> (chunks, top_score)
        self._retrieval_cache: Dict[str, Tuple[List[Chunk], float]] = {}

    def load(self):
        """
        Loads the embedding model, FAISS index, and chunk store from disk.
        Missing artifacts leave is_loaded=False so /ready can report not ready.
        """
        try:
            logger.info("Initializing embedding model...")
            if SentenceTransformer is not None:
                device = os.getenv("EMBEDDING_DEVICE", "cpu")
                self.model = SentenceTransformer(self.model_name, device=device)
            else:
                logger.warning("SentenceTransformer is not installed; running in mock mode.")

            logger.info(f"Loading chunks from {self.chunks_path}...")
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = [json.loads(line) for line in f]
            else:
                logger.warning(f"Chunks file not found at {self.chunks_path}")

            faiss_file = os.path.join(self.index_dir, "index.faiss")
            logger.info(f"Loading FAISS index from {faiss_file}...")
            if faiss is not None and os.path.exists(faiss_file) and self.chunks:
                self.index = faiss.read_index(faiss_file)
                self.is_loaded = True
                logger.info("FAISS index and chunks loaded successfully.")
            else:
                logger.warning(f"FAISS index file not found or FAISS package missing at {faiss_file}")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load retriever components: {str(e)}")
            self.is_loaded = False

    def clear_cache(self) -> None:
        self._retrieval_cache.clear()

    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        index_version: str,
        filters: Optional[Dict[str, Any]],
    ) -> str:
        filters_key = json.dumps(filters or {}, sort_keys=True, ensure_ascii=False)
        raw_key = f"{query}||{top_k}||{index_version}||{self.model_name}||{filters_key}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = getattr(chunk, key, None)
            if actual is None and hasattr(chunk, "model_dump"):
                actual = chunk.model_dump().get(key)
            if actual != expected:
                return False
        return True

    def _store_cache(self, cache_key: str, value: Tuple[List[Chunk], float]) -> None:
        if len(self._retrieval_cache) >= MAX_RETRIEVAL_CACHE_ENTRIES and cache_key not in self._retrieval_cache:
            # Drop an arbitrary oldest-ish entry (dict insertion order).
            self._retrieval_cache.pop(next(iter(self._retrieval_cache)))
        self._retrieval_cache[cache_key] = value

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        index_version: str = "v1",
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Tuple[List[Chunk], bool, float]:
        """
        Retrieves top_k chunks matching the query.
        Returns (chunks, cache_hit, top_score).
        """
        import time

        if not self.is_loaded:
            raise DatabaseConnectionError("Retriever database index is not loaded or unreachable.")

        cache_key = self._generate_cache_key(query, top_k, index_version, filters)
        if use_cache and cache_key in self._retrieval_cache:
            logger.info("Retrieval cache hit", extra={"extra_info": {"query": query, "cache_hit": True}})
            cached_chunks, top_score = self._retrieval_cache[cache_key]
            self.last_embed_ms = 0.0
            self.last_search_ms = 0.0
            return cached_chunks, True, top_score

        logger.info(
            "Retrieval cache miss. Querying vector database...",
            extra={"extra_info": {"query": query, "cache_hit": False}},
        )

        if self.model is None or self.index is None:
            logger.warning("Running retrieval mock fallback.")
            mock_chunks = [
                Chunk(
                    chunk_id=f"mock_{i}",
                    text=f"This is mock text chunk content for query: {query}",
                    source_file="mock_doc.pdf",
                    page=i + 1,
                )
                for i in range(min(top_k, len(self.chunks) if self.chunks else top_k))
            ]
            if use_cache:
                self._store_cache(cache_key, (mock_chunks, 1.0))
            return mock_chunks, False, 1.0

        try:
            t0 = time.perf_counter()
            query_vector = np.asarray(self.model.encode([query]), dtype=np.float32)
            if faiss is not None:
                faiss.normalize_L2(query_vector)
            self.last_embed_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            # Over-fetch when filters are applied so we can post-filter.
            search_k = top_k * 5 if filters else top_k
            scores, indices = self.index.search(query_vector, search_k)
            self.last_search_ms = (time.perf_counter() - t1) * 1000.0

            retrieved_chunks: List[Chunk] = []
            top_score = 0.0

            for rank, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk_data = self.chunks[idx]
                chunk = Chunk(
                    chunk_id=chunk_data.get("chunk_id", f"chk_{idx}"),
                    text=chunk_data.get("text", ""),
                    source_file=chunk_data.get("source_file", "unknown"),
                    page=chunk_data.get("page"),
                    section_title=chunk_data.get("section_title"),
                    char_start=chunk_data.get("char_start"),
                    char_end=chunk_data.get("char_end"),
                    document_version=chunk_data.get("document_version"),
                )
                if filters and not self._matches_filters(chunk, filters):
                    continue
                if not retrieved_chunks:
                    top_score = float(scores[0][rank])
                retrieved_chunks.append(chunk)
                if len(retrieved_chunks) >= top_k:
                    break

            if use_cache:
                self._store_cache(cache_key, (retrieved_chunks, top_score))
            return retrieved_chunks, False, top_score
        except DatabaseConnectionError:
            raise
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise DatabaseConnectionError(f"Database search failed: {str(e)}")
