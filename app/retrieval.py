import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from app.schemas import Chunk
from app.observability import logger
from app.reliability import DatabaseConnectionError

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Try importing FAISS and SentenceTransformers. Since packages aren't installed yet,
# we should structure imports so that syntax is valid and it can be run or stubbed easily.
try:
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    np = None
    TfidfVectorizer = None
    cosine_similarity = None
    SentenceTransformer = None

RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant
LEXICAL_RRF_WEIGHT = 0.3
EMBEDDING_RRF_WEIGHT = 0.7


class VectorRetriever:
    def __init__(self, index_dir: str, chunks_path: str, model_name: str):
        self.index_dir = index_dir
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []
        self.model: Optional[Any] = None
        # "lexical" (TF-IDF only), "embeddings" (FAISS only), or "hybrid" (RRF of both).
        self.retrieval_backend = os.getenv("RETRIEVAL_BACKEND", "hybrid").lower()
        self.tfidf_vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None
        self.is_loaded = False

        # In-memory retrieval cache: hash -> (chunks, top_score)
        self._retrieval_cache: Dict[str, Tuple[List[Chunk], float]] = {}

    def _resolve_model_path(self) -> str:
        if os.path.exists(self.model_name):
            return self.model_name

        if "/" not in self.model_name:
            return self.model_name

        cache_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_cache_name = "models--" + self.model_name.replace("/", "--")
        snapshots_dir = cache_home / "hub" / model_cache_name / "snapshots"
        if not snapshots_dir.exists():
            return self.model_name

        snapshots = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )
        return str(snapshots[0]) if snapshots else self.model_name

    def load(self):
        """
        Loads the document chunks store, and whichever of (TF-IDF, embedding model + FAISS
        index) the configured retrieval_backend needs. Does not raise on missing files so the
        server can still start in a non-ready state; readiness is exposed via /ready.
        """
        try:
            logger.info(f"Loading chunks from {self.chunks_path}...")
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = [json.loads(line) for line in f]
            else:
                logger.warning(f"Chunks file not found at {self.chunks_path}")

            needs_lexical = self.retrieval_backend in ("lexical", "hybrid")
            needs_embeddings = self.retrieval_backend in ("embeddings", "hybrid")

            lexical_ready = False
            if needs_lexical and self.chunks:
                if TfidfVectorizer is None or cosine_similarity is None:
                    logger.warning("scikit-learn is not installed; lexical retrieval unavailable.")
                else:
                    logger.info("Building runtime TF-IDF matrix for lexical retrieval...")
                    self.tfidf_vectorizer = TfidfVectorizer(
                        lowercase=True,
                        max_features=50000,
                        ngram_range=(1, 2)
                    )
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                        [chunk.get("text", "") for chunk in self.chunks]
                    )
                    lexical_ready = True

            embeddings_ready = False
            if needs_embeddings:
                faiss_file = os.path.join(self.index_dir, "index.faiss")
                if faiss is not None and os.path.exists(faiss_file):
                    logger.info(f"Loading FAISS index from {faiss_file}...")
                    self.index = faiss.read_index(faiss_file)
                else:
                    logger.warning(f"FAISS index file not found or FAISS package missing at {faiss_file}")

                if SentenceTransformer is not None:
                    logger.info("Initializing embedding model...")
                    model_path = self._resolve_model_path()
                    try:
                        self.model = SentenceTransformer(model_path, local_files_only=True)
                    except TypeError:
                        self.model = SentenceTransformer(model_path)
                else:
                    logger.warning("SentenceTransformer is not installed; embedding retrieval unavailable.")

                embeddings_ready = self.index is not None and self.model is not None

            if self.retrieval_backend == "lexical":
                self.is_loaded = lexical_ready
            elif self.retrieval_backend == "embeddings":
                self.is_loaded = embeddings_ready
            else:
                self.is_loaded = lexical_ready and embeddings_ready

            if self.is_loaded:
                logger.info(f"Retriever loaded successfully in '{self.retrieval_backend}' mode.")
        except Exception as e:
            logger.error(f"Failed to load retriever components: {str(e)}")
            self.is_loaded = False
            # We don't crash, but set is_loaded=False. Readiness check will expose this.

    def _generate_cache_key(self, query: str, top_k: int, index_version: str) -> str:
        raw_key = f"{query}||{top_k}||{index_version}||{self.model_name}||{self.retrieval_backend}||retrieval_v3"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _to_chunk(self, index: int) -> Chunk:
        chunk_data = self.chunks[index]
        return Chunk(
            chunk_id=chunk_data.get("chunk_id", f"chk_{index}"),
            text=chunk_data.get("text", ""),
            source_file=chunk_data.get("source_file", "unknown"),
            page=chunk_data.get("page"),
            section_title=chunk_data.get("section_title"),
            char_start=chunk_data.get("char_start"),
            char_end=chunk_data.get("char_end"),
            document_version=chunk_data.get("document_version")
        )

    def _lexical_ranked_indices(self, query: str, limit: int) -> List[int]:
        query_vector = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        return [int(i) for i in scores.argsort()[::-1][:limit]]

    def _embedding_ranked_indices(self, query: str, limit: int) -> List[int]:
        query_vector = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        query_vector = np.array(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)
        limit = min(limit, len(self.chunks))
        _, indices = self.index.search(query_vector, limit)
        return [int(i) for i in indices[0] if i >= 0]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[int]], weights: Optional[List[float]] = None, k: int = RRF_K
    ) -> List[Tuple[int, float]]:
        """
        Combines multiple rankings by rank position rather than raw score, so TF-IDF cosine
        similarity and FAISS inner-product scores (different scales) can be fused fairly.

        Weighted, not plain unweighted RRF: unweighted RRF assumes every signal is equally
        trustworthy for every query, which doesn't hold here. On a live colloquial query
        ("я выпил алкоголь за рулем, если остановят") TF-IDF had no matching vocabulary and
        produced near-noise rankings, yet a chunk that was merely mediocre in *both* signals
        (lexical rank 12, embedding rank 39) outscored the chunk embeddings ranked #1 (lexical
        rank 109) under equal weights -- unweighted RRF rewards cross-signal agreement over
        single-signal excellence. Weighting embeddings higher reflects that it's the stronger,
        more reliable signal on this corpus generally (embeddings-only recall@5 0.833 vs.
        lexical-only 0.500 on the ground truth, see README §6), not just for this one query.
        """
        weights = weights or [1.0] * len(ranked_lists)
        scores: Dict[int, float] = {}
        for ranked, weight in zip(ranked_lists, weights):
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 5, index_version: str = "v1") -> Tuple[List[Chunk], bool, float]:
        """
        Retrieves top_k chunks matching the query.
        Returns:
            - List[Chunk]: Retrieved chunks
            - bool: Cache hit flag
            - float: Top chunk retrieval score (or 0.0)
        """
        if not self.is_loaded:
            raise DatabaseConnectionError("Retriever database index is not loaded or unreachable.")

        cache_key = self._generate_cache_key(query, top_k, index_version)
        if cache_key in self._retrieval_cache:
            logger.info("Retrieval cache hit", extra={"extra_info": {"query": query, "cache_hit": True}})
            cached_chunks, top_score = self._retrieval_cache[cache_key]
            return cached_chunks, True, top_score

        logger.info(
            f"Retrieval cache miss. Querying {self.retrieval_backend} backend...",
            extra={"extra_info": {"query": query, "cache_hit": False, "retrieval_backend": self.retrieval_backend}}
        )

        if not self.chunks:
            mock_chunks = [
                Chunk(
                    chunk_id=f"mock_{i}",
                    text=f"This is mock text chunk content for query: {query}",
                    source_file="mock_doc.pdf",
                    page=i + 1
                ) for i in range(top_k)
            ]
            self._retrieval_cache[cache_key] = (mock_chunks, 1.0)
            return mock_chunks, False, 1.0

        try:
            if self.retrieval_backend == "lexical":
                if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                    raise DatabaseConnectionError("Lexical retrieval backend is not loaded.")
                ranked = self._lexical_ranked_indices(query, top_k)
                query_vector = self.tfidf_vectorizer.transform([query])
                scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
                retrieved_chunks = [self._to_chunk(i) for i in ranked]
                top_score = float(scores[ranked[0]]) if ranked else 0.0

            elif self.retrieval_backend == "embeddings":
                if self.model is None or self.index is None:
                    raise DatabaseConnectionError("Embedding retrieval backend is not loaded.")
                query_vector = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
                query_vector = np.array(query_vector, dtype=np.float32)
                faiss.normalize_L2(query_vector)
                scores, indices = self.index.search(query_vector, top_k)
                retrieved_chunks = [
                    self._to_chunk(int(idx)) for idx in indices[0] if 0 <= idx < len(self.chunks)
                ]
                top_score = float(scores[0][0]) if len(scores) and len(scores[0]) else 0.0

            else:  # hybrid: Reciprocal Rank Fusion of lexical + embedding rankings
                if self.tfidf_vectorizer is None or self.model is None or self.index is None:
                    raise DatabaseConnectionError("Hybrid retrieval backend is not fully loaded.")
                # Rank the FULL corpus in each signal, not a truncated top-N candidate pool.
                # A truncated pool silently zeroes out a chunk's RRF contribution from
                # whichever signal doesn't rank it inside the cutoff -- which punishes a
                # chunk that's #1 in one signal but merely mediocre (not absent) in the
                # other, in favor of a chunk that's moderately-ranked in *both* truncated
                # lists. Concretely: for a colloquial query like "я выпил алкоголь за рулем,
                # если остановят" (no legal vocabulary), embeddings correctly ranked the
                # right statute #1, but TF-IDF ranked it ~109th -- outside a 100-item pool
                # -- so an unrelated chunk that merely limped into both truncated lists
                # (ranks ~12 and ~39) won on total RRF score. This corpus is small enough
                # (a few thousand chunks) that ranking it in full costs nothing measurable.
                # top_k only truncates the final *fused* list, so retrieve(q, top_k=5) stays
                # a strict prefix of retrieve(q, top_k=10) regardless of corpus size.
                candidate_pool = len(self.chunks)
                lexical_ranked = self._lexical_ranked_indices(query, candidate_pool)
                embedding_ranked = self._embedding_ranked_indices(query, candidate_pool)
                fused = self._reciprocal_rank_fusion(
                    [lexical_ranked, embedding_ranked],
                    weights=[LEXICAL_RRF_WEIGHT, EMBEDDING_RRF_WEIGHT]
                )[:top_k]
                retrieved_chunks = [self._to_chunk(idx) for idx, _ in fused]
                top_score = float(fused[0][1]) if fused else 0.0

            self._retrieval_cache[cache_key] = (retrieved_chunks, top_score)
            return retrieved_chunks, False, top_score
        except DatabaseConnectionError:
            raise
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise DatabaseConnectionError(f"Database search failed: {str(e)}")
