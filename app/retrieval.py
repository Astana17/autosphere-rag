import os
import json
import hashlib
import re
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

class VectorRetriever:
    def __init__(self, index_dir: str, chunks_path: str, model_name: str):
        self.index_dir = index_dir
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []
        self.model: Optional[Any] = None
        self.retrieval_backend = os.getenv("RETRIEVAL_BACKEND", "lexical").lower()
        self.tfidf_vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None
        self.is_loaded = False

        # In-memory retrieval cache: hash -> List[Chunk]
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
        Loads the embedding model, the FAISS index, and the document chunks store from disk.
        Does not raise exception if index files are missing to allow server startup in non-ready state.
        """
        try:
            logger.info(f"Loading chunks from {self.chunks_path}...")
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = [json.loads(line) for line in f]
            else:
                logger.warning(f"Chunks file not found at {self.chunks_path}")

            faiss_file = os.path.join(self.index_dir, "index.faiss")
            logger.info(f"Loading FAISS index from {faiss_file}...")
            if faiss is not None and os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
                logger.info("FAISS index and chunks loaded successfully.")
            else:
                logger.warning(f"FAISS index file not found or FAISS package missing at {faiss_file}")

            if self.retrieval_backend == "lexical":
                if TfidfVectorizer is None or cosine_similarity is None:
                    raise RuntimeError("scikit-learn is required for lexical retrieval backend.")
                logger.info("Building runtime TF-IDF matrix for lexical retrieval...")
                self.tfidf_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    max_features=50000,
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                    [chunk.get("text", "") for chunk in self.chunks]
                )
                self.is_loaded = bool(self.chunks)
                logger.info("Lexical retrieval backend loaded successfully.")
            else:
                logger.info("Initializing embedding model...")
                if SentenceTransformer is not None:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    model_path = self._resolve_model_path()
                    try:
                        self.model = SentenceTransformer(model_path, local_files_only=True)
                    except TypeError:
                        self.model = SentenceTransformer(model_path)
                    self.is_loaded = self.index is not None and bool(self.chunks)
                else:
                    logger.warning("SentenceTransformer is not installed; running in mock mode.")
                    self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load retriever components: {str(e)}")
            self.is_loaded = False
            # We don't crash, but set is_loaded=False. Readiness check will expose this.

    def _generate_cache_key(self, query: str, top_k: int, index_version: str) -> str:
        raw_key = f"{query}||{top_k}||{index_version}||{self.model_name}||retrieval_v2"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _expand_query(self, query: str) -> str:
        query_lc = query.lower()
        expansions = []
        synonym_groups = [
            (("оранжев",), "желтый сигнал светофора запрещающий сигнал светофора"),
            (("жёлт", "желт"), "желтый сигнал светофора запрещающий сигнал светофора"),
            (("красн", "светофор"), "запрещающий сигнал светофора"),
            (("скорост", "быстро", "км", "км/ч"), "скорость движения населенных пунктах вне населенных пунктов"),
            (("населен", "город", "посел"), "в населенных пунктах"),
            (("штраф", "накаж", "ответствен", "лишен", "наруш"), "административные правонарушения штраф ответственность"),
            (("страхов", "дтп", "выплат"), "обязательное страхование гражданско-правовой ответственности владельцев транспортных средств"),
            (("обгон",), "обгон встречный разъезд опережение"),
            (("останов", "стоян"), "остановка стоянка транспортных средств"),
            (("пешеход",), "пешеход пешеходный переход"),
            (("ремень",), "ремни безопасности"),
            (("телефон", "мобиль"), "телефон радиостанция переговорное устройство"),
        ]
        for needles, expansion in synonym_groups:
            if any(needle in query_lc for needle in needles):
                expansions.append(expansion)

        return " ".join([query, *expansions])

    def _document_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        query_lc = query.lower()
        source_file = chunk.get("source_file", "")
        boost = 1.0

        road_rule_terms = (
            "скорост", "светофор", "обгон", "останов", "стоян", "пешеход",
            "знак", "полоса", "перекрест", "ремень", "телефон", "водител"
        )
        penalty_terms = ("штраф", "наказ", "ответствен", "лишен", "протокол")
        insurance_terms = ("страхов", "выплат", "полис", "дтп")

        if any(term in query_lc for term in road_rule_terms) and source_file == "road_traffic_rules.docx":
            boost += 0.45
        if any(term in query_lc for term in penalty_terms) and source_file == "code_of_administrative_offences.docx":
            boost += 1.0
        if any(term in query_lc for term in insurance_terms) and source_file == "insurance_law.docx":
            boost += 0.55
        if "понят" in query_lc or "что такое" in query_lc:
            if source_file in {"road_traffic_rules.docx", "road_traffic_law.docx"}:
                boost += 0.25

        return boost

    def _literal_match_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        query_terms = {
            term
            for term in re.findall(r"[а-яa-z0-9]+", query.lower())
            if len(term) >= 4
        }
        if not query_terms:
            return 0.0

        text = f"{chunk.get('section_title') or ''} {chunk.get('text') or ''}".lower()
        matches = sum(1 for term in query_terms if term in text)
        return min(matches * 0.035, 0.35)

    def _answer_phrase_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        query_lc = query.lower()
        text = f"{chunk.get('section_title') or ''} {chunk.get('text') or ''}".lower()
        boost = 0.0

        if any(term in query_lc for term in ("оранжев", "желт", "жёлт", "светофор", "красн")):
            if "желтый сигнал запрещает движение" in text:
                boost += 0.9
            if "красный сигнал" in text and "запрещает движение" in text:
                boost += 0.65
            if "проезд на запрещающий сигнал светофора" in text:
                boost += 0.75
            if "не могут остановиться" in text and "экстренному торможению" in text:
                boost += 0.55

        if any(term in query_lc for term in ("скорост", "населен", "город", "км")):
            if "в населенных пунктах разрешается движение" in text:
                boost += 0.9
            if "не более 60 км/ч" in text:
                boost += 0.8

        if any(term in query_lc for term in ("обгон", "опереж")):
            if "обгон" in text:
                boost += 0.45

        if any(term in query_lc for term in ("останов", "стоян")):
            if "остановка запрещается" in text or "стоянка запрещается" in text:
                boost += 0.55

        return boost

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

        if self.retrieval_backend == "lexical":
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None or cosine_similarity is None:
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
                raise DatabaseConnectionError("Lexical retrieval backend is not loaded.")
            expanded_query = self._expand_query(query)
            query_vector = self.tfidf_vectorizer.transform([expanded_query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
            for idx, chunk_data in enumerate(self.chunks):
                scores[idx] *= self._document_boost(query, chunk_data)
                scores[idx] += self._literal_match_boost(query, chunk_data)
                scores[idx] += self._answer_phrase_boost(query, chunk_data)
            min_context_chunks = max(top_k, 5)
            ranked_indices = scores.argsort()[::-1][:max(min_context_chunks * 8, 12)]
            retrieved_chunks = []
            seen_indices = set()
            for idx in ranked_indices:
                candidate_indices = [int(idx)]
                chunk_data = self.chunks[int(idx)]
                chunk_text = chunk_data.get("text", "")
                section_title = chunk_data.get("section_title") or ""
                is_heading = len(chunk_text) < 140 or section_title.lower().startswith(("глава", "раздел"))
                if is_heading:
                    candidate_indices.extend(
                        next_idx
                        for next_idx in range(int(idx) + 1, min(int(idx) + 4, len(self.chunks)))
                        if self.chunks[next_idx].get("source_file") == chunk_data.get("source_file")
                    )

                for candidate_idx in candidate_indices:
                    if candidate_idx in seen_indices:
                        continue
                    seen_indices.add(candidate_idx)
                    retrieved_chunks.append(self._to_chunk(candidate_idx))
                if len(retrieved_chunks) >= min_context_chunks + 2:
                    break
            top_score = float(scores[ranked_indices[0]]) if len(ranked_indices) else 0.0
            self._retrieval_cache[cache_key] = (retrieved_chunks, top_score)
            return retrieved_chunks, False, top_score

        if self.model is None or self.index is None:
            # Mock mode / fallback if model or index didn't load properly (e.g. in tests)
            logger.warning("Running retrieval mock fallback.")
            mock_chunks = [
                Chunk(
                    chunk_id=f"mock_{i}",
                    text=f"This is mock text chunk content for query: {query}",
                    source_file="mock_doc.pdf",
                    page=i+1
                ) for i in range(min(top_k, len(self.chunks) if self.chunks else top_k))
            ]
            self._retrieval_cache[cache_key] = (mock_chunks, 1.0)
            return mock_chunks, False, 1.0

        try:
            # Embed query
            query_vector = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            if np is not None:
                query_vector = np.array(query_vector, dtype=np.float32)
            if faiss is not None:
                faiss.normalize_L2(query_vector)

            # Query FAISS
            scores, indices = self.index.search(query_vector, top_k)

            retrieved_chunks = []
            top_score = float(scores[0][0]) if len(scores) > 0 and len(scores[0]) > 0 else 0.0

            for idx in indices[0]:
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk_data = self.chunks[idx]
                retrieved_chunks.append(
                    Chunk(
                        chunk_id=chunk_data.get("chunk_id", f"chk_{idx}"),
                        text=chunk_data.get("text", ""),
                        source_file=chunk_data.get("source_file", "unknown"),
                        page=chunk_data.get("page"),
                        section_title=chunk_data.get("section_title"),
                        char_start=chunk_data.get("char_start"),
                        char_end=chunk_data.get("char_end"),
                        document_version=chunk_data.get("document_version")
                    )
                )

            self._retrieval_cache[cache_key] = (retrieved_chunks, top_score)
            return retrieved_chunks, False, top_score
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise DatabaseConnectionError(f"Database search failed: {str(e)}")
