import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from app.schemas import Chunk
from app.observability import logger
from app.reliability import DatabaseConnectionError

# Try importing FAISS and SentenceTransformers. Since packages aren't installed yet,
# we should structure imports so that syntax is valid and it can be run or stubbed easily.
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

class VectorRetriever:
    def __init__(self, index_dir: str, chunks_path: str, model_name: str):
        self.index_dir = index_dir
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.index: Optional[Any] = None
        self.chunks: List[Dict[str, Any]] = []
        self.model: Optional[Any] = None
        self.is_loaded = False
        
        # In-memory retrieval cache: hash -> List[Chunk]
        self._retrieval_cache: Dict[str, Tuple[List[Chunk], float]] = {}

    def load(self):
        """
        Loads the embedding model, the FAISS index, and the document chunks store from disk.
        Does not raise exception if index files are missing to allow server startup in non-ready state.
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
            if faiss is not None and os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
                self.is_loaded = True
                logger.info("FAISS index and chunks loaded successfully.")
            else:
                logger.warning(f"FAISS index file not found or FAISS package missing at {faiss_file}")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load retriever components: {str(e)}")
            self.is_loaded = False
            # We don't crash, but set is_loaded=False. Readiness check will expose this.

    def _generate_cache_key(self, query: str, top_k: int, index_version: str) -> str:
        raw_key = f"{query}||{top_k}||{index_version}||{self.model_name}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

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

        logger.info("Retrieval cache miss. Querying vector database...", extra={"extra_info": {"query": query, "cache_hit": False}})
        
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
            query_vector = self.model.encode([query])
            
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
