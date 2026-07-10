import time
import uuid
import logging
import json
import sys
from collections import deque
from typing import Dict, Any, Optional
from contextvars import ContextVar

# ContextVar to hold the request_id across the thread context
request_id_var: ContextVar[str] = ContextVar("request_id", default="system")

class StructuredJsonFormatter(logging.Formatter):
    """
    Format logs as single-line JSON, including standard log fields
    plus the contextual request_id if available.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_var.get()
        }
        # Include extra attributes passed to the log
        if hasattr(record, "extra_info") and isinstance(record.extra_info, dict):
            log_data.update(record.extra_info)
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = StructuredJsonFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Initialize logging on import
setup_logging()
logger = logging.getLogger("rag_service")

class SimpleMetricsTracker:
    """
    Tracks recent requests in-memory to compute stats for the /metrics endpoint.
    Uses sliding windows (e.g. deque) for latency, cache hits, errors.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.cache_hits = deque(maxlen=window_size)
        self.retrieval_scores = deque(maxlen=window_size)
        self.total_tokens = 0
        self.total_requests = 0

    def record_request(
        self,
        latency_ms: float,
        is_error: bool,
        is_cache_hit: bool,
        tokens_used: int,
        top_retrieval_score: Optional[float] = None
    ):
        self.total_requests += 1
        self.latencies.append(latency_ms)
        self.errors.append(1.0 if is_error else 0.0)
        self.cache_hits.append(1.0 if is_cache_hit else 0.0)
        self.total_tokens += tokens_used
        if top_retrieval_score is not None:
            self.retrieval_scores.append(top_retrieval_score)

    def get_snapshot(self) -> Dict[str, Any]:
        lat_list = sorted(list(self.latencies))
        p50 = 0.0
        p95 = 0.0
        if lat_list:
            n = len(lat_list)
            p50 = lat_list[int(n * 0.5)]
            p95 = lat_list[int(n * 0.95)] if n > 1 else lat_list[-1]

        err_rate = sum(self.errors) / len(self.errors) if self.errors else 0.0
        hit_rate = sum(self.cache_hits) / len(self.cache_hits) if self.cache_hits else 0.0
        mean_score = sum(self.retrieval_scores) / len(self.retrieval_scores) if self.retrieval_scores else 0.0

        return {
            "total_requests": self.total_requests,
            "total_tokens_consumed": self.total_tokens,
            "window_size": self.window_size,
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "error_rate": round(err_rate, 4),
            "cache_hit_rate": round(hit_rate, 4),
            "mean_retrieval_score": round(mean_score, 4)
        }

metrics_tracker = SimpleMetricsTracker()
