import asyncio
import time
from typing import Callable, Any, TypeVar, Tuple
from app.observability import logger

T = TypeVar("T")

class ServiceTimeoutError(Exception):
    """Raised when an external API or operation exceeds the allocated time limit."""
    pass

class DatabaseConnectionError(Exception):
    """Raised when the vector database cannot be searched or is unreachable."""
    pass

async def run_with_timeout(async_func: Callable[..., Any], timeout_sec: float, *args, **kwargs) -> Any:
    """
    Executes an async function, raising a ServiceTimeoutError if it exceeds timeout_sec.
    """
    try:
        return await asyncio.wait_for(async_func(*args, **kwargs), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.error(
            f"Operation timed out after {timeout_sec} seconds.",
            extra={"extra_info": {"timeout_seconds": timeout_sec}}
        )
        raise ServiceTimeoutError("Operation timed out while communicating with external services.")

async def retry_with_backoff(
    async_func: Callable[..., T],
    exceptions_to_retry: Tuple[type, ...],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> T:
    """
    Retries an async function upon catching specified exceptions.
    Applies exponential backoff (e.g. 1s -> 2s -> 4s).
    """
    delay = initial_delay
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return await async_func(*args, **kwargs)
        except exceptions_to_retry as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed. Retrying in {delay}s. Error: {str(e)}",
                extra={"extra_info": {"attempt": attempt, "error": str(e)}}
            )
            if attempt == max_retries:
                break
            await asyncio.sleep(delay)
            delay *= backoff_factor
            
    logger.error(
        f"All {max_retries} retry attempts failed.",
        extra={"extra_info": {"max_retries": max_retries}}
    )
    raise last_exception if last_exception else RuntimeError("Retry failed")
