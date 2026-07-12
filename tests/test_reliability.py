import pytest
import asyncio
from unittest.mock import AsyncMock
import httpx
from app.reliability import retry_with_backoff, run_with_timeout, ServiceTimeoutError


@pytest.mark.asyncio
async def test_retry_eventual_success():
    call_count = 0

    async def mock_unstable_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Transient network failure")
        return "Success!"

    result = await retry_with_backoff(
        async_func=mock_unstable_call,
        exceptions_to_retry=(ConnectionError,),
        max_retries=3,
        initial_delay=0.01,
        backoff_factor=1.5,
    )

    assert result == "Success!"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_on_max_attempts():
    call_count = 0

    async def mock_failing_call():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Persistent network failure")

    with pytest.raises(ConnectionError):
        await retry_with_backoff(
            async_func=mock_failing_call,
            exceptions_to_retry=(ConnectionError,),
            max_retries=3,
            initial_delay=0.01,
            backoff_factor=1.5,
        )

    assert call_count == 3


@pytest.mark.asyncio
async def test_timeout_bounds_execution():
    async def slow_call():
        await asyncio.sleep(0.5)
        return "Finished"

    with pytest.raises(ServiceTimeoutError):
        await run_with_timeout(async_func=slow_call, timeout_sec=0.05)


@pytest.mark.asyncio
async def test_retry_on_httpx_errors():
    call_count = 0

    async def flaky_http():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ConnectError("boom")
        return "ok"

    result = await retry_with_backoff(
        async_func=flaky_http,
        exceptions_to_retry=(httpx.HTTPError,),
        max_retries=3,
        initial_delay=0.01,
        backoff_factor=2.0,
    )
    assert result == "ok"
    assert call_count == 2
