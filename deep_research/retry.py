"""Retry helper for transient failures against local model servers (llama.cpp,
Ollama). A brief connection reset, timeout, or restart-in-progress 503
shouldn't fail an entire extraction run, verification pass, or report just
because one HTTP call landed in the wrong half-second -- but a genuine 4xx
(bad request, model not found) should still fail immediately rather than be
retried and hide the real problem.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")

_TRANSIENT_STATUS_CODES = {502, 503, 504}
_TRANSIENT_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS_CODES
    return False


async def with_retries(func: Callable[[], Awaitable[T]], attempts: int = 3, base_delay: float = 1.0) -> T:
    """Calls func() up to `attempts` times total, retrying only on transient
    connection/timeout/5xx errors with exponential backoff (1s, 2s, ...).
    Any other exception propagates on the first occurrence."""
    for attempt in range(attempts):
        try:
            return await func()
        except Exception as e:
            if not _is_transient(e) or attempt == attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2**attempt))
