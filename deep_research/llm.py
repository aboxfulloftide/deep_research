import asyncio
import contextlib
import json
import re
import zlib
from typing import AsyncIterator

import asyncpg
import httpx

from deep_research.config import Config
from deep_research.retry import with_retries

# Pattern to strip reasoning/thinking tags from models like qwen3, deepseek
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
# Also catch unclosed think tags (model sometimes forgets to close)
_THINK_UNCLOSED_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)

# One shared semaphore per LLM endpoint, sized to that server's real
# concurrent-request capacity (llama.cpp's own --parallel N) -- every caller
# across the whole system (Extra Research, the interactive agent, KB
# extraction/verification/resolution) shares this same budget instead of
# each independently-sized layer of concurrency (e.g. verifying N claims at
# once, each of which may itself resolve M new claims concurrently)
# multiplying against the others well past the server's real capacity.
# Acquired only around the single network call itself, never held across a
# caller's other awaits, so a caller nested inside another caller's own
# semaphore-guarded section can still safely acquire it without deadlocking.
_semaphores_by_base_url: dict[str, asyncio.Semaphore] = {}
_semaphores_lock = asyncio.Lock()


async def _endpoint_semaphore(base_url: str, max_concurrent: int) -> asyncio.Semaphore:
    async with _semaphores_lock:
        semaphore = _semaphores_by_base_url.get(base_url)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max_concurrent)
            _semaphores_by_base_url[base_url] = semaphore
        return semaphore


# Cross-process counterpart to the in-process semaphore above -- discovered
# live: the main web service, the nightly verification cron job (its own
# separate `cli.kb verify-unverified` process), and an ad-hoc diagnostic
# script each independently respecting max_concurrent_requests *within
# themselves* still let their combined real demand on the one shared
# llama.cpp instance multiply past its actual --parallel N capacity, since
# each process gets its own fresh in-process semaphore. Implemented as N
# Postgres advisory-lock slots (0..max_concurrent-1) per endpoint, each
# backed by a dedicated held-open connection -- advisory locks are tied to
# that connection's session, so Postgres releases the slot automatically if
# the holding process crashes or the connection drops, with no manual
# cleanup needed. Best-effort throughout: any failure (Postgres unreachable,
# no DSN configured) falls back to no cross-process protection rather than
# blocking or failing the real LLM call -- the in-process semaphore above
# still applies regardless.
_pg_pools_by_dsn: dict[str, asyncpg.Pool] = {}
_pg_pools_lock = asyncio.Lock()


def _lock_namespace_for_endpoint(base_url: str) -> int:
    """A stable, cross-process-consistent 32-bit lock namespace derived from
    the endpoint URL, so two genuinely different LLM servers (e.g.
    extraction and verification pointing at different backends) never share
    one throttling budget. Python's built-in hash() is NOT safe here -- it's
    randomized per process by default (PYTHONHASHSEED), so two different
    processes would compute two different keys for the very same real
    endpoint, defeating cross-process coordination entirely. Distinct from
    kb/jobs.py's GPU_WORKER_ADVISORY_LOCK (734918, itself far too small a
    bigint to collide with any of these hash-derived 32-bit namespaces
    packed into the high bits of a two-key advisory lock)."""
    return zlib.crc32(base_url.encode()) & 0x7FFFFFFF


async def _advisory_lock_pool(dsn: str, max_concurrent: int) -> asyncpg.Pool | None:
    async with _pg_pools_lock:
        pool = _pg_pools_by_dsn.get(dsn)
        if pool is not None:
            return pool
        try:
            pool = await asyncpg.create_pool(dsn, min_size=0, max_size=max_concurrent + 1)
        except Exception:
            return None
        _pg_pools_by_dsn[dsn] = pool
        return pool


@contextlib.asynccontextmanager
async def _cross_process_llm_slot(dsn: str | None, base_url: str, max_concurrent: int):
    if not dsn:
        yield
        return
    pool = await _advisory_lock_pool(dsn, max_concurrent)
    if pool is None:
        yield
        return

    namespace = _lock_namespace_for_endpoint(base_url)
    conn = None
    slot = None
    try:
        while slot is None:
            for candidate_slot in range(max_concurrent):
                candidate_conn = None
                try:
                    candidate_conn = await pool.acquire()
                    got = await candidate_conn.fetchval(
                        "SELECT pg_try_advisory_lock($1, $2)", namespace, candidate_slot,
                    )
                except Exception:
                    if candidate_conn is not None:
                        with contextlib.suppress(Exception):
                            await pool.release(candidate_conn)
                    # Something about this attempt broke (pool exhausted,
                    # connection dropped) -- give up on cross-process
                    # protection for this call rather than looping forever.
                    yield
                    return
                if got:
                    conn, slot = candidate_conn, candidate_slot
                    break
                await pool.release(candidate_conn)
            if slot is None:
                await asyncio.sleep(0.5)
        yield
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.fetchval("SELECT pg_advisory_unlock($1, $2)", namespace, slot)
            with contextlib.suppress(Exception):
                await pool.release(conn)


def _strip_thinking(text: str | None) -> str | None:
    """Remove <think>...</think> blocks from reasoning models."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


class LLMClient:
    def __init__(self, config: Config):
        self.base_url = config.llm.base_url.rstrip("/")
        self.model = config.llm.model
        self.api_key = config.llm.api_key
        self.max_concurrent_requests = config.llm.max_concurrent_requests
        # Used only for the best-effort cross-process advisory-lock slot --
        # never required for LLMClient to function (see
        # _cross_process_llm_slot's graceful fallback if Postgres is down).
        self._postgres_dsn = config.kb.postgres_dsn
        self.supports_tools: bool | None = None  # Auto-detected on first call
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send a chat completion request. Returns the full response dict."""
        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Try with tools if supported (or unknown)
        use_tools = tools and self.supports_tools is not False
        if use_tools:
            payload["tools"] = tools

        async def _post() -> httpx.Response:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp

        semaphore = await _endpoint_semaphore(self.base_url, self.max_concurrent_requests)
        async with _cross_process_llm_slot(self._postgres_dsn, self.base_url, self.max_concurrent_requests):
            async with semaphore:
                try:
                    resp = await with_retries(_post)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 400 and use_tools:
                        # Model doesn't support tool calling — retry without tools
                        self.supports_tools = False
                        payload.pop("tools", None)
                        resp = await with_retries(_post)
                    else:
                        raise

        if use_tools and self.supports_tools is None:
            self.supports_tools = True

        data = resp.json()

        # Strip thinking tags from response content
        if data.get("choices"):
            msg = data["choices"][0].get("message", {})
            msg["content"] = _strip_thinking(msg.get("content"))

        return data

    async def chat_stream(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content chunks."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        semaphore = await _endpoint_semaphore(self.base_url, self.max_concurrent_requests)
        async with _cross_process_llm_slot(self._postgres_dsn, self.base_url, self.max_concurrent_requests):
            async with semaphore:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

    async def close(self):
        await self._client.aclose()
