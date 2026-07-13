"""Per-provider search call logging, so it's possible to answer "how many
searches have we used" and "is duckduckgo/brave/tavily currently responding"
without manually grepping run logs or curling each provider by hand.

Self-contained SQLite file rather than the KB Postgres DB or chat SQLite DB:
web_search() is called from several places (the interactive agent, the web
chat route, KB verification) that don't all have a KB/chat DB handle in
scope, and search usage isn't conceptually tied to either of those anyway.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from deep_research.config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    result_count INTEGER,
    error_message TEXT,
    elapsed_ms INTEGER,
    query TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_search_calls_provider_created ON search_calls(provider, created_at DESC);
"""

# Seed list shown even before any calls are logged. SearXNG can surface
# other engines too (bing, mojeek, wikipedia, ...) -- get_usage_summary
# discovers those dynamically from the log rather than hardcoding them here.
PROVIDERS = ("duckduckgo", "brave", "tavily", "serper")


def usage_db_path(config: Config) -> Path:
    return config.db_path.parent / "search_usage.db"


@asynccontextmanager
async def _connect(config: Config):
    path = usage_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path)
    try:
        await db.executescript(SCHEMA)
        yield db
    finally:
        await db.close()


async def log_search_call(
    config: Config, provider: str, mode: str, status: str, *,
    result_count: int | None = None, error_message: str | None = None,
    elapsed_ms: int | None = None, query: str | None = None,
) -> None:
    """Best-effort -- a logging hiccup must never break the actual search
    call it's describing, so this swallows its own errors rather than
    propagating them (same posture as trust.py's classification writes)."""
    try:
        async with _connect(config) as db:
            await db.execute(
                "INSERT INTO search_calls (provider, mode, status, result_count, error_message, "
                "elapsed_ms, query, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    provider, mode, status, result_count, error_message, elapsed_ms,
                    (query or "")[:200], datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
    except Exception:
        pass


class _Timer:
    def __init__(self):
        self.start = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.start) * 1000)


def timer() -> _Timer:
    return _Timer()


async def get_usage_summary(config: Config, recent_limit: int = 50) -> dict:
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    month = now.strftime("%Y-%m")

    async with _connect(config) as db:
        db.row_factory = aiosqlite.Row
        known = await db.execute_fetchall("SELECT DISTINCT provider FROM search_calls")
        provider_names = sorted({r["provider"] for r in known} | set(PROVIDERS))

        providers = {}
        for provider in provider_names:
            rows = await db.execute_fetchall(
                "SELECT status, result_count, error_message, elapsed_ms, created_at, mode "
                "FROM search_calls WHERE provider = ? ORDER BY created_at DESC LIMIT 500",
                (provider,),
            )
            total_today = sum(1 for r in rows if r["created_at"].startswith(today))
            total_month = sum(1 for r in rows if r["created_at"].startswith(month))
            ok = sum(1 for r in rows if r["status"] == "ok")
            empty = sum(1 for r in rows if r["status"] == "empty")
            error = sum(1 for r in rows if r["status"] == "error")
            last = rows[0] if rows else None
            providers[provider] = {
                "mode": last["mode"] if last else ("scrape" if provider in PROVIDERS[:1] else "api"),
                "calls_today": total_today,
                "calls_month": total_month,
                "ok_count": ok,
                "empty_count": empty,
                "error_count": error,
                "last_call_at": last["created_at"] if last else None,
                "last_status": last["status"] if last else None,
                "last_error": last["error_message"] if last else None,
                "last_result_count": last["result_count"] if last else None,
            }

        recent = await db.execute_fetchall(
            "SELECT provider, mode, status, result_count, error_message, elapsed_ms, query, created_at "
            "FROM search_calls ORDER BY created_at DESC LIMIT ?",
            (recent_limit,),
        )
        recent_calls = [dict(r) for r in recent]

    return {"providers": providers, "recent_calls": recent_calls}
