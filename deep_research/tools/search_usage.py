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
from datetime import datetime, timedelta, timezone
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
# other engines too (bing, mojeek, ...) -- get_usage_summary
# discovers those dynamically from the log rather than hardcoding them here.
PROVIDERS = ("duckduckgo", "brave", "tavily", "serper")

# These unreliable SearXNG scrape engines were replaced by the direct
# wikipedia_api and wikidata_api providers. Keep their old log rows for audit
# history, but do not expose them as current providers or recent calls.
RETIRED_PROVIDERS = ("wikipedia", "wikidata")


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


async def providers_allowed_by_circuit_breaker(
    config: Config, providers: tuple[str, ...], *, max_attempts: int | None, cooldown_hours: int,
) -> set[str]:
    """Return providers safe to include in a deliberately limited trial.

    The usage log makes this durable across web-worker restarts. A provider
    is withheld after its first error for the full cooldown, or once it has
    reached the rolling attempt cap even if every call was successful. This
    is for scrape engines with undocumented bot thresholds, not metered APIs.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)).isoformat()
    allowed: set[str] = set()
    async with _connect(config) as db:
        for provider in providers:
            row = await db.execute_fetchall(
                "SELECT COUNT(*) AS attempts, "
                "MAX(CASE WHEN status = 'error' THEN created_at END) AS last_error "
                "FROM search_calls WHERE provider = ? AND created_at >= ?",
                (provider, cutoff),
            )
            attempts, last_error = row[0]
            if last_error is None and (max_attempts is None or attempts < max_attempts):
                allowed.add(provider)
    return allowed


async def provider_monthly_quota_exhausted(config: Config, provider: str) -> bool:
    """Return whether a metered provider has exhausted this month's quota.

    A Brave per-second 429 is retried before it is logged, so a stored 429
    means that retry also failed. Treat that as monthly exhaustion and keep
    the decision durable across worker restarts. The calendar-month query
    naturally makes the provider eligible again on the first day of the next
    UTC month without a cleanup job or mutable cooldown record.
    """
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    async with _connect(config) as db:
        rows = await db.execute_fetchall(
            "SELECT 1 FROM search_calls "
            "WHERE provider = ? AND status = 'error' AND created_at LIKE ? "
            "AND error_message LIKE '%429%' LIMIT 1",
            (provider, f"{month}%"),
        )
    return bool(rows)


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
        provider_names = sorted(
            ({r["provider"] for r in known} | set(PROVIDERS))
            - set(RETIRED_PROVIDERS)
        )

        providers = {}
        for provider in provider_names:
            # Aggregate in SQLite.  The old implementation fetched the last
            # 500 rows and counted in Python, which made every busy provider
            # look permanently capped at exactly 500 calls this month.
            summary_rows = await db.execute_fetchall(
                "SELECT "
                "COUNT(*) FILTER (WHERE created_at LIKE ?) AS calls_today, "
                "COUNT(*) FILTER (WHERE created_at LIKE ?) AS calls_month, "
                "COUNT(*) FILTER (WHERE created_at LIKE ? AND status = 'ok') AS ok_count, "
                "COUNT(*) FILTER (WHERE created_at LIKE ? AND status = 'empty') AS empty_count, "
                "COUNT(*) FILTER (WHERE created_at LIKE ? AND status = 'error') AS error_count "
                "FROM search_calls WHERE provider = ?",
                (f"{today}%", f"{month}%", f"{month}%", f"{month}%", f"{month}%", provider),
            )
            summary = summary_rows[0]
            last_rows = await db.execute_fetchall(
                "SELECT status, result_count, error_message, elapsed_ms, created_at, mode "
                "FROM search_calls WHERE provider = ? ORDER BY created_at DESC LIMIT 1",
                (provider,),
            )
            last = last_rows[0] if last_rows else None
            providers[provider] = {
                "mode": last["mode"] if last else ("scrape" if provider in PROVIDERS[:1] else "api"),
                "calls_today": summary["calls_today"],
                "calls_month": summary["calls_month"],
                "ok_count": summary["ok_count"],
                "empty_count": summary["empty_count"],
                "error_count": summary["error_count"],
                "last_call_at": last["created_at"] if last else None,
                "last_status": last["status"] if last else None,
                "last_error": last["error_message"] if last else None,
                "last_result_count": last["result_count"] if last else None,
            }

        recent = await db.execute_fetchall(
            "SELECT provider, mode, status, result_count, error_message, elapsed_ms, query, created_at "
            "FROM search_calls WHERE provider NOT IN (?, ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (*RETIRED_PROVIDERS, recent_limit),
        )
        recent_calls = [dict(r) for r in recent]

    return {"providers": providers, "recent_calls": recent_calls}
