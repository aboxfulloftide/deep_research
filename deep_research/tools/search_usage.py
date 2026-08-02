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
from zoneinfo import ZoneInfo

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

# Nullable, additive columns so a search call can be joined back to the
# research run/plan/facet/attempt that made it and the capability it was
# serving. Unlike kb/db.py's Postgres schema, SQLite has no
# "ALTER TABLE ... ADD COLUMN IF NOT EXISTS" -- it errors with a syntax error,
# not a silent no-op -- so each missing column is added individually via
# PRAGMA table_info, safe to run on every connect.
_ADDITIVE_COLUMNS = ("run_id", "plan_id", "facet_id", "attempt_id", "capability", "error_category")


async def _ensure_additive_columns(db: aiosqlite.Connection) -> None:
    existing = {row[1] for row in await db.execute_fetchall("PRAGMA table_info(search_calls)")}
    added = False
    for column in _ADDITIVE_COLUMNS:
        if column not in existing:
            await db.execute(f"ALTER TABLE search_calls ADD COLUMN {column} TEXT")
            added = True
    if added:
        await db.commit()

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
        await _ensure_additive_columns(db)
        yield db
    finally:
        await db.close()


async def log_search_call(
    config: Config, provider: str, mode: str, status: str, *,
    result_count: int | None = None, error_message: str | None = None,
    elapsed_ms: int | None = None, query: str | None = None,
    run_id: str | None = None, plan_id: str | None = None,
    facet_id: str | None = None, attempt_id: str | None = None,
    capability: str | None = None, error_category: str | None = None,
) -> None:
    """Best-effort -- a logging hiccup must never break the actual search
    call it's describing, so this swallows its own errors rather than
    propagating them (same posture as trust.py's classification writes).

    run_id/plan_id/facet_id/attempt_id/capability are optional context IDs so
    a search call can later be joined back to the research run/plan/facet/
    attempt (or claim verification) that made it -- callers with no such
    concept (e.g. the plain chat tool loop) simply leave them unset.

    error_category is the caller's classify_http_error(exc) result (e.g.
    "rate_limited", "server_error") -- a typed field instead of re-parsing
    the free-text error_message later, used by provider_monthly_quota_exhausted
    below."""
    try:
        async with _connect(config) as db:
            await db.execute(
                "INSERT INTO search_calls (provider, mode, status, result_count, error_message, "
                "elapsed_ms, query, run_id, plan_id, facet_id, attempt_id, capability, error_category, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    provider, mode, status, result_count, error_message, elapsed_ms,
                    (query or "")[:200], run_id, plan_id, facet_id, attempt_id, capability,
                    error_category, datetime.now(timezone.utc).isoformat(),
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

    A Brave per-second 429 is retried before it is logged, so a stored
    rate-limited row means that retry also failed. Treat that as monthly
    exhaustion and keep the decision durable across worker restarts. The
    calendar-month query naturally makes the provider eligible again on the
    first day of the next UTC month without a cleanup job or mutable cooldown
    record. Uses the typed error_category column (classify_http_error's
    "rate_limited") rather than string-matching "429" in error_message --
    the caller must pass error_category to log_search_call for this to see it.
    """
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    async with _connect(config) as db:
        rows = await db.execute_fetchall(
            "SELECT 1 FROM search_calls "
            "WHERE provider = ? AND status = 'error' AND created_at LIKE ? "
            "AND error_category = 'rate_limited' LIMIT 1",
            (provider, f"{month}%"),
        )
    return bool(rows)


async def serper_quota_remaining(config: Config) -> int | None:
    """Estimates Serper's remaining paid-credit balance from a manually
    recorded snapshot (config.serper.quota_remaining_snapshot/_at) minus
    every call logged against 'serper' since that snapshot was taken.

    Serper's balance is a running total that never resets monthly (unlike
    provider_monthly_quota_exhausted's calendar-month rate-limit tracking
    above) and isn't exposed via their search API, so there's no way to
    query it live -- this only stays accurate between snapshots because
    log_search_call records every real call made to their API, successful
    or not (an error response still consumed a real HTTP request against
    their endpoint). Returns None if no snapshot has ever been recorded.
    """
    if not config.serper.quota_remaining_snapshot or not config.serper.quota_remaining_snapshot_at:
        return None
    async with _connect(config) as db:
        rows = await db.execute_fetchall(
            "SELECT COUNT(*) FROM search_calls WHERE provider = 'serper' AND created_at >= ?",
            (config.serper.quota_remaining_snapshot_at,),
        )
    used_since_snapshot = rows[0][0]
    return max(0, config.serper.quota_remaining_snapshot - used_since_snapshot)


class _Timer:
    def __init__(self):
        self.start = time.monotonic()

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.start) * 1000)


def timer() -> _Timer:
    return _Timer()


def _utc_period_bounds(
    timezone_name: str, now_utc: datetime | None = None,
) -> tuple[datetime, datetime, datetime, datetime]:
    """Return local day/month boundaries converted to UTC for storage queries.

    Search-call timestamps stay in UTC. The usage cards, however, describe
    periods to a person ("today" and "this month"), so their boundaries need
    to follow that person's timezone rather than rolling over at UTC midnight.
    """
    display_timezone = ZoneInfo(timezone_name)
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    local_now = now_utc.astimezone(display_timezone)

    day_start_local = datetime(
        local_now.year, local_now.month, local_now.day, tzinfo=display_timezone,
    )
    day_end_local = day_start_local + timedelta(days=1)
    month_start_local = datetime(
        local_now.year, local_now.month, 1, tzinfo=display_timezone,
    )
    if local_now.month == 12:
        month_end_local = datetime(
            local_now.year + 1, 1, 1, tzinfo=display_timezone,
        )
    else:
        month_end_local = datetime(
            local_now.year, local_now.month + 1, 1, tzinfo=display_timezone,
        )

    return tuple(
        boundary.astimezone(timezone.utc)
        for boundary in (
            day_start_local, day_end_local, month_start_local, month_end_local,
        )
    )


async def get_usage_summary(
    config: Config,
    recent_limit: int = 50,
    *,
    timezone_name: str = "UTC",
    now_utc: datetime | None = None,
) -> dict:
    day_start, day_end, month_start, month_end = _utc_period_bounds(
        timezone_name, now_utc,
    )
    day_start_iso = day_start.isoformat()
    day_end_iso = day_end.isoformat()
    month_start_iso = month_start.isoformat()
    month_end_iso = month_end.isoformat()

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
                "COUNT(*) FILTER (WHERE created_at >= ? AND created_at < ?) AS calls_today, "
                "COUNT(*) FILTER (WHERE created_at >= ? AND created_at < ?) AS calls_month, "
                "COUNT(*) FILTER (WHERE created_at >= ? AND created_at < ? AND status = 'ok') AS ok_count, "
                "COUNT(*) FILTER (WHERE created_at >= ? AND created_at < ? AND status = 'empty') AS empty_count, "
                "COUNT(*) FILTER (WHERE created_at >= ? AND created_at < ? AND status = 'error') AS error_count "
                "FROM search_calls WHERE provider = ?",
                (
                    day_start_iso, day_end_iso,
                    month_start_iso, month_end_iso,
                    month_start_iso, month_end_iso,
                    month_start_iso, month_end_iso,
                    month_start_iso, month_end_iso,
                    provider,
                ),
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
            # Unlike Serper's running paid balance (never resets, no API to
            # query it -- see serper_quota_remaining below), these providers'
            # quotas are real calendar-month allowances that reset on the
            # same boundary already computed above for calls_month, so each
            # can be derived directly from logged calls with no manual
            # snapshot needed. 0/unset means unknown -- omit quota_remaining
            # rather than show a misleading number.
            monthly_quota = {
                "tavily": config.tavily.monthly_quota,
                "brave": config.brave.monthly_quota,
                "brave_fallback": config.brave.fallback_monthly_quota,
            }.get(provider)
            if monthly_quota:
                providers[provider]["quota_remaining"] = max(0, monthly_quota - summary["calls_month"])

        recent = await db.execute_fetchall(
            "SELECT provider, mode, status, result_count, error_message, elapsed_ms, query, created_at "
            "FROM search_calls WHERE provider NOT IN (?, ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (*RETIRED_PROVIDERS, recent_limit),
        )
        recent_calls = [dict(r) for r in recent]

    if "serper" in providers:
        providers["serper"]["quota_remaining"] = await serper_quota_remaining(config)

    return {
        "providers": providers,
        "recent_calls": recent_calls,
        "timezone": timezone_name,
    }
