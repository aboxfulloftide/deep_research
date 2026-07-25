"""Caches web_search() results by normalized query/capability/freshness so
concurrent or repeated identical searches don't each spend a fresh provider
call. Self-contained SQLite file, same posture as search_usage.py:
web_search() and everything under it (the plain chat tool loop, Extra
Research's collection pipeline) only ever receive a Config, never a
KBDatabase handle -- putting this in the KB Postgres DB would mean threading
one through every one of those call sites before caching could work at all.
"""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from deep_research.config import Config
from deep_research.models import SearchResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    cache_key TEXT PRIMARY KEY,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON search_cache(expires_at);
"""

# "Give current-event queries short TTLs and stable historical queries longer
# TTLs" (RESEARCH_WORK_HANDOFF.md) -- no caller currently distinguishes these,
# so every caller defaults to "default" until one has an actual signal to key
# on (e.g. a future "current events" capability or facet metadata).
FRESHNESS_TTL_SECONDS = {
    "volatile": 600,      # 10 minutes
    "default": 7200,      # 2 hours
    "stable": 604800,     # 7 days
}


def cache_db_path(config: Config) -> Path:
    return config.db_path.parent / "search_cache.db"


@asynccontextmanager
async def _connect(config: Config):
    path = cache_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path)
    try:
        await db.executescript(SCHEMA)
        yield db
    finally:
        await db.close()


def normalize_cache_key(
    query: str, *, capability: str | None, include_alternate_query_engines: bool, freshness: str,
) -> str:
    """Deliberately narrow query normalization (casefold + collapsed
    whitespace) -- aggressive normalization risks merging genuinely distinct
    queries; this only avoids cache misses from incidental case/whitespace
    differences. capability is part of the key so a "grounding" search and an
    ordinary search for the same text never share an entry -- they call
    different providers by design. run_id/plan_id/facet_id/attempt_id are
    deliberately excluded -- those are per-run identifiers for logging, not
    part of what was actually searched, and including them would make every
    call a unique key."""
    normalized_query = " ".join(query.casefold().split())
    return f"{normalized_query}|{capability or ''}|{include_alternate_query_engines}|{freshness}"


async def get_cached_results(config: Config, cache_key: str) -> list[SearchResult] | None:
    """None for both a genuine miss and an expired entry -- callers don't
    need to distinguish the two, both mean "go compute a fresh result."."""
    now = datetime.now(timezone.utc).isoformat()
    async with _connect(config) as db:
        rows = await db.execute_fetchall(
            "SELECT results_json FROM search_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now),
        )
    if not rows:
        return None
    return [SearchResult.model_validate(item) for item in json.loads(rows[0][0])]


async def store_cached_results(
    config: Config, cache_key: str, results: list[SearchResult], freshness: str,
) -> None:
    ttl = FRESHNESS_TTL_SECONDS.get(freshness, FRESHNESS_TTL_SECONDS["default"])
    now = time.time()
    async with _connect(config) as db:
        await db.execute(
            "INSERT INTO search_cache (cache_key, results_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
            "results_json = excluded.results_json, created_at = excluded.created_at, "
            "expires_at = excluded.expires_at",
            (
                cache_key, json.dumps([r.model_dump() for r in results]),
                datetime.now(timezone.utc).isoformat(),
                datetime.fromtimestamp(now + ttl, tz=timezone.utc).isoformat(),
            ),
        )
        await db.commit()
