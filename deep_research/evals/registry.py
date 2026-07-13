"""Registry of models and test sources under evaluation, so setting up the
next side-by-side model comparison is "register it, point a server at it,
run the same commands" instead of hand-copy-pasting a config file and
re-deriving an asyncpg script.

SQLite, modeled directly on deep_research/tools/search_usage.py's connection
pattern -- lives alongside search_usage.db/research.db in the user's app
data dir, not the KB Postgres DB or the repo, since it's local machine state
describing *how to reach* N separate KB instances, not KB content itself.

Deliberately two tables only. There's no eval_runs/stats table: report.py
computes claim counts and resolution rates live from each model's own
Postgres DB on every call, so there's nothing to keep in sync or go stale --
the source of truth for "how did this model do" is always the KB itself.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from deep_research.config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_models (
    slug TEXT PRIMARY KEY,
    display_name TEXT,
    model_path TEXT NOT NULL,
    port INTEGER NOT NULL,
    server_args_json TEXT NOT NULL,
    postgres_dsn TEXT NOT NULL,
    snapshot_dir TEXT NOT NULL,
    config_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_sources (
    slug TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    added_at TEXT NOT NULL
);
"""


def registry_db_path(config: Config) -> Path:
    return config.db_path.parent / "eval_registry.db"


@asynccontextmanager
async def _connect(config: Config):
    path = registry_db_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path)
    db.row_factory = aiosqlite.Row
    try:
        await db.executescript(SCHEMA)
        yield db
    finally:
        await db.close()


async def register_model(
    config: Config, slug: str, *, model_path: str, port: int, server_args_json: str,
    postgres_dsn: str, snapshot_dir: str, config_path: str, display_name: str | None = None,
) -> dict:
    async with _connect(config) as db:
        await db.execute(
            "INSERT INTO eval_models (slug, display_name, model_path, port, server_args_json, "
            "postgres_dsn, snapshot_dir, config_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET display_name=excluded.display_name, model_path=excluded.model_path, "
            "port=excluded.port, server_args_json=excluded.server_args_json, postgres_dsn=excluded.postgres_dsn, "
            "snapshot_dir=excluded.snapshot_dir, config_path=excluded.config_path",
            (
                slug, display_name or slug, model_path, port, server_args_json,
                postgres_dsn, snapshot_dir, config_path, datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()
        row = await db.execute_fetchall("SELECT * FROM eval_models WHERE slug = ?", (slug,))
        return dict(row[0])


async def get_model(config: Config, slug: str) -> dict | None:
    async with _connect(config) as db:
        rows = await db.execute_fetchall("SELECT * FROM eval_models WHERE slug = ?", (slug,))
        return dict(rows[0]) if rows else None


async def list_models(config: Config) -> list[dict]:
    async with _connect(config) as db:
        rows = await db.execute_fetchall("SELECT * FROM eval_models ORDER BY created_at")
        return [dict(r) for r in rows]


async def add_source(config: Config, slug: str, *, url: str, title: str | None = None) -> dict:
    async with _connect(config) as db:
        await db.execute(
            "INSERT INTO eval_sources (slug, url, title, added_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET url=excluded.url, title=excluded.title",
            (slug, url, title, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
        rows = await db.execute_fetchall("SELECT * FROM eval_sources WHERE slug = ?", (slug,))
        return dict(rows[0])


async def get_source(config: Config, slug: str) -> dict | None:
    async with _connect(config) as db:
        rows = await db.execute_fetchall("SELECT * FROM eval_sources WHERE slug = ?", (slug,))
        return dict(rows[0]) if rows else None


async def list_sources(config: Config) -> list[dict]:
    async with _connect(config) as db:
        rows = await db.execute_fetchall("SELECT * FROM eval_sources ORDER BY added_at")
        return [dict(r) for r in rows]
