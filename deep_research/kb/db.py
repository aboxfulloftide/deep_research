"""Source registry + versioned ingestion storage (PLAN_KB_ARCHITECTURE.md, build
order step 2).

Deliberately a separate SQLite database from chat sessions/messages (deep_research/db.py)
per the plan's design goal of keeping chat history and knowledge-base data apart.
Schema is the SQLite-first subset of the PostgreSQL draft in the plan: source_types,
trust_tiers, sources, source_versions, source_fetch_attempts. Chunking/artifacts
(build order step 3) and everything past it are not part of this module yet.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_types (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trust_tiers (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    rank_weight REAL NOT NULL DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    source_type_id INTEGER NOT NULL REFERENCES source_types(id),
    canonical_uri TEXT NOT NULL,
    canonical_key TEXT NOT NULL UNIQUE,
    title TEXT,
    author TEXT,
    publisher TEXT,
    published_at TEXT,
    trust_tier_id INTEGER REFERENCES trust_tiers(id),
    trust_score REAL,
    language_code TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type_id);
CREATE INDEX IF NOT EXISTS idx_sources_trust_tier ON sources(trust_tier_id);

CREATE TABLE IF NOT EXISTS source_versions (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    version_number INTEGER NOT NULL,
    snapshot_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    http_status INTEGER,
    mime_type TEXT,
    byte_size INTEGER,
    captured_at TEXT NOT NULL,
    is_first_version INTEGER NOT NULL DEFAULT 0,
    is_latest INTEGER NOT NULL DEFAULT 0,
    retention_locked INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_source_versions_source_captured
    ON source_versions(source_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_versions_latest ON source_versions(is_latest);

CREATE TABLE IF NOT EXISTS source_fetch_attempts (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    source_version_id TEXT REFERENCES source_versions(id),
    attempt_type TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_uri TEXT,
    final_uri TEXT,
    http_status INTEGER,
    error_code TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fetch_attempts_source
    ON source_fetch_attempts(source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_attempts_status ON source_fetch_attempts(status);
"""

SOURCE_TYPES = [
    ("web", "Web page"),
    ("youtube_video", "YouTube video"),
    ("youtube_playlist", "YouTube playlist"),
    ("pdf", "PDF document"),
    ("markdown", "Markdown document"),
    ("html_file", "Local HTML file"),
    ("docx", "Word document"),
    ("text", "Plain text document"),
]

TRUST_TIERS = [
    ("official", "Official source", 1.0, "Primary/official statements, filings, government or organization sources"),
    ("reputable_reporting", "Reputable reporting", 0.75, "Established news organizations and journalism"),
    ("secondary_analysis", "Secondary analysis", 0.5, "Analysis, commentary, or aggregation of primary reporting"),
    ("user_generated", "User-generated content", 0.25, "Forums, social media, unvetted user content"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class KBDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def init(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            for code, label in SOURCE_TYPES:
                await db.execute(
                    "INSERT OR IGNORE INTO source_types (code, label) VALUES (?, ?)",
                    (code, label),
                )
            for code, label, rank_weight, description in TRUST_TIERS:
                await db.execute(
                    "INSERT OR IGNORE INTO trust_tiers (code, label, rank_weight, description) "
                    "VALUES (?, ?, ?, ?)",
                    (code, label, rank_weight, description),
                )
            await db.commit()

    # -- reference tables ---------------------------------------------------

    async def get_source_type_id(self, code: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT id FROM source_types WHERE code = ?", (code,))
            row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"Unknown source type code: {code!r}")
        return row[0]

    async def get_trust_tier_id(self, code: str | None) -> int | None:
        if code is None:
            return None
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT id FROM trust_tiers WHERE code = ?", (code,))
            row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"Unknown trust tier code: {code!r}")
        return row[0]

    # -- sources --------------------------------------------------------

    async def get_source_by_canonical_key(self, canonical_key: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM sources WHERE canonical_key = ?", (canonical_key,)
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_source(self, source_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_or_create_source(
        self,
        source_type_code: str,
        canonical_uri: str,
        canonical_key: str,
        title: str | None = None,
        author: str | None = None,
        publisher: str | None = None,
        published_at: str | None = None,
        trust_tier_code: str | None = None,
        language_code: str | None = None,
    ) -> tuple[dict, bool]:
        """Returns (source_row, created). Dedupes on canonical_key."""
        existing = await self.get_source_by_canonical_key(canonical_key)
        if existing is not None:
            return existing, False

        source_type_id = await self.get_source_type_id(source_type_code)
        trust_tier_id = await self.get_trust_tier_id(trust_tier_code)
        source_id = _new_id()
        now = _now()

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    "INSERT INTO sources (id, source_type_id, canonical_uri, canonical_key, "
                    "title, author, publisher, published_at, trust_tier_id, language_code, "
                    "is_active, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        source_id, source_type_id, canonical_uri, canonical_key,
                        title, author, publisher, published_at, trust_tier_id, language_code,
                        now, now,
                    ),
                )
                await db.commit()
            except aiosqlite.IntegrityError:
                # Lost a race against another writer on canonical_key; fall through to re-fetch.
                pass

        source = await self.get_source_by_canonical_key(canonical_key)
        assert source is not None
        return source, source["id"] == source_id

    async def set_source_title_if_missing(self, source_id: str, title: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sources SET title = ?, updated_at = ? "
                "WHERE id = ? AND (title IS NULL OR title = '')",
                (title, _now(), source_id),
            )
            await db.commit()

    # -- fetch attempts -------------------------------------------------

    async def add_fetch_attempt(
        self,
        source_id: str | None,
        attempt_type: str,
        status: str,
        requested_uri: str | None = None,
        source_version_id: str | None = None,
        final_uri: str | None = None,
        http_status: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        attempt_id = _new_id()
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO source_fetch_attempts "
                "(id, source_id, source_version_id, attempt_type, status, requested_uri, "
                "final_uri, http_status, error_code, error_message, started_at, completed_at, "
                "metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id, source_id, source_version_id, attempt_type, status,
                    requested_uri, final_uri, http_status, error_code, error_message,
                    started_at or now, completed_at, json.dumps(metadata) if metadata else None, now,
                ),
            )
            await db.commit()
        return attempt_id

    # -- source versions --------------------------------------------------

    async def get_latest_version(self, source_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM source_versions WHERE source_id = ? AND is_latest = 1", (source_id,)
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_versions(self, source_id: str) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM source_versions WHERE source_id = ? ORDER BY version_number",
                (source_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_next_version_number(self, source_id: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(version_number), 0) FROM source_versions WHERE source_id = ?",
                (source_id,),
            )
            (max_version,) = await cursor.fetchone()
        return max_version + 1

    async def add_source_version(
        self,
        source_id: str,
        content_hash: str,
        snapshot_path: str,
        http_status: int | None = None,
        mime_type: str | None = None,
        byte_size: int | None = None,
        captured_at: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[dict, bool]:
        """Returns (version_row, created). If the content hash matches the current
        latest version, no new version is created (unchanged-content dedup) and the
        caller should not persist the newly-written snapshot bytes."""
        latest = await self.get_latest_version(source_id)
        if latest is not None and latest["content_hash"] == content_hash:
            return latest, False

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(version_number), 0) FROM source_versions WHERE source_id = ?",
                (source_id,),
            )
            (max_version,) = await cursor.fetchone()
            version_number = max_version + 1
            version_id = _new_id()
            now = _now()

            if latest is not None:
                await db.execute(
                    "UPDATE source_versions SET is_latest = 0 WHERE id = ?", (latest["id"],)
                )

            await db.execute(
                "INSERT INTO source_versions "
                "(id, source_id, version_number, snapshot_path, content_hash, http_status, "
                "mime_type, byte_size, captured_at, is_first_version, is_latest, "
                "retention_locked, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)",
                (
                    version_id, source_id, version_number, snapshot_path, content_hash,
                    http_status, mime_type, byte_size, captured_at or now,
                    1 if version_number == 1 else 0,
                    json.dumps(metadata) if metadata else None, now,
                ),
            )
            await db.commit()

            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM source_versions WHERE id = ?", (version_id,))
            row = await cursor.fetchone()

        return dict(row), True

    async def prune_versions(self, source_id: str) -> list[dict]:
        """Enforce the retention policy: keep the first version, the newest two
        versions, and any version with retention_locked set (decisions 13/16 — the
        evidence-integrity carve-out is enforced by never clearing retention_locked
        anywhere in this module). Returns the deleted rows (with snapshot_path) so the
        caller can remove the corresponding files from disk."""
        rows = await self.list_versions(source_id)
        if not rows:
            return []

        first_version_number = rows[0]["version_number"]
        newest_numbers = {r["version_number"] for r in sorted(rows, key=lambda r: r["version_number"])[-2:]}

        keep_ids = set()
        for r in rows:
            if r["version_number"] == first_version_number:
                keep_ids.add(r["id"])
            if r["version_number"] in newest_numbers:
                keep_ids.add(r["id"])
            if r["retention_locked"]:
                keep_ids.add(r["id"])

        to_delete = [r for r in rows if r["id"] not in keep_ids]
        if not to_delete:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "DELETE FROM source_versions WHERE id = ?", [(r["id"],) for r in to_delete]
            )
            await db.commit()

        return to_delete

    async def lock_version_retention(self, version_id: str) -> None:
        """Mark a version as evidence-referenced so it is never pruned. Not called by
        anything yet in step 2 — wired here so step 4 (claim_evidence) has nothing
        left to add to the schema, only a call site."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE source_versions SET retention_locked = 1 WHERE id = ?", (version_id,)
            )
            await db.commit()

    # -- listing / display ------------------------------------------------

    async def list_sources(self, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT s.*, st.code AS source_type_code, tt.code AS trust_tier_code "
                "FROM sources s "
                "JOIN source_types st ON st.id = s.source_type_id "
                "LEFT JOIN trust_tiers tt ON tt.id = s.trust_tier_id "
                "ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def list_fetch_attempts(self, source_id: str) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM source_fetch_attempts WHERE source_id = ? ORDER BY created_at DESC",
                (source_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]
