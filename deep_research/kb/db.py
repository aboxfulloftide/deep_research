"""Source registry, versioned ingestion, chunk storage/retrieval, and
extraction/resolution storage (PLAN_KB_ARCHITECTURE.md, build order steps 2-5).

Deliberately a separate PostgreSQL database from chat sessions/messages
(deep_research/db.py, still SQLite) per the plan's design goal of keeping chat
history and knowledge-base data apart.

Migrated from SQLite to PostgreSQL per build order step 5, once the schema had
been exercised end-to-end (steps 2-4) against real ingested/chunked/extracted
data. Notable upgrades that came for free with the migration:
- JSONB instead of TEXT+json.dumps/loads for metadata/payload columns
- real TIMESTAMPTZ instead of ISO8601 strings
- native full-text search (tsvector + GIN + websearch_to_tsquery) replacing the
  SQLite FTS5 virtual table — websearch_to_tsquery also fixes the punctuation-
  crash bug from the SQLite version for free, since it never raises a syntax
  error on arbitrary user input (unlike FTS5's MATCH operator, which did)
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from deep_research.kb.chunking import normalize_name, normalize_ws

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_types (
    id SERIAL PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trust_tiers (
    id SERIAL PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    rank_weight DOUBLE PRECISION NOT NULL DEFAULT 0,
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
    trust_score DOUBLE PRECISION,
    language_code TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
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
    captured_at TIMESTAMPTZ NOT NULL,
    is_first_version BOOLEAN NOT NULL DEFAULT FALSE,
    is_latest BOOLEAN NOT NULL DEFAULT FALSE,
    retention_locked BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL,
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
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fetch_attempts_source
    ON source_fetch_attempts(source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fetch_attempts_status ON source_fetch_attempts(status);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES source_versions(id),
    artifact_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    chunk_params_hash TEXT NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_source_version
    ON artifacts(source_version_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_current ON artifacts(is_current);

-- artifact_chunks referenced by claim_evidence must be immutable: re-chunking
-- creates a new artifact generation (is_current flips), old chunk rows are
-- never updated or deleted. See "Retention vs. Evidence Integrity" in
-- PLAN_KB_ARCHITECTURE.md. chunk_tsv is a generated column — Postgres keeps it
-- in sync automatically, no manual FTS-index sync code needed (unlike the
-- SQLite FTS5 virtual table this replaces).
CREATE TABLE IF NOT EXISTS artifact_chunks (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    token_estimate INTEGER,
    section_label TEXT,
    page_number INTEGER,
    time_start_seconds DOUBLE PRECISION,
    time_end_seconds DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    chunk_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    UNIQUE(artifact_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_artifact_chunks_artifact
    ON artifact_chunks(artifact_id, page_number);
CREATE INDEX IF NOT EXISTS idx_artifact_chunks_time
    ON artifact_chunks(artifact_id, time_start_seconds);
CREATE INDEX IF NOT EXISTS idx_artifact_chunks_tsv
    ON artifact_chunks USING GIN(chunk_tsv);

-- Canonical entities. UNIQUE(entity_type, normalized_name) *is* the exact-match
-- auto-merge tier from decision 25 — INSERT ... ON CONFLICT DO NOTHING + a
-- re-select handles "same entity, same type, identical normalized name" for
-- free. Anything less than exact (fuzzy/substring) never merges here; it only
-- ever produces a resolution_candidates row for review. entity_mentions
-- (per-chunk mention locations) is a deferred second-wave table — v1 reads
-- mention locations out of extracted_observations.raw_payload instead.
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE(entity_type, normalized_name)
);

-- Canonical events. UNIQUE(normalized_title) is the exact-match auto-merge tier
-- decision 25 specifies for events (same pattern as entities). Fuzzy event
-- matching is explicitly deferred until real event volume justifies it.
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL UNIQUE,
    description TEXT,
    event_type TEXT,
    start_at TEXT,
    end_at TEXT,
    date_precision TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_at);

-- Canonical claims. UNIQUE(claim_type, normalized_text) auto-merges only
-- byte-identical normalized claims — decision 25 says claims have "no auto-merge
-- tier" for *fuzzy/lexical* similarity (the spike measured that catching zero
-- real duplicates), but an exact-text match is the same safe case as entities/
-- events, so it gets the same treatment. Near-duplicate claims (different
-- phrasing, same fact) never merge here; they go through embedding-similarity
-- candidate generation into resolution_candidates instead. No subject/object
-- entity columns yet — the validated extraction prompt returns a flat entity
-- list per claim, not subject/object roles, so per-claim entity association
-- lives in extracted_observations.raw_payload for v1 (entity_mentions is a
-- deferred second-wave table, same as events).
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    claim_type TEXT NOT NULL,
    event_id TEXT REFERENCES events(id),
    canonical_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unverified',
    preferred_source_id TEXT REFERENCES sources(id),
    confidence DOUBLE PRECISION,
    importance_score DOUBLE PRECISION,
    verification_attempted_at TIMESTAMPTZ,
    is_user_reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE(claim_type, normalized_text)
);

CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_event ON claims(event_id);

-- Provenance for one extraction pass over one artifact's current chunks.
-- run_signature (model+prompt+schema hash) makes re-extraction idempotent: the
-- same signature against the same artifact is a no-op unless forced.
CREATE TABLE IF NOT EXISTS extraction_runs (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    run_signature TEXT NOT NULL,
    model_id TEXT NOT NULL,
    runtime TEXT,
    prompt_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    extraction_schema_version TEXT NOT NULL,
    parameters JSONB,
    status TEXT NOT NULL DEFAULT 'running',
    chunk_count INTEGER,
    observation_count INTEGER,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_extraction_runs_artifact ON extraction_runs(artifact_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_extraction_runs_signature ON extraction_runs(artifact_id, run_signature);
CREATE INDEX IF NOT EXISTS idx_extraction_runs_status ON extraction_runs(status);

-- Raw model output before resolution — the audit trail / noise buffer that
-- protects the curated claims table from first-pass extraction noise.
CREATE TABLE IF NOT EXISTS extracted_observations (
    id TEXT PRIMARY KEY,
    extraction_run_id TEXT NOT NULL REFERENCES extraction_runs(id),
    artifact_chunk_id TEXT NOT NULL REFERENCES artifact_chunks(id),
    observation_type TEXT NOT NULL DEFAULT 'claim',
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    candidate_claim_id TEXT REFERENCES claims(id),
    confidence DOUBLE PRECISION,
    importance_score DOUBLE PRECISION,
    char_start INTEGER,
    char_end INTEGER,
    quote_match_type TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_run ON extracted_observations(extraction_run_id);
CREATE INDEX IF NOT EXISTS idx_observations_chunk ON extracted_observations(artifact_chunk_id);
CREATE INDEX IF NOT EXISTS idx_observations_status ON extracted_observations(status);

-- Provenance links from claims to the exact chunk they came from. Creating one
-- of these is what locks the referenced source_version's retention (see
-- lock_version_retention / "Retention vs. Evidence Integrity").
CREATE TABLE IF NOT EXISTS claim_evidence (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    extraction_run_id TEXT REFERENCES extraction_runs(id),
    extracted_observation_id TEXT REFERENCES extracted_observations(id),
    artifact_chunk_id TEXT NOT NULL REFERENCES artifact_chunks(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    source_version_id TEXT NOT NULL REFERENCES source_versions(id),
    evidence_type TEXT NOT NULL,
    excerpt_text TEXT,
    excerpt_hash TEXT,
    char_start INTEGER,
    char_end INTEGER,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim ON claim_evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_chunk ON claim_evidence(artifact_chunk_id);

-- The v1 merge/review queue (decision 25). Uncertain matches land here instead
-- of being silently merged. Accepting/rejecting a candidate only changes its
-- status in this step — executing an accepted merge (reassigning evidence,
-- tombstoning the duplicate) is explicitly deferred, not implemented yet.
CREATE TABLE IF NOT EXISTS resolution_candidates (
    id TEXT PRIMARY KEY,
    candidate_type TEXT NOT NULL,
    left_entity_id TEXT REFERENCES entities(id),
    right_entity_id TEXT REFERENCES entities(id),
    left_event_id TEXT REFERENCES events(id),
    right_event_id TEXT REFERENCES events(id),
    left_claim_id TEXT REFERENCES claims(id),
    right_claim_id TEXT REFERENCES claims(id),
    score DOUBLE PRECISION,
    reason TEXT,
    method TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resolution_candidates_type_status
    ON resolution_candidates(candidate_type, status);

CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    claim_id TEXT REFERENCES claims(id),
    event_id TEXT REFERENCES events(id),
    entity_id TEXT REFERENCES entities(id),
    metric_name TEXT NOT NULL,
    value_numeric DOUBLE PRECISION,
    value_text TEXT,
    unit TEXT,
    currency_code TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_claim ON metrics(claim_id);
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


async def _register_jsonb_codec(conn: asyncpg.Connection) -> None:
    """Lets us pass/receive Python dicts directly for jsonb columns instead of
    manually json.dumps/loads at every call site."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
    )


class KBDatabase:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def init(self):
        self.pool = await asyncpg.create_pool(self.dsn, init=_register_jsonb_codec)
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA)
            for code, label in SOURCE_TYPES:
                await conn.execute(
                    "INSERT INTO source_types (code, label) VALUES ($1, $2) "
                    "ON CONFLICT (code) DO NOTHING",
                    code, label,
                )
            for code, label, rank_weight, description in TRUST_TIERS:
                await conn.execute(
                    "INSERT INTO trust_tiers (code, label, rank_weight, description) "
                    "VALUES ($1, $2, $3, $4) ON CONFLICT (code) DO NOTHING",
                    code, label, rank_weight, description,
                )

    async def close(self):
        if self.pool is not None:
            await self.pool.close()

    # -- reference tables ---------------------------------------------------

    async def get_source_type_id(self, code: str) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchval("SELECT id FROM source_types WHERE code = $1", code)
        if row is None:
            raise ValueError(f"Unknown source type code: {code!r}")
        return row

    async def get_source_type_code(self, source_type_id: int) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchval("SELECT code FROM source_types WHERE id = $1", source_type_id)
        if row is None:
            raise ValueError(f"Unknown source type id: {source_type_id!r}")
        return row

    async def get_trust_tier_id(self, code: str | None) -> int | None:
        if code is None:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchval("SELECT id FROM trust_tiers WHERE code = $1", code)
        if row is None:
            raise ValueError(f"Unknown trust tier code: {code!r}")
        return row

    # -- sources --------------------------------------------------------

    async def get_source_by_canonical_key(self, canonical_key: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sources WHERE canonical_key = $1", canonical_key)
        return dict(row) if row else None

    async def get_source(self, source_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sources WHERE id = $1", source_id)
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

        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sources (id, source_type_id, canonical_uri, canonical_key, "
                "title, author, publisher, published_at, trust_tier_id, language_code, "
                "is_active, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE, $11, $12) "
                "ON CONFLICT (canonical_key) DO NOTHING",
                source_id, source_type_id, canonical_uri, canonical_key,
                title, author, publisher, published_at, trust_tier_id, language_code,
                now, now,
            )

        source = await self.get_source_by_canonical_key(canonical_key)
        assert source is not None
        return source, source["id"] == source_id

    async def set_source_title_if_missing(self, source_id: str, title: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE sources SET title = $1, updated_at = $2 "
                "WHERE id = $3 AND (title IS NULL OR title = '')",
                title, _now(), source_id,
            )

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
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> str:
        attempt_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO source_fetch_attempts "
                "(id, source_id, source_version_id, attempt_type, status, requested_uri, "
                "final_uri, http_status, error_code, error_message, started_at, completed_at, "
                "metadata, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
                attempt_id, source_id, source_version_id, attempt_type, status,
                requested_uri, final_uri, http_status, error_code, error_message,
                started_at or now, completed_at, metadata, now,
            )
        return attempt_id

    # -- source versions --------------------------------------------------

    async def get_latest_version(self, source_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM source_versions WHERE source_id = $1 AND is_latest = TRUE", source_id,
            )
        return dict(row) if row else None

    async def list_versions(self, source_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM source_versions WHERE source_id = $1 ORDER BY version_number", source_id,
            )
        return [dict(r) for r in rows]

    async def get_source_version(self, version_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM source_versions WHERE id = $1", version_id)
        return dict(row) if row else None

    async def get_next_version_number(self, source_id: str) -> int:
        async with self.pool.acquire() as conn:
            max_version = await conn.fetchval(
                "SELECT COALESCE(MAX(version_number), 0) FROM source_versions WHERE source_id = $1",
                source_id,
            )
        return max_version + 1

    async def add_source_version(
        self,
        source_id: str,
        content_hash: str,
        snapshot_path: str,
        http_status: int | None = None,
        mime_type: str | None = None,
        byte_size: int | None = None,
        captured_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> tuple[dict, bool]:
        """Returns (version_row, created). If the content hash matches the current
        latest version, no new version is created (unchanged-content dedup) and the
        caller should not persist the newly-written snapshot bytes."""
        latest = await self.get_latest_version(source_id)
        if latest is not None and latest["content_hash"] == content_hash:
            return latest, False

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                max_version = await conn.fetchval(
                    "SELECT COALESCE(MAX(version_number), 0) FROM source_versions WHERE source_id = $1",
                    source_id,
                )
                version_number = max_version + 1
                version_id = _new_id()
                now = _now()

                if latest is not None:
                    await conn.execute(
                        "UPDATE source_versions SET is_latest = FALSE WHERE id = $1", latest["id"],
                    )

                row = await conn.fetchrow(
                    "INSERT INTO source_versions "
                    "(id, source_id, version_number, snapshot_path, content_hash, http_status, "
                    "mime_type, byte_size, captured_at, is_first_version, is_latest, "
                    "retention_locked, metadata, created_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE, FALSE, $11, $12) "
                    "RETURNING *",
                    version_id, source_id, version_number, snapshot_path, content_hash,
                    http_status, mime_type, byte_size, captured_at or now,
                    version_number == 1, metadata, now,
                )

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

        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM source_versions WHERE id = ANY($1::text[])",
                [r["id"] for r in to_delete],
            )

        return to_delete

    async def lock_version_retention(self, version_id: str) -> None:
        """Mark a version as evidence-referenced so it is never pruned."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE source_versions SET retention_locked = TRUE WHERE id = $1", version_id,
            )

    # -- listing / display ------------------------------------------------

    async def list_sources(self, limit: int = 50) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT s.*, st.code AS source_type_code, tt.code AS trust_tier_code "
                "FROM sources s "
                "JOIN source_types st ON st.id = s.source_type_id "
                "LEFT JOIN trust_tiers tt ON tt.id = s.trust_tier_id "
                "ORDER BY s.updated_at DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]

    async def list_fetch_attempts(self, source_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM source_fetch_attempts WHERE source_id = $1 ORDER BY created_at DESC",
                source_id,
            )
        return [dict(r) for r in rows]

    # -- artifacts & chunks (build order step 3) -------------------------

    async def get_current_artifact(self, source_version_id: str, artifact_type: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM artifacts WHERE source_version_id = $1 AND artifact_type = $2 "
                "AND is_current = TRUE",
                source_version_id, artifact_type,
            )
        return dict(row) if row else None

    async def get_current_artifacts_for_version(self, source_version_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM artifacts WHERE source_version_id = $1 AND is_current = TRUE",
                source_version_id,
            )
        return [dict(r) for r in rows]

    async def get_artifact(self, artifact_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM artifacts WHERE id = $1", artifact_id)
        return dict(row) if row else None

    async def upsert_artifact(
        self,
        artifact_id: str,
        source_version_id: str,
        artifact_type: str,
        storage_path: str,
        content_hash: str,
        chunk_params_hash: str,
        title: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[dict, bool]:
        """Returns (artifact_row, is_new_generation). If the current artifact for this
        (source_version_id, artifact_type) already used the same chunk_params_hash,
        this is a no-op — the caller should not re-chunk, and artifact_id is ignored
        (the existing row is returned). Otherwise the old artifact (if any) is marked
        non-current — its chunk rows are left untouched, never updated or deleted,
        since re-chunking must produce a new generation rather than mutate rows
        `claim_evidence` may later reference. artifact_id is caller-supplied (rather
        than generated here) so the caller can write the extracted-text file to disk
        at that id before/while the DB row is inserted."""
        current = await self.get_current_artifact(source_version_id, artifact_type)
        if current is not None and current["chunk_params_hash"] == chunk_params_hash:
            return current, False

        now = _now()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if current is not None:
                    await conn.execute("UPDATE artifacts SET is_current = FALSE WHERE id = $1", current["id"])
                row = await conn.fetchrow(
                    "INSERT INTO artifacts (id, source_version_id, artifact_type, storage_path, "
                    "content_hash, title, summary, chunk_params_hash, is_current, metadata, "
                    "created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, TRUE, $9, $10, $11) "
                    "RETURNING *",
                    artifact_id, source_version_id, artifact_type, storage_path, content_hash,
                    title, summary, chunk_params_hash, metadata, now, now,
                )

        return dict(row), True

    async def add_chunk(
        self,
        artifact_id: str,
        chunk_index: int,
        chunk_text: str,
        chunk_hash: str,
        char_start: int | None = None,
        char_end: int | None = None,
        token_estimate: int | None = None,
        section_label: str | None = None,
        page_number: int | None = None,
        time_start_seconds: float | None = None,
        time_end_seconds: float | None = None,
        metadata: dict | None = None,
    ) -> dict:
        chunk_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO artifact_chunks (id, artifact_id, chunk_index, chunk_text, chunk_hash, "
                "char_start, char_end, token_estimate, section_label, page_number, "
                "time_start_seconds, time_end_seconds, metadata, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
                "RETURNING *",
                chunk_id, artifact_id, chunk_index, chunk_text, chunk_hash,
                char_start, char_end, token_estimate, section_label, page_number,
                time_start_seconds, time_end_seconds, metadata, now,
            )
        return dict(row)

    async def list_chunks(self, artifact_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM artifact_chunks WHERE artifact_id = $1 ORDER BY chunk_index", artifact_id,
            )
        return [dict(r) for r in rows]

    async def search_chunks(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over current-generation chunks only (is_current =
        TRUE), joined back to source metadata for display. websearch_to_tsquery
        never raises a syntax error on arbitrary user input (unlike SQLite
        FTS5's MATCH, which crashed on punctuation like '92% GDP growth')."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chunk_id, c.chunk_index, c.page_number,
                    c.time_start_seconds, c.time_end_seconds,
                    a.id AS artifact_id, a.artifact_type,
                    sv.id AS source_version_id, sv.version_number,
                    s.id AS source_id, s.title AS source_title, s.canonical_uri,
                    ts_rank_cd(c.chunk_tsv, websearch_to_tsquery('english', $1)) AS score,
                    ts_headline(
                        'english', c.chunk_text, websearch_to_tsquery('english', $1),
                        'StartSel=>>>,StopSel=<<<,MaxWords=15,MinWords=5,MaxFragments=1'
                    ) AS snippet
                FROM artifact_chunks c
                JOIN artifacts a ON a.id = c.artifact_id AND a.is_current = TRUE
                JOIN source_versions sv ON sv.id = a.source_version_id
                JOIN sources s ON s.id = sv.source_id
                WHERE c.chunk_tsv @@ websearch_to_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query, limit,
            )
        return [dict(r) for r in rows]

    # -- entities/events/claims (build order step 4) ---------------------

    async def get_or_create_entity(
        self, entity_type: str, name: str, description: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[dict, bool]:
        """Exact-match auto-merge tier (decision 25): same entity_type + same
        normalized_name reuses the existing row via the UNIQUE constraint."""
        normalized = normalize_name(name)
        entity_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO entities "
                "(id, entity_type, name, normalized_name, description, metadata, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
                "ON CONFLICT (entity_type, normalized_name) DO NOTHING",
                entity_id, entity_type, name, normalized, description, metadata, now, now,
            )
            row = await conn.fetchrow(
                "SELECT * FROM entities WHERE entity_type = $1 AND normalized_name = $2",
                entity_type, normalized,
            )
        row = dict(row)
        return row, row["id"] == entity_id

    async def get_entity(self, entity_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM entities WHERE id = $1", entity_id)
        return dict(row) if row else None

    async def list_entities(self, entity_type: str | None = None, limit: int = 500) -> list[dict]:
        async with self.pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    "SELECT * FROM entities WHERE entity_type = $1 ORDER BY updated_at DESC LIMIT $2",
                    entity_type, limit,
                )
            else:
                rows = await conn.fetch("SELECT * FROM entities ORDER BY updated_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]

    async def get_or_create_event(
        self, title: str, description: str | None = None, event_type: str | None = None,
        start_at: str | None = None, end_at: str | None = None,
        date_precision: str | None = None, metadata: dict | None = None,
    ) -> tuple[dict, bool]:
        """Exact-match auto-merge tier (decision 25): normalized_title match
        reuses the existing row. Fuzzy event matching is deferred."""
        normalized = normalize_name(title)
        event_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO events "
                "(id, title, normalized_title, description, event_type, start_at, end_at, "
                "date_precision, metadata, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
                "ON CONFLICT (normalized_title) DO NOTHING",
                event_id, title, normalized, description, event_type, start_at, end_at,
                date_precision, metadata, now, now,
            )
            row = await conn.fetchrow("SELECT * FROM events WHERE normalized_title = $1", normalized)
        row = dict(row)
        return row, row["id"] == event_id

    async def list_events(self, limit: int = 500) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM events ORDER BY updated_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]

    async def get_or_create_claim(
        self, claim_type: str, canonical_text: str, event_id: str | None = None,
        confidence: float | None = None, importance_score: float | None = None,
    ) -> tuple[dict, bool]:
        """Exact-match auto-merge tier: byte-identical (claim_type,
        normalized_text) reuses the existing claim row — this is the same safe
        exact-match case as entities/events, not the fuzzy/lexical similarity
        decision 25 ruled out for claims. Anything less than exact never merges
        here; see the embedding-based candidate generation in resolution.py."""
        normalized = normalize_ws(canonical_text).lower()
        claim_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO claims "
                "(id, claim_type, event_id, canonical_text, normalized_text, status, "
                "confidence, importance_score, is_user_reviewed, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, 'unverified', $6, $7, FALSE, $8, $9) "
                "ON CONFLICT (claim_type, normalized_text) DO NOTHING",
                claim_id, claim_type, event_id, canonical_text, normalized,
                confidence, importance_score, now, now,
            )
            row = await conn.fetchrow(
                "SELECT * FROM claims WHERE claim_type = $1 AND normalized_text = $2",
                claim_type, normalized,
            )
        row = dict(row)
        return row, row["id"] == claim_id

    async def get_claim(self, claim_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM claims WHERE id = $1", claim_id)
        return dict(row) if row else None

    async def list_claims(self, limit: int = 100) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM claims ORDER BY updated_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]

    async def add_claim_evidence(
        self, claim_id: str, artifact_chunk_id: str, source_id: str, source_version_id: str,
        evidence_type: str = "support", excerpt_text: str | None = None,
        excerpt_hash: str | None = None, extraction_run_id: str | None = None,
        extracted_observation_id: str | None = None, char_start: int | None = None,
        char_end: int | None = None, confidence: float | None = None,
    ) -> dict:
        """Creates the evidence link AND locks the referenced source_version's
        retention (see lock_version_retention / "Retention vs. Evidence Integrity")."""
        evidence_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO claim_evidence "
                "(id, claim_id, extraction_run_id, extracted_observation_id, artifact_chunk_id, "
                "source_id, source_version_id, evidence_type, excerpt_text, excerpt_hash, "
                "char_start, char_end, confidence, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
                "RETURNING *",
                evidence_id, claim_id, extraction_run_id, extracted_observation_id, artifact_chunk_id,
                source_id, source_version_id, evidence_type, excerpt_text, excerpt_hash,
                char_start, char_end, confidence, now,
            )
        await self.lock_version_retention(source_version_id)
        return dict(row)

    async def list_claim_evidence(self, claim_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ce.*, s.title AS source_title, s.canonical_uri "
                "FROM claim_evidence ce "
                "JOIN sources s ON s.id = ce.source_id "
                "WHERE ce.claim_id = $1 ORDER BY ce.created_at",
                claim_id,
            )
        return [dict(r) for r in rows]

    async def add_metric(
        self, metric_name: str, claim_id: str | None = None, event_id: str | None = None,
        entity_id: str | None = None, value_numeric: float | None = None,
        value_text: str | None = None, unit: str | None = None,
        currency_code: str | None = None, metadata: dict | None = None,
    ) -> dict:
        metric_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO metrics (id, claim_id, event_id, entity_id, metric_name, "
                "value_numeric, value_text, unit, currency_code, metadata, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING *",
                metric_id, claim_id, event_id, entity_id, metric_name, value_numeric,
                value_text, unit, currency_code, metadata, now,
            )
        return dict(row)

    # -- resolution candidates (v1 merge/review queue) --------------------

    async def add_entity_resolution_candidate(
        self, entity_a_id: str, entity_b_id: str, score: float, method: str, reason: str | None = None,
    ) -> tuple[dict, bool]:
        left_id, right_id = sorted([entity_a_id, entity_b_id])
        return await self._add_resolution_candidate(
            "entity_duplicate", score, method, reason,
            left_entity_id=left_id, right_entity_id=right_id,
        )

    async def add_claim_resolution_candidate(
        self, claim_a_id: str, claim_b_id: str, score: float, method: str, reason: str | None = None,
    ) -> tuple[dict, bool]:
        left_id, right_id = sorted([claim_a_id, claim_b_id])
        return await self._add_resolution_candidate(
            "claim_duplicate", score, method, reason,
            left_claim_id=left_id, right_claim_id=right_id,
        )

    async def _add_resolution_candidate(
        self, candidate_type: str, score: float, method: str, reason: str | None,
        left_entity_id: str | None = None, right_entity_id: str | None = None,
        left_event_id: str | None = None, right_event_id: str | None = None,
        left_claim_id: str | None = None, right_claim_id: str | None = None,
    ) -> tuple[dict, bool]:
        """De-duplicates on (candidate_type, left/right id pair) in application
        code — a DB UNIQUE constraint can't do this here because Postgres (like
        standard SQL) never treats NULL as equal to NULL, and every row has at
        least one NULL id-pair (only one of entity/event/claim applies per
        candidate_type). IS NOT DISTINCT FROM is Postgres's NULL-safe equality
        operator, needed since some of these columns are NULL for any given row."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT * FROM resolution_candidates WHERE candidate_type = $1 "
                    "AND left_entity_id IS NOT DISTINCT FROM $2 "
                    "AND right_entity_id IS NOT DISTINCT FROM $3 "
                    "AND left_event_id IS NOT DISTINCT FROM $4 "
                    "AND right_event_id IS NOT DISTINCT FROM $5 "
                    "AND left_claim_id IS NOT DISTINCT FROM $6 "
                    "AND right_claim_id IS NOT DISTINCT FROM $7",
                    candidate_type, left_entity_id, right_entity_id,
                    left_event_id, right_event_id, left_claim_id, right_claim_id,
                )
                if existing:
                    return dict(existing), False

                candidate_id = _new_id()
                now = _now()
                row = await conn.fetchrow(
                    "INSERT INTO resolution_candidates "
                    "(id, candidate_type, left_entity_id, right_entity_id, left_event_id, right_event_id, "
                    "left_claim_id, right_claim_id, score, reason, method, status, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'open', $12, $13) "
                    "RETURNING *",
                    candidate_id, candidate_type, left_entity_id, right_entity_id,
                    left_event_id, right_event_id, left_claim_id, right_claim_id,
                    score, reason, method, now, now,
                )
        return dict(row), True

    async def list_resolution_candidates(
        self, candidate_type: str | None = None, status: str | None = "open", limit: int = 100,
    ) -> list[dict]:
        query = "SELECT * FROM resolution_candidates WHERE TRUE"
        params: list[Any] = []
        if candidate_type:
            params.append(candidate_type)
            query += f" AND candidate_type = ${len(params)}"
        if status:
            params.append(status)
            query += f" AND status = ${len(params)}"
        params.append(limit)
        query += f" ORDER BY score DESC LIMIT ${len(params)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(r) for r in rows]

    async def get_resolution_candidate(self, candidate_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM resolution_candidates WHERE id = $1", candidate_id)
        return dict(row) if row else None

    async def review_resolution_candidate(
        self, candidate_id: str, decision: str, reviewed_by: str | None = None,
    ) -> dict:
        """Changes the candidate's status only. Executing an accepted merge
        (reassigning evidence to one canonical row, tombstoning the other) is
        explicitly deferred — not implemented yet."""
        if decision not in ("accepted", "rejected"):
            raise ValueError(f"decision must be 'accepted' or 'rejected', got {decision!r}")
        now = _now()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE resolution_candidates SET status = $1, reviewed_by = $2, reviewed_at = $3, "
                "updated_at = $4 WHERE id = $5 RETURNING *",
                decision, reviewed_by, now, now, candidate_id,
            )
        return dict(row)

    # -- extraction runs & observations ------------------------------------

    async def find_extraction_run_by_signature(self, artifact_id: str, run_signature: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM extraction_runs WHERE artifact_id = $1 AND run_signature = $2 "
                "AND status = 'completed'",
                artifact_id, run_signature,
            )
        return dict(row) if row else None

    async def create_extraction_run(
        self, artifact_id: str, run_signature: str, model_id: str, prompt_name: str,
        prompt_version: str, extraction_schema_version: str, runtime: str | None = None,
        parameters: dict | None = None, chunk_count: int | None = None,
    ) -> dict:
        run_id = _new_id()
        now = _now()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO extraction_runs (id, artifact_id, run_signature, model_id, runtime, "
                "prompt_name, prompt_version, extraction_schema_version, parameters, status, "
                "chunk_count, observation_count, started_at, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'running', $10, 0, $11, $12, $13) "
                "RETURNING *",
                run_id, artifact_id, run_signature, model_id, runtime, prompt_name,
                prompt_version, extraction_schema_version, parameters, chunk_count, now, now, now,
            )
        return dict(row)

    async def complete_extraction_run(self, run_id: str, observation_count: int, status: str = "completed") -> None:
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE extraction_runs SET status = $1, observation_count = $2, "
                "completed_at = $3, updated_at = $4 WHERE id = $5",
                status, observation_count, now, now, run_id,
            )

    async def add_observation(
        self, extraction_run_id: str, artifact_chunk_id: str, raw_text: str,
        raw_payload: dict, confidence: float | None = None, importance_score: float | None = None,
        char_start: int | None = None, char_end: int | None = None,
        quote_match_type: str | None = None, observation_type: str = "claim",
    ) -> dict:
        obs_id = _new_id()
        now = _now()
        normalized = normalize_ws(raw_text).lower()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO extracted_observations "
                "(id, extraction_run_id, artifact_chunk_id, observation_type, raw_text, "
                "normalized_text, raw_payload, confidence, importance_score, char_start, "
                "char_end, quote_match_type, status, created_at, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'new', $13, $14) "
                "RETURNING *",
                obs_id, extraction_run_id, artifact_chunk_id, observation_type, raw_text,
                normalized, raw_payload, confidence, importance_score,
                char_start, char_end, quote_match_type, now, now,
            )
        return dict(row)

    async def list_observations(self, extraction_run_id: str, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM extracted_observations WHERE extraction_run_id = $1"
        params: list[Any] = [extraction_run_id]
        if status:
            params.append(status)
            query += f" AND status = ${len(params)}"
        query += " ORDER BY created_at"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(r) for r in rows]

    async def mark_observation_promoted(self, observation_id: str, claim_id: str) -> None:
        now = _now()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE extracted_observations SET status = 'promoted', candidate_claim_id = $1, "
                "updated_at = $2 WHERE id = $3",
                claim_id, now, observation_id,
            )

    async def get_artifact_chunk(self, chunk_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM artifact_chunks WHERE id = $1", chunk_id)
        return dict(row) if row else None
