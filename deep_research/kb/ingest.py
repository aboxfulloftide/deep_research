"""Source ingestion: web pages, YouTube transcripts, local files.

Build order step 2 (PLAN_KB_ARCHITECTURE.md). Each function registers/looks up
the canonical source, records a fetch attempt (success or failure), and — on
success — writes a versioned snapshot to disk and applies the retention policy
(first + newest two + retention_locked, decisions 13/16).

Chunking, clean-text extraction, and claim extraction are step 3+ and are not
part of this module.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from deep_research.config import Config
from deep_research.kb.canonical import (
    canonical_key_for_file_path,
    canonical_key_for_url,
    canonical_key_for_youtube,
    canonical_uri_for_youtube,
    infer_file_source_type,
    normalize_url,
    sha256_bytes,
    youtube_video_id_from_url,
)
from deep_research.kb.db import KBDatabase
from deep_research.kb.storage import SnapshotStore

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_FILE_MIME_TYPES = {
    "pdf": "application/pdf",
    "markdown": "text/markdown",
    "html_file": "text/html",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text": "text/plain",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class IngestResult:
    status: str  # "ingested" | "unchanged" | "failed"
    source_id: str | None = None
    source_created: bool = False
    version_id: str | None = None
    version_created: bool = False
    pruned_version_ids: list[str] | None = None
    error: str | None = None


async def _finalize_version(
    kb_db: KBDatabase,
    snapshot_store: SnapshotStore,
    source_id: str,
    content: bytes,
    content_hash: str,
    ext: str,
    http_status: int | None,
    mime_type: str | None,
    metadata: dict | None,
) -> tuple[dict, bool, list[str]]:
    """Write the snapshot (if the content actually changed) and enforce retention.

    Returns (version_row, version_created, pruned_version_ids).
    """
    latest = await kb_db.get_latest_version(source_id)
    if latest is not None and latest["content_hash"] == content_hash:
        return latest, False, []

    version_number = await kb_db.get_next_version_number(source_id)
    path = snapshot_store.write(source_id, version_number, content, ext)

    version_row, created = await kb_db.add_source_version(
        source_id=source_id,
        content_hash=content_hash,
        snapshot_path=str(path),
        http_status=http_status,
        mime_type=mime_type,
        byte_size=len(content),
        metadata=metadata,
    )

    pruned = await kb_db.prune_versions(source_id)
    pruned_ids = []
    for row in pruned:
        if row["id"] == version_row["id"]:
            continue
        snapshot_store.delete(row["snapshot_path"])
        pruned_ids.append(row["id"])

    return version_row, created, pruned_ids


async def ingest_web_page(
    url: str,
    config: Config,
    kb_db: KBDatabase,
    snapshot_store: SnapshotStore,
    trust_tier_code: str | None = None,
) -> IngestResult:
    normalized = normalize_url(url)
    canonical_key = canonical_key_for_url(url)
    started_at = _now()

    source, source_created = await kb_db.get_or_create_source(
        source_type_code="web",
        canonical_uri=normalized,
        canonical_key=canonical_key,
        trust_tier_code=trust_tier_code,
    )
    source_id = source["id"]

    try:
        async with httpx.AsyncClient(
            timeout=config.scraping.timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = "not_found" if e.response.status_code == 404 else "failed"
        await kb_db.add_fetch_attempt(
            source_id=source_id, attempt_type="fetch", status=status,
            requested_uri=url, http_status=e.response.status_code,
            error_code=str(e.response.status_code), error_message=str(e),
            started_at=started_at, completed_at=_now(),
        )
        return IngestResult(status="failed", source_id=source_id, source_created=source_created, error=str(e))
    except httpx.HTTPError as e:
        await kb_db.add_fetch_attempt(
            source_id=source_id, attempt_type="fetch", status="failed",
            requested_uri=url, error_code=type(e).__name__, error_message=str(e),
            started_at=started_at, completed_at=_now(),
        )
        return IngestResult(status="failed", source_id=source_id, source_created=source_created, error=str(e))

    content = resp.content
    content_hash = sha256_bytes(content)
    mime_type = resp.headers.get("content-type", "text/html").split(";")[0].strip()
    ext = ".html" if "html" in mime_type else ".dat"

    title = None
    if "html" in mime_type:
        try:
            soup = BeautifulSoup(content, "lxml")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = None
    if title:
        await kb_db.set_source_title_if_missing(source_id, title)

    version_row, version_created, pruned_ids = await _finalize_version(
        kb_db, snapshot_store, source_id, content, content_hash, ext,
        http_status=resp.status_code, mime_type=mime_type,
        metadata={"title": title} if title else None,
    )

    await kb_db.add_fetch_attempt(
        source_id=source_id, attempt_type="fetch", status="succeeded",
        requested_uri=url, source_version_id=version_row["id"],
        final_uri=str(resp.url), http_status=resp.status_code,
        started_at=started_at, completed_at=_now(),
        metadata=None if version_created else {"note": "content_unchanged"},
    )

    return IngestResult(
        status="ingested" if version_created else "unchanged",
        source_id=source_id, source_created=source_created,
        version_id=version_row["id"], version_created=version_created,
        pruned_version_ids=pruned_ids,
    )


async def ingest_youtube_video(
    url_or_id: str,
    kb_db: KBDatabase,
    snapshot_store: SnapshotStore,
    trust_tier_code: str | None = None,
) -> IngestResult:
    video_id = youtube_video_id_from_url(url_or_id)
    if video_id is None:
        return IngestResult(status="failed", error=f"Could not parse a YouTube video ID from {url_or_id!r}")

    canonical_key = canonical_key_for_youtube(video_id)
    canonical_uri = canonical_uri_for_youtube(video_id)
    started_at = _now()

    title = None
    author = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": canonical_uri, "format": "json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title")
                author = data.get("author_name")
    except httpx.HTTPError:
        pass  # oEmbed metadata is a nice-to-have; transcript ingestion continues regardless

    source, source_created = await kb_db.get_or_create_source(
        source_type_code="youtube_video",
        canonical_uri=canonical_uri,
        canonical_key=canonical_key,
        title=title,
        author=author,
        trust_tier_code=trust_tier_code,
    )
    source_id = source["id"]

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
    except YouTubeTranscriptApiException as e:
        await kb_db.add_fetch_attempt(
            source_id=source_id, attempt_type="transcript_fetch", status="failed",
            requested_uri=canonical_uri, error_code=type(e).__name__, error_message=str(e),
            started_at=started_at, completed_at=_now(),
        )
        return IngestResult(status="failed", source_id=source_id, source_created=source_created, error=str(e))

    raw_data = transcript.to_raw_data()
    content = json.dumps(raw_data, ensure_ascii=False).encode("utf-8")
    content_hash = sha256_bytes(content)

    version_row, version_created, pruned_ids = await _finalize_version(
        kb_db, snapshot_store, source_id, content, content_hash, ".json",
        http_status=None, mime_type="application/json",
        metadata={"language": getattr(transcript, "language", None)},
    )

    await kb_db.add_fetch_attempt(
        source_id=source_id, attempt_type="transcript_fetch", status="succeeded",
        requested_uri=canonical_uri, source_version_id=version_row["id"],
        started_at=started_at, completed_at=_now(),
        metadata=None if version_created else {"note": "content_unchanged"},
    )

    return IngestResult(
        status="ingested" if version_created else "unchanged",
        source_id=source_id, source_created=source_created,
        version_id=version_row["id"], version_created=version_created,
        pruned_version_ids=pruned_ids,
    )


async def ingest_file(
    path: str | Path,
    kb_db: KBDatabase,
    snapshot_store: SnapshotStore,
    trust_tier_code: str | None = None,
) -> IngestResult:
    file_path = Path(path).expanduser().resolve()
    started_at = _now()

    if not file_path.exists() or not file_path.is_file():
        await kb_db.add_fetch_attempt(
            source_id=None, attempt_type="fetch", status="not_found",
            requested_uri=str(file_path), error_code="file_not_found",
            error_message=f"No such file: {file_path}",
            started_at=started_at, completed_at=_now(),
        )
        return IngestResult(status="failed", error=f"No such file: {file_path}")

    content = file_path.read_bytes()
    content_hash = sha256_bytes(content)
    source_type_code = infer_file_source_type(file_path)
    canonical_key = canonical_key_for_file_path(file_path)

    source, source_created = await kb_db.get_or_create_source(
        source_type_code=source_type_code,
        canonical_uri=str(file_path),
        canonical_key=canonical_key,
        title=file_path.stem,
        trust_tier_code=trust_tier_code,
    )
    source_id = source["id"]

    mime_type = _FILE_MIME_TYPES.get(source_type_code, "application/octet-stream")
    ext = file_path.suffix or ".dat"

    version_row, version_created, pruned_ids = await _finalize_version(
        kb_db, snapshot_store, source_id, content, content_hash, ext,
        http_status=None, mime_type=mime_type, metadata=None,
    )

    await kb_db.add_fetch_attempt(
        source_id=source_id, attempt_type="fetch", status="succeeded",
        requested_uri=str(file_path), source_version_id=version_row["id"],
        started_at=started_at, completed_at=_now(),
        metadata=None if version_created else {"note": "content_unchanged"},
    )

    return IngestResult(
        status="ingested" if version_created else "unchanged",
        source_id=source_id, source_created=source_created,
        version_id=version_row["id"], version_created=version_created,
        pruned_version_ids=pruned_ids,
    )
