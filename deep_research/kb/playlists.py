"""YouTube playlist discovery; playlists are discovery metadata, never sources."""

import asyncio
import json
import sys
from urllib.parse import parse_qs, urlparse

from deep_research.config import Config
from deep_research.kb.db import KBDatabase
from deep_research.kb.ingest import ingest_youtube_video
from deep_research.kb.jobs import enqueue_source_pipeline
from deep_research.kb.storage import SnapshotStore


def youtube_playlist_id(url: str) -> str | None:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("list", [None])[0]


async def track_youtube_playlist(kb_db: KBDatabase, url: str, trust_tier_code: str | None = None) -> tuple[dict, bool]:
    playlist_id = youtube_playlist_id(url)
    if not playlist_id:
        raise ValueError("Could not parse a YouTube playlist ID from the URL")
    return await kb_db.get_or_create_tracked_playlist(
        "youtube", playlist_id, url, default_trust_tier_code=trust_tier_code,
    )


async def enumerate_youtube_playlist(url: str) -> list[dict]:
    """Use yt-dlp's flat mode: no API key/quota and no video downloads."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-single-json", "--no-warnings", url,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"yt-dlp playlist enumeration failed: {stderr.decode(errors='replace').strip()}")
    payload = json.loads(stdout)
    return [
        {"video_id": item.get("id"), "title": item.get("title")}
        for item in payload.get("entries", []) if item and item.get("id")
    ]


async def poll_playlist(
    kb_db: KBDatabase, config: Config, snapshot_store: SnapshotStore, playlist_id: str,
    *, limit: int | None = None,
) -> dict:
    playlists = await kb_db.list_tracked_playlists()
    playlist = next((p for p in playlists if p["id"] == playlist_id), None)
    if not playlist:
        raise ValueError(f"No tracked playlist {playlist_id!r}")
    videos = await enumerate_youtube_playlist(playlist["url"])
    discovered = 0
    for video in videos:
        _, created = await kb_db.add_playlist_video(playlist_id, video["video_id"], video.get("title"))
        discovered += created
    queued = 0
    batch_limit = limit if limit is not None else config.kb.playlist_max_videos_per_run
    for video in await kb_db.list_pending_playlist_videos(playlist_id, batch_limit):
        result = await ingest_youtube_video(
            video["video_id"], kb_db, snapshot_store,
            trust_tier_code=video.get("default_trust_tier_code"), source_purpose="playlist_discovered",
        )
        if result.source_id:
            await kb_db.mark_playlist_video_ingested(playlist_id, video["video_id"], result.source_id)
            await kb_db.record_decision(
                "playlist_video_ingested", "source", result.source_id, "ingested from tracked playlist",
                related_ids=[playlist_id, video["video_id"]],
                resulting_state={"source_purpose": "playlist_discovered"}, reversible=False,
            )
            if result.version_id:
                await enqueue_source_pipeline(
                    kb_db, result.source_id, result.version_id, priority=10,
                )
            queued += 1
    pending_after = len(await kb_db.list_pending_playlist_videos(playlist_id, 1))
    return {"discovered": discovered, "queued": queued, "pending_after": pending_after}
