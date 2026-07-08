"""Canonical identity rules for knowledge-base sources.

Every source needs a stable `canonical_key` so the same web page, YouTube video,
or file is recognized as one source across repeated ingestion, per the
"Source identity and lifecycle rules" section of PLAN_KB_ARCHITECTURE.md.
"""

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query params that don't change page identity — stripped before hashing/comparing.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src",
    "igshid", "spm", "_hsenc", "_hsmi",
}

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_FILE_SOURCE_TYPES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html_file",
    ".htm": "html_file",
    ".docx": "docx",
    ".txt": "text",
}


def normalize_url(url: str) -> str:
    """Normalize a URL for stable dedupe: lowercase host, drop default port,
    drop fragment, strip tracking params, sort remaining params, drop trailing
    slash (except root)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    host = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    if parts.username:
        userinfo = parts.username + (f":{parts.password}" if parts.password else "")
        netloc = f"{userinfo}@{netloc}"

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query_pairs.sort()
    query = urlencode(query_pairs)

    return urlunsplit((scheme, netloc, path, query, ""))


def canonical_key_for_url(url: str) -> str:
    return f"web:{normalize_url(url)}"


def youtube_video_id_from_url(url_or_id: str) -> str | None:
    """Extract an 11-char YouTube video ID from any common URL shape, or pass
    through a bare video ID."""
    candidate = url_or_id.strip()
    if _YOUTUBE_ID_RE.match(candidate):
        return candidate

    parts = urlsplit(candidate)
    host = (parts.hostname or "").lower().removeprefix("www.").removeprefix("m.")

    if host == "youtu.be":
        video_id = parts.path.lstrip("/").split("/")[0]
        return video_id if _YOUTUBE_ID_RE.match(video_id) else None

    if host in ("youtube.com", "music.youtube.com"):
        if parts.path == "/watch":
            qs = dict(parse_qsl(parts.query))
            video_id = qs.get("v", "")
            return video_id if _YOUTUBE_ID_RE.match(video_id) else None
        for prefix in ("/embed/", "/shorts/", "/live/", "/v/"):
            if parts.path.startswith(prefix):
                video_id = parts.path[len(prefix):].split("/")[0]
                return video_id if _YOUTUBE_ID_RE.match(video_id) else None

    return None


def canonical_key_for_youtube(video_id: str) -> str:
    return f"youtube_video:{video_id}"


def canonical_uri_for_youtube(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_key_for_file_path(path: Path) -> str:
    """File identity is the path, not the content hash — the content hash is used
    per-version to detect changes (like a web page's URL vs. its fetched bytes).
    Keying identity on content hash would make every edit a brand new unrelated
    source instead of a new version of the same one, defeating versioning."""
    return f"file:{path.resolve()}"


def infer_file_source_type(path: Path) -> str:
    return _FILE_SOURCE_TYPES.get(path.suffix.lower(), "text")
