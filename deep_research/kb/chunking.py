"""Pure chunking and text-normalization functions — no I/O. Naive fixed-size
chunking is intentional here: PLAN_KB_ARCHITECTURE.md explicitly defers smarter
(semantic/structural) chunking, and the step-0 spike validated that extraction
quality holds up fine on naive chunks.
"""

import re

DEFAULT_CHUNK_SIZE_CHARS = 1200


def normalize_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_name(s: str | None) -> str:
    """Exact-match key for entities/events (decision 25): lowercase,
    punctuation-stripped. This is the *only* auto-merge tier — anything less
    than an exact match on this key must go through resolution_candidates."""
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def find_quote(quote: str, chunk_text: str) -> tuple[str, int, int] | None:
    """Locate a model-reported supporting quote inside the chunk it supposedly
    came from. Whitespace-normalized exact/case-insensitive substring match —
    validated in the step-0 spike at 96% match rate, with the misses being
    genuine excerpts with minor drift, never fabrication."""
    q = normalize_ws(quote)
    c = normalize_ws(chunk_text)
    if not q:
        return None
    idx = c.find(q)
    if idx >= 0:
        return ("exact", idx, idx + len(q))
    idx_ci = c.lower().find(q.lower())
    if idx_ci >= 0:
        return ("case_insensitive", idx_ci, idx_ci + len(q))
    return None


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE_CHARS) -> list[tuple[str, int, int]]:
    """Fixed-size chunking snapped to the nearest whitespace boundary.

    Returns a list of (chunk_text, char_start, char_end) against the original text.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            # Preserve accurate offsets against the original text even though
            # leading/trailing whitespace is stripped from the stored chunk.
            lstrip_amount = len(raw) - len(raw.lstrip())
            chunk_start = start + lstrip_amount
            chunk_end = chunk_start + len(stripped)
            chunks.append((stripped, chunk_start, chunk_end))
        start = end
    return chunks


def chunk_transcript_segments(
    segments: list[dict], size: int = DEFAULT_CHUNK_SIZE_CHARS
) -> list[tuple[str, float, float]]:
    """Group transcript segments (dicts with text/start/duration) into ~size-char
    chunks, keeping the time range each chunk actually spans.

    Returns a list of (chunk_text, time_start_seconds, time_end_seconds).
    """
    chunks = []
    buf_text: list[str] = []
    buf_len = 0
    t_start = None
    t_end = None

    for seg in segments:
        if t_start is None:
            t_start = seg["start"]
        buf_text.append(seg["text"])
        buf_len += len(seg["text"]) + 1
        t_end = seg["start"] + seg["duration"]
        if buf_len >= size:
            chunks.append((" ".join(buf_text), t_start, t_end))
            buf_text, buf_len, t_start, t_end = [], 0, None, None

    if buf_text:
        chunks.append((" ".join(buf_text), t_start, t_end))

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough chars-per-token heuristic — good enough for storage metadata, not
    a real tokenizer."""
    return max(1, len(text) // 4)
