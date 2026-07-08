"""Pure chunking functions — no I/O. Naive fixed-size chunking is intentional
here: PLAN_KB_ARCHITECTURE.md explicitly defers smarter (semantic/structural)
chunking, and the step-0 spike validated that extraction quality holds up
fine on naive chunks.
"""

DEFAULT_CHUNK_SIZE_CHARS = 1200


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
