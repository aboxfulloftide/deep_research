"""Extraction + resolution spike (step 0 of PLAN_KB_ARCHITECTURE.md).

Throwaway harness. Loads one article + one YouTube transcript on the same
topic, chunks them, runs claim/entity/event extraction against the local
fast model, and writes results to SQLite + JSONL for hand inspection.

Do not extend this into production code — see spike/README.md.
"""

import asyncio
import json
import re
import sqlite3
import statistics
import sys
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.config import Config, LLMConfig  # noqa: E402
from deep_research.llm import LLMClient  # noqa: E402
from deep_research.tools.scrape import _extract_text  # noqa: E402

from youtube_transcript_api import YouTubeTranscriptApi  # noqa: E402

ARTICLE_URL = (
    "https://www.oliverwyman.com/our-expertise/insights/2026/jan/"
    "impact-ai-bubble-burst-on-global-financial-markets.html"
)
YOUTUBE_VIDEO_ID = "2J2Fb1bBufA"

BASE_URL = "http://localhost:8080/v1"
PROMPT_VERSION = "spike-v1"

CHUNK_SIZE_CHARS = 1200

OUTPUT_DIR = Path(__file__).parent / "output"
DB_PATH = OUTPUT_DIR / "spike.db"
JSONL_PATH = OUTPUT_DIR / "observations.jsonl"

EXTRACTION_SYSTEM_PROMPT = """/no_think
You are a claim extraction engine for a knowledge base. You will be given one chunk of text from a source. Extract atomic factual claims made in this chunk.

For each claim, output an object with these exact fields:
- claim_text: a single atomic factual statement, in your own words, not a blob of multiple facts
- claim_type: one of "fact", "event_fact", "economic", "historical", "product_spec", "quote"
- entities: array of {"name": str, "type": one of "person","organization","product","location","concept"} mentioned in the claim
- event: {"title": str, "date": str or null} if the claim describes a dated/time-bound happening, otherwise null
- supporting_quote: a short VERBATIM excerpt (max ~30 words) copied EXACTLY from the chunk text that supports this claim. Do not paraphrase this field.
- confidence: your confidence 0.0-1.0 that this claim is accurately extracted from the text
- importance: your estimate 0.0-1.0 of how important/central this claim is to the source's overall point

Rules:
- Only extract claims actually present in the chunk. Do not invent facts, numbers, or dates not present in the text.
- Keep each claim atomic: one fact per claim.
- If the chunk has no extractable claims, return an empty array.
- Return ONLY a JSON array. No prose, no markdown fences, no explanation.
"""


def normalize_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def normalize_name(s: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def find_quote(quote: str, chunk_text: str) -> tuple[str, int, int] | None:
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


def chunk_text_naive(text: str, size: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Fixed-size chunking, snapped to the nearest whitespace boundary."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def chunk_transcript(segments, size: int = CHUNK_SIZE_CHARS):
    """Group transcript segments into ~size-char chunks, keeping time bounds."""
    chunks = []
    buf_text = []
    buf_len = 0
    t_start = None
    t_end = None
    for seg in segments:
        if t_start is None:
            t_start = seg.start
        buf_text.append(seg.text)
        buf_len += len(seg.text) + 1
        t_end = seg.start + seg.duration
        if buf_len >= size:
            chunks.append((" ".join(buf_text), t_start, t_end))
            buf_text, buf_len, t_start, t_end = [], 0, None, None
    if buf_text:
        chunks.append((" ".join(buf_text), t_start, t_end))
    return chunks


async def detect_model(base_url: str) -> str:
    """Ask the llama.cpp server which model it currently has loaded."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base_url}/models")
        resp.raise_for_status()
        data = resp.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by server at {base_url}")
    return models[0]["id"]


async def fetch_article(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(
        timeout=15,
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    title, text = _extract_text(html)
    return title, text


def fetch_transcript(video_id: str):
    api = YouTubeTranscriptApi()
    return api.fetch(video_id)


def parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            data = json.loads(match.group(1))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    return []


def init_db(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE spike_chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            time_start_seconds REAL,
            time_end_seconds REAL
        );

        CREATE TABLE spike_extraction_runs (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            status TEXT NOT NULL,
            chunk_count INTEGER,
            observation_count INTEGER,
            started_at REAL,
            completed_at REAL
        );

        CREATE TABLE spike_extracted_observations (
            id TEXT PRIMARY KEY,
            extraction_run_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            source TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            claim_type TEXT,
            entities_json TEXT,
            event_json TEXT,
            supporting_quote TEXT,
            quote_match_type TEXT,
            quote_char_start INTEGER,
            quote_char_end INTEGER,
            confidence REAL,
            importance REAL,
            raw_payload_json TEXT
        );

        CREATE TABLE spike_resolution_candidates (
            id TEXT PRIMARY KEY,
            candidate_type TEXT NOT NULL,
            left_observation_id TEXT NOT NULL,
            right_observation_id TEXT NOT NULL,
            score REAL,
            method TEXT,
            reason TEXT
        );
        """
    )
    conn.commit()


async def extract_source(
    llm: LLMClient,
    conn: sqlite3.Connection,
    source: str,
    chunks: list[tuple[str, float | None, float | None]],
    jsonl_f,
    model: str,
) -> None:
    run_id = str(uuid.uuid4())
    started_at = time.time()
    conn.execute(
        "INSERT INTO spike_extraction_runs "
        "(id, source, model, prompt_version, status, chunk_count, observation_count, started_at, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, source, model, PROMPT_VERSION, "running", len(chunks), 0, started_at, None),
    )
    conn.commit()

    total_observations = 0

    for idx, (chunk_text, t_start, t_end) in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO spike_chunks (id, source, chunk_index, chunk_text, time_start_seconds, time_end_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, source, idx, chunk_text, t_start, t_end),
        )
        conn.commit()

        print(f"  [{source}] chunk {idx + 1}/{len(chunks)} ({len(chunk_text)} chars)...", flush=True)

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Chunk text:\n\n{chunk_text}"},
        ]
        try:
            resp = await llm.chat(messages)
            content = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"    ERROR calling model: {e}", flush=True)
            continue

        claims = parse_json_array(content)
        if not claims:
            print(f"    no claims parsed (raw len={len(content)})", flush=True)

        for claim in claims:
            if not isinstance(claim, dict) or not claim.get("claim_text"):
                continue
            obs_id = str(uuid.uuid4())
            quote = claim.get("supporting_quote", "")
            match = find_quote(quote, chunk_text)
            match_type, q_start, q_end = (match or (None, None, None))

            entities_json = json.dumps(claim.get("entities") or [])
            event_json = json.dumps(claim.get("event")) if claim.get("event") else None

            conn.execute(
                "INSERT INTO spike_extracted_observations "
                "(id, extraction_run_id, chunk_id, source, claim_text, claim_type, entities_json, event_json, "
                "supporting_quote, quote_match_type, quote_char_start, quote_char_end, confidence, importance, raw_payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    obs_id,
                    run_id,
                    chunk_id,
                    source,
                    claim.get("claim_text"),
                    claim.get("claim_type"),
                    entities_json,
                    event_json,
                    quote,
                    match_type,
                    q_start,
                    q_end,
                    claim.get("confidence"),
                    claim.get("importance"),
                    json.dumps(claim),
                ),
            )
            jsonl_f.write(
                json.dumps(
                    {
                        "id": obs_id,
                        "source": source,
                        "chunk_index": idx,
                        "claim_text": claim.get("claim_text"),
                        "claim_type": claim.get("claim_type"),
                        "entities": claim.get("entities"),
                        "event": claim.get("event"),
                        "supporting_quote": quote,
                        "quote_match_type": match_type,
                        "confidence": claim.get("confidence"),
                        "importance": claim.get("importance"),
                    }
                )
                + "\n"
            )
            total_observations += 1
        conn.commit()

    conn.execute(
        "UPDATE spike_extraction_runs SET status = ?, observation_count = ?, completed_at = ? WHERE id = ?",
        ("completed", total_observations, time.time(), run_id),
    )
    conn.commit()
    print(f"  [{source}] done: {total_observations} observations from {len(chunks)} chunks", flush=True)


def find_duplicate_candidates(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT id, source, claim_text, entities_json, event_json FROM spike_extracted_observations"
    ).fetchall()

    candidates = []

    # Entity duplicate candidates: compare every pair of entity mentions from different sources
    entity_mentions = []  # (obs_id, source, entity_name, normalized_name)
    for obs_id, source, _claim_text, entities_json, _event_json in rows:
        try:
            entities = json.loads(entities_json) if entities_json else []
        except json.JSONDecodeError:
            entities = []
        for ent in entities:
            name = ent.get("name") if isinstance(ent, dict) else None
            if name:
                entity_mentions.append((obs_id, source, name, normalize_name(name)))

    seen_pairs = set()
    for i in range(len(entity_mentions)):
        for j in range(i + 1, len(entity_mentions)):
            oid_a, src_a, name_a, norm_a = entity_mentions[i]
            oid_b, src_b, name_b, norm_b = entity_mentions[j]
            if src_a == src_b or not norm_a or not norm_b:
                continue
            key = tuple(sorted([norm_a, norm_b]))
            if norm_a == norm_b:
                score, method, reason = 1.0, "normalized_text", f"exact normalized match: '{name_a}' == '{name_b}'"
            elif norm_a in norm_b or norm_b in norm_a:
                score, method, reason = 0.85, "substring", f"substring match: '{name_a}' / '{name_b}'"
            else:
                ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
                if ratio < 0.82:
                    continue
                score, method, reason = ratio, "trigram", f"fuzzy match ({ratio:.2f}): '{name_a}' / '{name_b}'"
            pair_key = (key, oid_a, oid_b)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            candidates.append(("entity_duplicate", oid_a, oid_b, score, method, reason))

    # Claim duplicate candidates: fuzzy match claim_text across different sources
    claim_rows = [(r[0], r[1], normalize_ws(r[2])) for r in rows]
    for i in range(len(claim_rows)):
        for j in range(i + 1, len(claim_rows)):
            oid_a, src_a, text_a = claim_rows[i]
            oid_b, src_b, text_b = claim_rows[j]
            if src_a == src_b or not text_a or not text_b:
                continue
            ratio = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
            if ratio >= 0.6:
                candidates.append(
                    (
                        "claim_duplicate",
                        oid_a,
                        oid_b,
                        ratio,
                        "trigram",
                        f"fuzzy claim match ({ratio:.2f})",
                    )
                )

    for candidate_type, oid_a, oid_b, score, method, reason in candidates:
        conn.execute(
            "INSERT INTO spike_resolution_candidates "
            "(id, candidate_type, left_observation_id, right_observation_id, score, method, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), candidate_type, oid_a, oid_b, score, method, reason),
        )
    conn.commit()
    return candidates


def print_summary(conn: sqlite3.Connection):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for (source,) in conn.execute("SELECT DISTINCT source FROM spike_extracted_observations"):
        count = conn.execute(
            "SELECT COUNT(*) FROM spike_extracted_observations WHERE source = ?", (source,)
        ).fetchone()[0]
        print(f"\nSource: {source}")
        print(f"  claims extracted: {count}")

        entity_names = set()
        for (entities_json,) in conn.execute(
            "SELECT entities_json FROM spike_extracted_observations WHERE source = ?", (source,)
        ):
            try:
                entities = json.loads(entities_json) if entities_json else []
            except json.JSONDecodeError:
                entities = []
            for ent in entities:
                name = ent.get("name") if isinstance(ent, dict) else None
                if name:
                    entity_names.add(normalize_name(name))
        print(f"  roughly-unique entities: {len(entity_names)}")

        quote_rows = conn.execute(
            "SELECT quote_match_type FROM spike_extracted_observations WHERE source = ?", (source,)
        ).fetchall()
        total = len(quote_rows)
        matched = sum(1 for (m,) in quote_rows if m in ("exact", "case_insensitive"))
        pct = (matched / total * 100) if total else 0.0
        print(f"  supporting_quote verbatim-matched in chunk: {matched}/{total} ({pct:.0f}%)")

        confidences = [
            c
            for (c,) in conn.execute(
                "SELECT confidence FROM spike_extracted_observations WHERE source = ? AND confidence IS NOT NULL",
                (source,),
            )
        ]
        importances = [
            i
            for (i,) in conn.execute(
                "SELECT importance FROM spike_extracted_observations WHERE source = ? AND importance IS NOT NULL",
                (source,),
            )
        ]
        if confidences:
            print(
                f"  confidence: min={min(confidences):.2f} mean={statistics.mean(confidences):.2f} max={max(confidences):.2f}"
            )
        if importances:
            print(
                f"  importance: min={min(importances):.2f} mean={statistics.mean(importances):.2f} max={max(importances):.2f}"
            )

    print("\nCross-source resolution candidates:")
    for (candidate_type,) in conn.execute("SELECT DISTINCT candidate_type FROM spike_resolution_candidates"):
        count = conn.execute(
            "SELECT COUNT(*) FROM spike_resolution_candidates WHERE candidate_type = ?", (candidate_type,)
        ).fetchone()[0]
        print(f"  {candidate_type}: {count}")

    print(f"\nFull observation dump: {JSONL_PATH}")
    print(f"SQLite database: {DB_PATH}")
    print("=" * 70)


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print(f"Fetching article: {ARTICLE_URL}")
    title, article_text = await fetch_article(ARTICLE_URL)
    print(f"  title: {title!r}, {len(article_text)} chars")

    print(f"Fetching YouTube transcript: {YOUTUBE_VIDEO_ID}")
    segments = fetch_transcript(YOUTUBE_VIDEO_ID)
    print(f"  {len(segments)} segments")

    article_chunks = [(c, None, None) for c in chunk_text_naive(article_text)]
    transcript_chunks = chunk_transcript(segments)
    print(f"article chunks: {len(article_chunks)}, transcript chunks: {len(transcript_chunks)}")

    model = await detect_model(BASE_URL)
    print(f"Using model reported by llama.cpp server: {model}")
    config = Config(llm=LLMConfig(base_url=BASE_URL, model=model, api_key="not-needed"))
    llm = LLMClient(config)

    with open(JSONL_PATH, "w") as jsonl_f:
        try:
            print("\nExtracting from article...")
            await extract_source(llm, conn, "article", article_chunks, jsonl_f, model)

            print("\nExtracting from youtube_transcript...")
            await extract_source(llm, conn, "youtube_transcript", transcript_chunks, jsonl_f, model)
        finally:
            await llm.close()

    print("\nFinding resolution candidates...")
    find_duplicate_candidates(conn)

    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
