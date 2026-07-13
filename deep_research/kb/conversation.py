"""Parses a pasted chat conversation into (speaker, turn_text) pairs so
claims extracted from it can be attributed to whoever said them -- a
conversation can be between two or more people (or a person and an AI), and
every claim gets fact-checked regardless of who made it.

Heuristic, not a real chat-export parser: a line starting with a short
name-like token followed by a colon (e.g. "Alice:", "User:", "ChatGPT:")
starts a new turn; everything until the next such line belongs to that
speaker. Text with no recognizable speaker lines at all is returned as one
untagged turn (speaker=None), so a plain-prose paste still works like any
other document instead of failing to parse.
"""

import re

from deep_research.kb.db import KBDatabase

_SPEAKER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ._'-]{0,40}):\s?(.*)$")


def parse_conversation_turns(text: str) -> list[tuple[str | None, str]]:
    turns: list[tuple[str | None, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            turns.append((current_speaker, content))

    for line in text.splitlines():
        match = _SPEAKER_LINE_RE.match(line)
        if match:
            flush()
            current_speaker = match.group(1).strip()
            current_lines = [match.group(2)] if match.group(2) else []
        else:
            current_lines.append(line)
    flush()

    if not turns:
        return [(None, text.strip())] if text.strip() else []

    # If nothing ever looked like a real speaker line (every "turn" is
    # actually just the whole text under one label from line 1), it's
    # probably not a conversation -- but this is rare enough in practice
    # (a real transcript has multiple turns) not to special-case further.
    return turns


async def get_topic_conversation_transcript(kb_db: KBDatabase, topic_id: str) -> list[dict]:
    """Reconstructs the ordered turn-by-turn transcript for every pasted
    conversation attached to a topic, for the Topic page's timeline: read
    the conversation with each claim shown inline, right after the turn
    that said it, instead of a flat claims list with no back-and-forth
    context. Only claims actually attached to this topic are included per
    turn (a chunk can in principle carry evidence for a claim that was
    reviewed and rejected from this topic -- that shouldn't resurface here).

    A source counts as a transcript if its current artifact_type is
    "conversation_turns" (set once, at ingest, by build_artifact_for_version
    for the "conversation" source type -- see artifacts.py) -- not whether
    its chunks happen to carry a section_label speaker tag, since
    parse_conversation_turns falls back to one *untagged* turn (speaker=
    None) for a paste with no recognizable "Name:" lines, and that
    untagged turn still gets chunk-split like any long text. Checking
    artifact_type catches those too instead of silently dropping them.
    Multiple pasted conversations under one topic are concatenated in
    attach order, oldest first."""
    sources = await kb_db.list_topic_sources(topic_id, link_status="attached")
    sources = sorted(sources, key=lambda s: s["created_at"])

    turns: list[dict] = []
    for source in sources:
        version = await kb_db.get_latest_version(source["id"])
        if version is None:
            continue
        artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
        if not artifacts or artifacts[0]["artifact_type"] != "conversation_turns":
            continue
        chunks = await kb_db.list_chunks(artifacts[0]["id"])
        for chunk in chunks:
            turns.append({
                "source_id": source["id"],
                "source_title": source["title"],
                "speaker": chunk["section_label"],
                "text": chunk["chunk_text"],
                "chunk_id": chunk["id"],
                "claims": [],
            })
    if not turns:
        return turns

    attached_claim_ids = await kb_db.get_topic_claim_ids(topic_id, link_status="attached")
    claims_by_chunk = await kb_db.get_claims_by_chunk_ids([t["chunk_id"] for t in turns])
    for turn in turns:
        turn["claims"] = [
            c for c in claims_by_chunk.get(turn["chunk_id"], []) if c["id"] in attached_claim_ids
        ]
    return turns
