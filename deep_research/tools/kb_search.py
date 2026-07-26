"""Local knowledge-base search as a research-agent tool.

Wires the chat research agent (deep_research/agent.py, predates the KB by
several build-order steps and never consulted it) into the KB's existing FTS5
search from build order step 3 — this is decision 23's "prefer stored
knowledge first unless stale or incomplete" hybrid retrieval, applied to the
agent the user actually talks to, not just the Topics view.

Step 8 adds a second retrieval signal on top of keyword FTS: embedding-based
semantic search. Neither signal alone is sufficient — FTS catches exact terms
but misses paraphrases ("almost double GDP" vs. "nearly 2x GDP"); vector
search catches paraphrases but is weaker on exact numbers/names/codes. Both
result sets are combined via Reciprocal Rank Fusion (RRF), a standard
rank-based blend that needs no score-scale tuning between the two signals.
"""

from deep_research.config import Config
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import embed_texts

RRF_K = 60  # standard RRF damping constant; de-emphasizes rank differences past the top few


def _location_string(record: dict) -> str:
    location = f"chunk {record['chunk_index']}"
    if record.get("page_number") is not None:
        location += f", page {record['page_number']}"
    if record.get("time_start_seconds") is not None:
        location += f", t={record['time_start_seconds']:.0f}s"
    return location


async def kb_search_records(query: str, kb_db: KBDatabase, config: Config, limit: int = 5) -> list[dict]:
    """Structured hybrid (FTS + semantic, RRF-fused) chunk records, most
    relevant first -- each dict carries chunk_id, rrf_score, location
    (chunk/page/timestamp), title, canonical_uri, and the full chunk_text
    (not just a rendered snippet), so a structured consumer (e.g. a future
    local-KB-first research adapter) doesn't need to re-parse kb_search()'s
    formatted string."""
    fts_results = await kb_db.search_chunks(query, limit=limit * 4)

    semantic_results = []
    try:
        vectors = await embed_texts([query], config.kb.embedding_base_url, config.kb.embedding_model)
        semantic_results = await kb_db.search_chunks_semantic(vectors[0], limit=limit * 4)
    except Exception:
        pass  # best-effort: fall back to FTS-only if the embedding backend is unreachable

    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for result_list in (fts_results, semantic_results):
        for rank, r in enumerate(result_list):
            scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0.0) + 1 / (RRF_K + rank + 1)
            rows.setdefault(r["chunk_id"], r)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]

    records = []
    for chunk_id in ranked_ids:
        r = rows[chunk_id]
        records.append({
            "chunk_id": chunk_id,
            "rrf_score": scores[chunk_id],
            "chunk_index": r["chunk_index"],
            "page_number": r.get("page_number"),
            "time_start_seconds": r.get("time_start_seconds"),
            "title": r.get("source_title") or r.get("canonical_uri"),
            "canonical_uri": r.get("canonical_uri"),
            "chunk_text": r.get("chunk_text") or "",
            "snippet": r.get("snippet") or (r.get("chunk_text") or "")[:400],
        })
    return records


async def kb_search(query: str, kb_db: KBDatabase, config: Config, limit: int = 5) -> str:
    """Search the local knowledge base's chunked content. Formatted like
    web_search's output so the agent can reason about/cite it the same way.
    Output is unchanged from before kb_search_records() was split out --
    this is now a thin formatting wrapper over it."""
    records = await kb_search_records(query, kb_db, config, limit=limit)
    if not records:
        return "No results found in the local knowledge base."

    return "\n".join(
        f"**{record['title']}** ({_location_string(record)})\n{record['snippet']}\n"
        for record in records
    )
