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


async def kb_search(query: str, kb_db: KBDatabase, config: Config, limit: int = 5) -> str:
    """Search the local knowledge base's chunked content. Formatted like
    web_search's output so the agent can reason about/cite it the same way."""
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

    if not scores:
        return "No results found in the local knowledge base."

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]

    lines = []
    for chunk_id in ranked_ids:
        r = rows[chunk_id]
        location = f"chunk {r['chunk_index']}"
        if r.get("page_number") is not None:
            location += f", page {r['page_number']}"
        if r.get("time_start_seconds") is not None:
            location += f", t={r['time_start_seconds']:.0f}s"
        title = r.get("source_title") or r.get("canonical_uri")
        snippet = r.get("snippet") or (r.get("chunk_text") or "")[:400]
        lines.append(f"**{title}** ({location})\n{snippet}\n")
    return "\n".join(lines)
