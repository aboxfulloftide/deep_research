"""Local knowledge-base search as a research-agent tool.

Wires the chat research agent (deep_research/agent.py, predates the KB by
several build-order steps and never consulted it) into the KB's existing FTS5
search from build order step 3 — this is decision 23's "prefer stored
knowledge first unless stale or incomplete" hybrid retrieval, applied to the
agent the user actually talks to, not just the Topics view.
"""

from deep_research.kb.db import KBDatabase


async def kb_search(query: str, kb_db: KBDatabase, limit: int = 5) -> str:
    """Search the local knowledge base's chunked content. Formatted like
    web_search's output so the agent can reason about/cite it the same way."""
    results = await kb_db.search_chunks(query, limit=limit)
    if not results:
        return "No results found in the local knowledge base."

    lines = []
    for r in results:
        location = f"chunk {r['chunk_index']}"
        if r.get("page_number") is not None:
            location += f", page {r['page_number']}"
        if r.get("time_start_seconds") is not None:
            location += f", t={r['time_start_seconds']:.0f}s"
        title = r.get("source_title") or r.get("canonical_uri")
        lines.append(f"**{title}** ({location})\n{r['snippet']}\n")
    return "\n".join(lines)
