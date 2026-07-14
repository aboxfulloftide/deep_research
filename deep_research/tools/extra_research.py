"""Bounded three-level web research for the interactive Extra mode.

This intentionally has hard caps. "Deep" should mean progressively better
evidence, not an unbounded number of browser requests or a prompt too large
for a practical local model to synthesize.
"""

import asyncio
import re
from dataclasses import dataclass

from deep_research.config import Config
from deep_research.llm import LLMClient
from deep_research.models import SearchResult
from deep_research.tools.scrape import scrape_page
from deep_research.tools.search import web_search

SOURCES_PER_QUERY = 1
SOURCE_EXCERPT_CHARS = 3_000
FOLLOW_UP_QUERY_LIMIT = 2


@dataclass
class ResearchSource:
    title: str
    url: str
    content: str
    level: int
    query: str


def _is_html_result(result: SearchResult) -> bool:
    return not result.url.lower().split("?", 1)[0].endswith(".pdf")


def _title_key(title: str) -> str:
    """Identify syndicated copies that have different URLs but the same story."""
    article_title = re.split(r"\s+[|–—-]\s+", title, maxsplit=1)[0]
    return "title:" + re.sub(r"\W+", " ", article_title.lower()).strip()


async def collect_sources(
    queries: list[str], config: Config, level: int, seen_urls: set[str],
) -> list[ResearchSource]:
    """Search each query and read new HTML sources from it.

    The first level receives two independent starting sources. Later levels
    each have two focused queries, so one source per query keeps the final
    synthesis at six readable sources.
    """
    selections: list[tuple[str, SearchResult]] = []
    pending_urls: set[str] = set()
    pending_titles: set[str] = set()
    per_query_limit = 2 if len(queries) == 1 else SOURCES_PER_QUERY
    for query in queries:
        try:
            results = await web_search(query, config)
        except Exception:
            continue
        added = 0
        for result in results:
            title_key = _title_key(result.title)
            if (
                result.url in seen_urls
                or title_key in seen_urls
                or result.url in pending_urls
                or title_key in pending_titles
                or not _is_html_result(result)
            ):
                continue
            selections.append((query, result))
            pending_urls.add(result.url)
            pending_titles.add(title_key)
            added += 1
            if added >= per_query_limit:
                break

    async def read(query: str, result: SearchResult) -> ResearchSource:
        try:
            page = await scrape_page(result.url, config)
            content = page.text_content or result.snippet
            title = page.title or result.title
        except Exception:
            content = result.snippet
            title = result.title
        return ResearchSource(
            title=title,
            url=result.url,
            content=content[:SOURCE_EXCERPT_CHARS],
            level=level,
            query=query,
        )

    sources = await asyncio.gather(*(read(query, result) for query, result in selections))
    for source in sources:
        seen_urls.add(source.url)
        seen_urls.add(_title_key(source.title))
    return sources


def source_context(sources: list[ResearchSource], *, per_source_chars: int = SOURCE_EXCERPT_CHARS) -> str:
    return "\n\n".join(
        f"=== Level {source.level} source: {source.title} ({source.url}) ===\n"
        f"Found while searching: {source.query}\n"
        f"{source.content[:per_source_chars]}"
        for source in sources
    )


def _parse_queries(content: str, original_query: str) -> list[str]:
    queries: list[str] = []
    seen = {original_query.strip().lower()}
    for line in content.splitlines():
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(candidate) < 8 or len(candidate) > 220:
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(candidate)
        if len(queries) == FOLLOW_UP_QUERY_LIMIT:
            break
    return queries


async def derive_follow_up_queries(
    llm: LLMClient, original_query: str, evidence: list[ResearchSource], level: int,
) -> list[str]:
    """Use the evidence itself to choose the next, narrower research branch."""
    evidence_brief = source_context(evidence[-3:], per_source_chars=900)
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou plan web research. Return exactly two concise search queries, one per line. "
                "Use the evidence to find primary material, independent corroboration, or a "
                "specific unresolved tradeoff. Do not answer the original question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n"
                f"This is research level {level}; choose the next level's queries from this evidence:\n\n"
                f"{evidence_brief}"
            ),
        },
    ]
    try:
        # Query planning is helpful but must never hold up the whole research
        # run. Smaller local models can spend a long time in reasoning mode
        # for this tiny task, so fall back promptly to evidence-derived
        # searches instead.
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        content = response["choices"][0]["message"].get("content", "")
        queries = _parse_queries(content, original_query)
    except Exception:
        queries = []

    # A reliable fallback keeps Extra mode useful with smaller models that
    # cannot follow the query-planning format.
    if len(queries) < FOLLOW_UP_QUERY_LIMIT:
        anchor = next((source.title for source in reversed(evidence) if source.title), original_query)
        anchor = anchor[:120]
        fallbacks = [
            f"{anchor} official documentation technical details",
            f"{anchor} independent comparison limitations benchmarks",
        ]
        known = {query.lower() for query in queries} | {original_query.lower()}
        queries.extend(query for query in fallbacks if query.lower() not in known)
    return queries[:FOLLOW_UP_QUERY_LIMIT]
