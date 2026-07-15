"""Bounded multi-level web research for the interactive Extra mode.

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
INITIAL_QUERY_LIMIT = 2
GAP_CLOSING_QUERY_LIMIT = 1
SOURCE_ANALYSIS_CHARS = 2_500


@dataclass
class ResearchSource:
    title: str
    url: str
    content: str
    level: int
    query: str
    full_content: str = ""


def _is_html_result(result: SearchResult) -> bool:
    return not result.url.lower().split("?", 1)[0].endswith(".pdf")


def _title_key(title: str) -> str:
    """Identify syndicated copies that have different URLs but the same story."""
    article_title = re.split(r"\s+[|–—-]\s+", title, maxsplit=1)[0]
    return "title:" + re.sub(r"\W+", " ", article_title.lower()).strip()


async def collect_sources(
    queries: list[str], config: Config, level: int, seen_urls: set[str], *, sources_per_query: int | None = None,
) -> list[ResearchSource]:
    """Search each query and read new HTML sources from it.

    The first level combines the original wording with planned subquestions.
    Follow-up levels use focused evidence branches; the final level uses one
    gap-closing query. This keeps the whole run bounded while diversifying
    evidence beyond the user's exact wording.
    """
    selections: list[tuple[str, SearchResult]] = []
    pending_urls: set[str] = set()
    pending_titles: set[str] = set()
    per_query_limit = sources_per_query if sources_per_query is not None else (2 if len(queries) == 1 else SOURCES_PER_QUERY)
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
            full_content=content,
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


def _parse_queries(content: str, original_query: str, limit: int = FOLLOW_UP_QUERY_LIMIT) -> list[str]:
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
        if len(queries) == limit:
            break
    return queries


async def derive_starting_queries(llm: LLMClient, original_query: str) -> list[str]:
    """Plan two complementary searches before querying the web.

    The original wording remains a search query. These additions turn a broad
    question into smaller factual branches instead of betting the first round
    on whichever exact words the user happened to type.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou plan a concise web-research search set. Return exactly two search queries, one per line. "
                "Break the user's question into complementary factual branches: important constraints, comparisons, "
                "primary data, or decision criteria. Keep each query independently searchable. Do not answer the question."
            ),
        },
        {"role": "user", "content": f"Research question: {original_query}"},
    ]
    try:
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        content = response["choices"][0]["message"].get("content", "")
        queries = _parse_queries(content, original_query, INITIAL_QUERY_LIMIT)
    except Exception:
        queries = []

    if len(queries) < INITIAL_QUERY_LIMIT:
        fallbacks = [
            f"{original_query} primary sources data",
            f"{original_query} comparison tradeoffs limitations",
        ]
        known = {original_query.lower()} | {query.lower() for query in queries}
        queries.extend(query for query in fallbacks if query.lower() not in known)
    return queries[:INITIAL_QUERY_LIMIT]


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
        queries = _parse_queries(content, original_query, FOLLOW_UP_QUERY_LIMIT)
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


async def derive_gap_closing_query(
    llm: LLMClient, original_query: str, evidence: list[ResearchSource],
) -> list[str]:
    """Choose one final source that can corroborate the biggest remaining gap."""
    evidence_brief = source_context(evidence[-4:], per_source_chars=700)
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are closing a web-research evidence gap. Return exactly one concise search query. "
                "Prefer a primary source, authoritative dataset, or independent corroboration for the most important "
                "unresolved factual claim. Do not answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Original question: {original_query}\n\nEvidence so far:\n{evidence_brief}",
        },
    ]
    try:
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        content = response["choices"][0]["message"].get("content", "")
        queries = _parse_queries(content, original_query, GAP_CLOSING_QUERY_LIMIT)
    except Exception:
        queries = []

    if not queries:
        anchor = next((source.title for source in reversed(evidence) if source.title), original_query)[:120]
        queries = [f"{anchor} primary source verification"]
    return queries[:GAP_CLOSING_QUERY_LIMIT]


async def analyze_sources_separately(
    llm: LLMClient, original_query: str, sources: list[ResearchSource],
) -> list[str]:
    """Create compact, source-attributed briefs before cross-source synthesis."""
    semaphore = asyncio.Semaphore(2)

    async def analyze(source: ResearchSource) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think\nAnalyze one research source for a later synthesis. Extract only facts relevant to the "
                    "original question, distinguish claims from evidence, note limitations or uncertainty, and do not "
                    "invent information. Keep it under 180 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original_query}\n"
                    f"Source: {source.title} ({source.url})\n\n"
                    f"Source text:\n{source.content[:SOURCE_ANALYSIS_CHARS]}"
                ),
            },
        ]
        try:
            async with semaphore:
                response = await llm.chat(messages)
            content = response["choices"][0]["message"].get("content", "").strip()
        except Exception:
            content = ""
        if not content:
            content = source.content[:900]
        return f"=== {source.title} ({source.url}) ===\n{content}"

    return await asyncio.gather(*(analyze(source) for source in sources))


def analysis_context(analyses: list[str]) -> str:
    return "\n\n".join(analyses)
