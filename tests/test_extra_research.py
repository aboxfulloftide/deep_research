import pytest

from deep_research.config import Config
from deep_research.models import ScrapedPage, SearchResult
from deep_research.tools import extra_research as extra
from web import app


class _FakeLLM:
    async def chat(self, messages):
        return {"choices": [{"message": {"content": "primary source comparison\nindependent benchmark analysis"}}]}


@pytest.mark.asyncio
async def test_collect_sources_reads_unique_sources_and_keeps_context_bounded(monkeypatch):
    async def fake_search(query, config):
        return [
            SearchResult(title=f"{query} first", url="https://example.test/one", snippet="first"),
            SearchResult(title=f"{query} second", url="https://example.test/two", snippet="second"),
        ]

    async def fake_scrape(url, config):
        return ScrapedPage(url=url, title=f"page {url}", text_content="x" * 4_000)

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)
    seen = set()

    first_level = await extra.collect_sources(["initial question"], Config(), 1, seen)
    repeated = await extra.collect_sources(["follow-up"], Config(), 2, seen)

    assert len(first_level) == 2
    assert repeated == []
    assert all(len(source.content) == extra.SOURCE_EXCERPT_CHARS for source in first_level)
    assert "Level 1 source" in extra.source_context(first_level)


@pytest.mark.asyncio
async def test_collect_sources_skips_syndicated_title_copies(monkeypatch):
    async def fake_search(query, config):
        return [
            SearchResult(title="One article", url="https://first.test/article", snippet="first"),
            SearchResult(title="One article", url="https://copy.test/article", snippet="copy"),
            SearchResult(title="Independent article", url="https://second.test/article", snippet="second"),
        ]

    async def fake_scrape(url, config):
        return ScrapedPage(url=url, title="", text_content="source text")

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)

    sources = await extra.collect_sources(["question"], Config(), 1, set())

    assert [source.url for source in sources] == [
        "https://first.test/article",
        "https://second.test/article",
    ]


@pytest.mark.asyncio
async def test_gap_closing_level_can_cap_a_single_query_to_one_source(monkeypatch):
    async def fake_search(query, config):
        return [
            SearchResult(title="First", url="https://example.test/one", snippet="first"),
            SearchResult(title="Second", url="https://example.test/two", snippet="second"),
        ]

    async def fake_scrape(url, config):
        return ScrapedPage(url=url, title=url, text_content="source text")

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)
    sources = await extra.collect_sources(
        ["one gap-closing query"], Config(), 4, set(), sources_per_query=1,
    )

    assert len(sources) == 1
    assert sources[0].full_content == "source text"


@pytest.mark.asyncio
async def test_extra_research_runs_four_levels_with_source_briefs_and_fact_check(monkeypatch):
    calls = []

    async def fake_collect(queries, config, level, seen_urls, **kwargs):
        calls.append((level, queries))
        return [extra.ResearchSource("Source", f"https://example.test/{level}", "evidence", level, queries[0])]

    async def fake_follow_ups(llm, query, sources, level):
        return [f"level {level} first", f"level {level} second"]

    async def fake_starting_queries(llm, query):
        return ["first factual branch", "second factual branch"]

    monkeypatch.setattr(extra, "collect_sources", fake_collect)
    monkeypatch.setattr(extra, "derive_follow_up_queries", fake_follow_ups)
    monkeypatch.setattr(extra, "derive_starting_queries", fake_starting_queries)
    monkeypatch.setattr(extra, "derive_gap_closing_query", lambda *_: __import__("asyncio").sleep(0, result=["gap-closing source"]))
    events = [event async for event in app._extra_research_answer(_FakeLLM(), "question", Config())]

    assert [level for level, _ in calls] == [1, 2, 3, 4]
    assert calls[0][1] == ["question", "first factual branch", "second factual branch"]
    assert calls[-1][1] == ["gap-closing source"]
    assert len([event for event in events if event["event"] == "status"]) == 11
    assert events[-1] == {
        "event": "answer",
        "data": (
            "primary source comparison\nindependent benchmark analysis\n\n"
            "### Sources consulted\n"
            "- [Source](https://example.test/1)\n"
            "- [Source](https://example.test/2)\n"
            "- [Source](https://example.test/3)\n"
            "- [Source](https://example.test/4)"
        ),
    }


@pytest.mark.asyncio
async def test_follow_up_query_planning_falls_back_to_evidence_title():
    class FailingLLM:
        async def chat(self, messages):
            raise RuntimeError("model unavailable")

    queries = await extra.derive_follow_up_queries(
        FailingLLM(),
        "original question",
        [extra.ResearchSource("Qwen coding guide", "https://example.test", "evidence", 1, "original question")],
        1,
    )

    assert queries == [
        "Qwen coding guide official documentation technical details",
        "Qwen coding guide independent comparison limitations benchmarks",
    ]


@pytest.mark.asyncio
async def test_starting_query_planning_keeps_original_out_of_derived_queries():
    class PlanningLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "original question\nprimary data source\nindependent comparison"}}]}

    queries = await extra.derive_starting_queries(PlanningLLM(), "original question")

    assert queries == ["primary data source", "independent comparison"]
