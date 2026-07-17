import pytest

from deep_research.config import Config
from deep_research.models import ScrapedPage, SearchResult
from deep_research.tools import extra_research as extra
from web import app


class _FakeLLM:
    async def chat(self, messages):
        system = messages[0]["content"]
        if "extract evidence for a research claim ledger" in system.lower():
            return {"choices": [{"message": {"content": '[{"statement":"Evidence supports the answer.","quote":"source evidence text","confidence":0.9}]'}}]}
        return {"choices": [{"message": {"content": "primary source comparison\nindependent benchmark analysis"}}]}


@pytest.mark.asyncio
async def test_collect_sources_reads_unique_sources_and_keeps_context_bounded(monkeypatch):
    async def fake_search(query, config):
        return [
            SearchResult(title=f"{query} first", url="https://huggingface.co/one", snippet="first"),
            SearchResult(title=f"{query} second", url="https://github.com/two", snippet="second"),
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
            SearchResult(title="One article", url="https://huggingface.co/first", snippet="first"),
            SearchResult(title="One article", url="https://github.com/copy", snippet="copy"),
            SearchResult(title="Independent article", url="https://arxiv.org/second", snippet="second"),
        ]

    async def fake_scrape(url, config):
        return ScrapedPage(url=url, title="", text_content="source text " * 30)

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)

    sources = await extra.collect_sources(["question"], Config(), 1, set())

    assert len(sources) == 2
    assert "https://arxiv.org/second" in [source.url for source in sources]
    assert sum(source.title == "One article" for source in sources) == 1


@pytest.mark.asyncio
async def test_gap_closing_level_can_cap_a_single_query_to_one_source(monkeypatch):
    async def fake_search(query, config):
        return [
            SearchResult(title="First", url="https://huggingface.co/one", snippet="first"),
            SearchResult(title="Second", url="https://github.com/two", snippet="second"),
        ]

    async def fake_scrape(url, config):
        return ScrapedPage(url=url, title=url, text_content="source text " * 30)

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)
    sources = await extra.collect_sources(
        ["one gap-closing query"], Config(), 4, set(), sources_per_query=1,
    )

    assert len(sources) == 1
    assert sources[0].full_content == "source text " * 30


@pytest.mark.asyncio
async def test_extra_research_runs_four_levels_with_source_briefs_and_fact_check(monkeypatch):
    source = extra.ResearchSource("Source", "https://huggingface.co/Qwen/example", "source evidence text", 1, "core evidence", quality_score=5, source_kind="primary")
    plan = extra.ResearchPlan("question", [], [extra.ResearchFacet("core", "core evidence", "Direct evidence")])
    async def fake_bundle(*args):
        return extra.ResearchBundle(plan, [source], [], extra._coverage_for(plan, [source]))
    monkeypatch.setattr(extra, "collect_research_bundle", fake_bundle)
    events = [event async for event in app._extra_research_answer(_FakeLLM(), "question", Config())]

    assert len([event for event in events if event["event"] == "status"]) == 6
    assert events[-1] == {
        "event": "answer",
        "data": (
            "primary source comparison\nindependent benchmark analysis\n\n"
            "### Sources consulted\n"
                "- [Source](https://huggingface.co/Qwen/example)"
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
async def test_starting_query_planning_uses_the_topic_aware_plan():
    class PlanningLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "original question\nprimary data source\nindependent comparison"}}]}

    queries = await extra.derive_starting_queries(PlanningLLM(), "original question")

    assert queries == ["primary data source", "independent comparison"]


def test_huggingface_quantization_is_not_labeled_primary_and_family_is_deduplicated():
    assert extra.classify_source("https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct")[0] == "primary"
    assert extra.classify_source("https://huggingface.co/QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ")[0] == "technical_reference"
    base = extra._model_family_key(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct · Hugging Face",
        "https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    )
    awq = extra._model_family_key(
        "QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ · Hugging Face",
        "https://huggingface.co/QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ",
    )
    assert base == awq


def test_broken_marketplace_scrape_is_not_usable_evidence():
    assert not extra._usable_scrape("Found 2 products: participant risks " * 30)


@pytest.mark.asyncio
async def test_research_bundle_routes_facets_and_records_fitness(monkeypatch):
    class Planner:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"ambiguities":[],"facets":[{"id":"spec","question":"official requirements for example software","search_query":"example software official requirements","purpose":"constraints","capabilities":["official_documentation","repository"]},{"id":"evidence","question":"independent evidence for example software","search_query":"example software independent evidence","purpose":"corroboration","capabilities":["scholarly"]}]}'}}]}

    calls = []
    async def fake_collect(queries, config, level, seen_urls, **kwargs):
        calls.append((level, queries[0]))
        return [extra.ResearchSource("Example software official requirements", f"https://example.test/{len(calls)}", "official requirements for example software " * 20, level, queries[0], quality_score=5, source_kind="primary")]

    monkeypatch.setattr(extra, "collect_sources", fake_collect)
    bundle = await extra.collect_research_bundle(Planner(), "example question", Config(), extra.ResearchBudget(max_sources=3, max_gap_rounds=0))

    assert len(bundle.sources) == 3
    assert {attempt["adapter"] for attempt in bundle.collection_attempts} == {"official_documentation", "repository", "scholarly"}
    assert all("directness" in assessment for assessment in bundle.assessments)


@pytest.mark.asyncio
async def test_research_plan_rejects_a_facet_that_searches_the_raw_user_question():
    class BadPlanner:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"ambiguities":[],"facets":[{"id":"repeat","question":"direct evidence","search_query":"What local LLM should I use for coding?","purpose":"answer","capabilities":["web"]},{"id":"other","question":"other evidence","search_query":"What local LLM should I use for coding?","purpose":"corroborate","capabilities":["web"]}]}'}}]}

    question = "What local LLM should I use for coding?"
    plan = await extra.plan_research(BadPlanner(), question)

    assert all(facet.search_query.lower() != question.lower() for facet in plan.facets)
    assert all(facet.search_query for facet in plan.facets)


@pytest.mark.asyncio
async def test_research_plan_repairs_malformed_json_with_simple_lines():
    class LinePlanner:
        calls = 0
        async def chat(self, messages):
            self.calls += 1
            content = "not json" if self.calls == 1 else (
                "specs | official capabilities and limits | product official specifications limits | primary,official_documentation\n"
                "evidence | independent evaluation | independent benchmark methodology results | scholarly,repository\n"
                "constraints | practical constraints | deployment requirements tradeoffs | web"
            )
            return {"choices": [{"message": {"content": content}}]}

    plan = await extra.plan_research(LinePlanner(), "Which option is best for a constrained deployment?")

    assert [facet.id for facet in plan.facets] == ["specs", "evidence", "constraints"]
    assert plan.facets[0].search_query == "product official specifications limits"


@pytest.mark.asyncio
async def test_claim_ledger_rejects_a_claim_without_a_verbatim_quote():
    source = extra.ResearchSource("Source", "https://huggingface.co/example", "the supported fact is here", 1, "query", quality_score=5)
    claims = extra._parse_ledger(
        '[{"statement":"Unsupported statement", "quote":"not present", "confidence":0.9}]', source,
    )
    assert claims == []


@pytest.mark.asyncio
async def test_claim_ledger_keeps_source_attributed_verbatim_evidence():
    source = extra.ResearchSource("Source", "https://huggingface.co/example", "The supported fact is here.", 1, "query", quality_score=5)
    claims = extra._parse_ledger(
        '[{"statement":"A supported fact exists.", "quote":"supported fact is here", "confidence":0.9}]', source,
    )
    assert claims[0].source_url == "https://huggingface.co/example"
    assert "[Source](https://huggingface.co/example)" in extra.claim_ledger_context(claims)
