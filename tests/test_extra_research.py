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
    async def fake_search(query, config, **kwargs):
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
    async def fake_search(query, config, **kwargs):
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
    async def fake_search(query, config, **kwargs):
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
async def test_collect_sources_backfills_to_the_next_candidate_when_the_top_result_fails_to_fetch(monkeypatch):
    async def fake_search(query, config, **kwargs):
        return [
            SearchResult(title="Top paper", url="https://arxiv.org/abs/1111.11111", snippet="top"),
            SearchResult(title="Second choice", url="https://github.com/two", snippet="second"),
        ]

    async def fake_scrape(url, config):
        if "arxiv.org" in url:
            raise RuntimeError("connection reset")
        return ScrapedPage(url=url, title="Second choice", text_content="source text " * 30)

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)

    outcomes = []
    sources = await extra.collect_sources(
        ["question"], Config(), 1, set(), sources_per_query=1, outcomes=outcomes,
    )

    assert len(sources) == 1
    assert sources[0].url == "https://github.com/two"
    decisions = {outcome["url"]: outcome["decision"] for outcome in outcomes}
    assert decisions["https://arxiv.org/abs/1111.11111"] == "fetch_failed"
    assert decisions["https://github.com/two"] == "accepted"


@pytest.mark.asyncio
async def test_collect_sources_never_uses_snippet_as_content(monkeypatch):
    long_snippet = "This snippet is deliberately long enough to pass the old usable-scrape threshold. " * 5

    async def fake_search(query, config, **kwargs):
        return [SearchResult(title="Broken page", url="https://github.com/broken", snippet=long_snippet)]

    async def fake_scrape(url, config):
        raise RuntimeError("500 server error")

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)

    outcomes = []
    sources = await extra.collect_sources(["question"], Config(), 1, set(), outcomes=outcomes)

    assert sources == []
    assert outcomes[0]["decision"] == "fetch_failed"


@pytest.mark.asyncio
async def test_collect_sources_records_a_terminal_outcome_for_every_considered_candidate(monkeypatch):
    async def fake_search(query, config, **kwargs):
        return [
            SearchResult(title="Community post", url="https://reddit.com/one", snippet="low quality"),
            SearchResult(title="Duplicate", url="https://github.com/dup", snippet="dup"),
            SearchResult(title="Duplicate", url="https://github.com/dup-again", snippet="dup again"),
            SearchResult(title="Too short", url="https://github.com/short", snippet="short"),
        ]

    async def fake_scrape(url, config):
        if "short" in url:
            return ScrapedPage(url=url, title="Too short", text_content="short")
        return ScrapedPage(url=url, title="Duplicate", text_content="source text " * 30)

    monkeypatch.setattr(extra, "web_search", fake_search)
    monkeypatch.setattr(extra, "scrape_page", fake_scrape)

    outcomes = []
    sources = await extra.collect_sources(
        ["question"], Config(), 1, set(), sources_per_query=4, outcomes=outcomes,
    )

    assert len(outcomes) == 4
    decisions = {outcome["url"]: outcome["decision"] for outcome in outcomes}
    assert decisions["https://reddit.com/one"] == "rejected_low_quality"
    assert decisions["https://github.com/dup-again"] == "rejected_duplicate"
    assert decisions["https://github.com/short"] == "rejected_unusable_scrape"
    assert decisions["https://github.com/dup"] == "accepted"
    assert len(sources) == 1


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
        "Qwen coding guide independent comparison limitations evidence",
    ]


@pytest.mark.asyncio
async def test_starting_query_planning_uses_the_topic_aware_plan():
    class PlanningLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "original question\nprimary data source\nindependent comparison"}}]}

    queries = await extra.derive_starting_queries(PlanningLLM(), "original question")

    assert queries == ["primary data source", "independent comparison"]


def test_classify_source_uses_domain_neutral_authority_signals():
    assert extra.classify_source("https://www.irs.gov/publications/p17") == ("primary", 5)
    assert extra.classify_source("https://arxiv.org/abs/2401.00000") == ("paper", 5)
    assert extra.classify_source("https://docs.stripe.com/api") == ("technical_reference", 4)
    assert extra.classify_source("https://github.com/example/repo") == ("technical_reference", 4)
    assert extra.classify_source("https://github.com/example/repo/discussions/1") == ("community", 1)
    assert extra.classify_source("https://reddit.com/r/test") == ("community", 1)
    assert extra.classify_source("https://randomblog.example.com/post") == ("secondary", 2)


def test_has_authoritative_source_accepts_official_documentation_not_just_papers():
    docs_only = [extra.ResearchSource("Docs", "https://docs.example.com/x", "x" * 300, 1, "q", quality_score=4, source_kind="technical_reference")]
    assert extra.has_authoritative_source(docs_only) is True

    secondary_only = [extra.ResearchSource("Blog", "https://example.com/x", "x" * 300, 1, "q", quality_score=2, source_kind="secondary")]
    assert extra.has_authoritative_source(secondary_only) is False


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
