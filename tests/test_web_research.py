import pytest

from deep_research.config import Config
from deep_research.models import ScrapedPage, SearchResult
from deep_research.tools import extra_research
from web import app


class _FakeLLM:
    def __init__(self):
        self.messages = []

    async def chat(self, messages):
        self.messages = messages
        return {"choices": [{"message": {"content": "answer"}}]}


@pytest.mark.asyncio
async def test_text_research_adds_full_source_context_and_quote_guidance(monkeypatch):
    result = SearchResult(
        title="Fact check",
        url="https://example.test/fact-check",
        snippet="A short result snippet.",
    )

    async def fake_search(query, config):
        return [result]

    async def fake_scrape(url, config):
        return ScrapedPage(
            url=url,
            title="Fact check full article",
            text_content="The complete surrounding quotation and its qualification.",
        )

    monkeypatch.setattr(app, "web_search", fake_search)
    monkeypatch.setattr(app, "scrape_page", fake_scrape)
    llm = _FakeLLM()

    answer = await app._text_mode_answer(llm, "Was this quote accurate?", "", Config())

    assert answer == "answer"
    prompt = llm.messages[1]["content"]
    assert "complete surrounding quotation" in prompt
    assert "disputed quotes or claims" in prompt
    assert "Claim, Rating, or Context" in prompt
    assert "Markdown links" in prompt


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    async def chat(self, messages):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": "answer"}}]}


@pytest.mark.asyncio
async def test_extra_research_tells_synthesis_about_uncovered_facets(monkeypatch):
    source = extra_research.ResearchSource(
        "Direct Source", "https://example.test/direct", "content", 1, "q",
        full_content="content", quality_score=5, source_kind="paper",
    )
    coverage = {
        "facets": [{"id": "core", "purpose": "Direct evidence for the question."},
                   {"id": "constraints", "purpose": "Definitions and limits relevant to the question."}],
        "missing_facet_ids": ["constraints"],
    }
    bundle = extra_research.ResearchBundle(
        plan=extra_research.ResearchPlan("q", [], []), sources=[source],
        collection_attempts=[], coverage=coverage,
    )

    async def fake_bundle(*args, **kwargs):
        return bundle

    async def fake_analyze(*args, **kwargs):
        return ["=== Direct Source (https://example.test/direct) ===\nbrief"]

    async def fake_ledger(*args, **kwargs):
        return [extra_research.EvidenceClaim(
            statement="A fact.", quote="content", source_title="Direct Source",
            source_url="https://example.test/direct", source_kind="paper", confidence=0.9,
        )]

    monkeypatch.setattr(extra_research, "collect_research_bundle", fake_bundle)
    monkeypatch.setattr(extra_research, "analyze_sources_separately", fake_analyze)
    monkeypatch.setattr(extra_research, "build_claim_ledger", fake_ledger)

    llm = _RecordingLLM()
    events = [event async for event in app._extra_research_answer(llm, "example question", Config())]

    synthesis_prompt = llm.calls[0][1]["content"]
    assert "Definitions and limits relevant to the question." in synthesis_prompt
    assert "could NOT find direct, authoritative sources" in synthesis_prompt
    assert events[-1]["event"] == "answer"
