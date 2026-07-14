import pytest

from deep_research.config import Config
from deep_research.models import ScrapedPage, SearchResult
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
    assert "Markdown links" in prompt
