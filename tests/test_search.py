import httpx
import pytest

import deep_research.tools.search as search_module
from deep_research.config import BraveConfig, Config
from deep_research.models import SearchResult
from deep_research.tools.search import _brave_search_layered, _rank_results


def _result(title: str, snippet: str = "") -> SearchResult:
    return SearchResult(title=title, url=f"https://example.test/{title}", snippet=snippet)


def test_rank_results_prefers_results_about_the_question_over_stale_results():
    query = "Did Donald Trump say racists were very fine people?"
    results = [
        _result("Dissociative Identity Disorder", "A medical condition with multiple identities."),
        _result("Trump's very fine people comments", "Donald Trump discussed Charlottesville."),
    ]

    ranked = _rank_results(results, query)

    assert ranked[0].title == "Trump's very fine people comments"


def test_rank_results_keeps_single_term_queries_searchable():
    result = _result("Qwen language model", "Qwen is a family of large language models.")

    assert _rank_results([result], "Qwen") == [result]


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


@pytest.fixture
def brave_config(monkeypatch):
    """Config with both Brave keys set, logging silenced, and the free
    tier's 429 retry sleep zeroed so tests don't wait it out."""

    async def noop_log(*args, **kwargs):
        pass

    monkeypatch.setattr(search_module, "log_search_call", noop_log)
    monkeypatch.setattr(search_module, "BRAVE_RATE_LIMIT_RETRY_SECONDS", 0)
    return Config(brave=BraveConfig(api_key="free-key", fallback_api_key="paid-key"))


async def test_brave_fallback_key_used_when_primary_errors(brave_config, monkeypatch):
    calls = []

    async def fake_search(query, api_key):
        calls.append(api_key)
        if api_key == "free-key":
            raise _http_error(429)  # quota exhausted: retry fails too
        return [_result("from paid key")]

    monkeypatch.setattr(search_module, "_brave_api_search", fake_search)

    results = await _brave_search_layered("anything", brave_config)

    assert [r.title for r in results] == ["from paid key"]
    # free key tried twice (initial + the one 429 retry), then paid once
    assert calls == ["free-key", "free-key", "paid-key"]


async def test_brave_fallback_not_spent_on_empty_primary_result(brave_config, monkeypatch):
    calls = []

    async def fake_search(query, api_key):
        calls.append(api_key)
        return []

    monkeypatch.setattr(search_module, "_brave_api_search", fake_search)

    results = await _brave_search_layered("anything", brave_config)

    assert results == []
    assert calls == ["free-key"]  # empty is an answer, not a failure


async def test_brave_single_429_retries_free_key_without_paid_spend(brave_config, monkeypatch):
    calls = []

    async def fake_search(query, api_key):
        calls.append(api_key)
        if len(calls) == 1:
            raise _http_error(429)  # per-second rate blip, not quota
        return [_result("from free key retry")]

    monkeypatch.setattr(search_module, "_brave_api_search", fake_search)

    results = await _brave_search_layered("anything", brave_config)

    assert [r.title for r in results] == ["from free key retry"]
    assert calls == ["free-key", "free-key"]


async def test_brave_non_429_error_skips_retry_and_uses_fallback(brave_config, monkeypatch):
    calls = []

    async def fake_search(query, api_key):
        calls.append(api_key)
        if api_key == "free-key":
            raise _http_error(401)  # subscription problem: retrying won't help
        return [_result("from paid key")]

    monkeypatch.setattr(search_module, "_brave_api_search", fake_search)

    results = await _brave_search_layered("anything", brave_config)

    assert [r.title for r in results] == ["from paid key"]
    assert calls == ["free-key", "paid-key"]
