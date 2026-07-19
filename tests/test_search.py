import httpx
import pytest

import deep_research.tools.search as search_module
from deep_research.config import BraveConfig, Config, SerperConfig, TavilyConfig
from deep_research.models import SearchResult
from deep_research.tools.search import (
    SEARXNG_BASE_ENGINES,
    SEARXNG_RECOVERED_ENGINES,
    _brave_search_layered,
    _rank_results,
    _wikidata_entity_query,
    web_search,
)


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


def test_wikidata_uses_relevant_wikipedia_title_instead_of_claim_sentence():
    query = "The global financial crisis of 2008 took out a lot of hedge funds."
    wikipedia_results = [
        _result("Hedge fund", "investment fund"),
        _result("2007–2008 financial crisis", "global financial crisis and hedge funds"),
    ]

    assert _wikidata_entity_query(query, wikipedia_results) == "2007–2008 financial crisis"


def test_wikidata_prefers_claims_named_subject_over_secondary_wikipedia_detail():
    query = "Masayoshi Son entered the tech space before he graduated from university."
    wikipedia_results = [
        _result("University of California, Berkeley", "university where Masayoshi Son studied"),
    ]

    assert _wikidata_entity_query(query, wikipedia_results) == "Masayoshi Son"


def test_wikidata_prefers_first_named_subject_over_later_named_detail():
    query = "John Klug's strategy became the playbook for every Silicon Valley CEO."

    assert _wikidata_entity_query(query, []) == "John Klug"


def test_wikidata_uses_short_entity_query_and_skips_unmapped_sentence():
    assert _wikidata_entity_query("Masayoshi Son", []) == "Masayoshi Son"
    assert _wikidata_entity_query(
        "This long sentence has no useful matching entity page in the available results.", []
    ) is None


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

    async def quota_available(*args, **kwargs):
        return False

    monkeypatch.setattr(search_module, "log_search_call", noop_log)
    monkeypatch.setattr(
        search_module, "provider_monthly_quota_exhausted", quota_available,
    )
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


async def test_brave_primary_stays_paused_after_monthly_quota_failure(brave_config, monkeypatch):
    calls = []

    async def quota_exhausted(*args, **kwargs):
        return True

    async def fake_search(query, api_key):
        calls.append(api_key)
        return [_result("from paid key")]

    monkeypatch.setattr(
        search_module, "provider_monthly_quota_exhausted", quota_exhausted,
    )
    monkeypatch.setattr(search_module, "_brave_api_search", fake_search)

    results = await _brave_search_layered("anything", brave_config)

    assert [r.title for r in results] == ["from paid key"]
    assert calls == ["paid-key"]


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


async def test_web_search_only_opts_recovered_engines_into_alternate_queries(monkeypatch):
    requested_params = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [], "unresponsive_engines": []}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None, headers=None):
            requested_params.append(params)
            return FakeResponse()

    async def no_results(*args, **kwargs):
        return []

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(search_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(search_module, "_throttle_searxng", noop)
    monkeypatch.setattr(search_module, "_log_searxng_engines", noop)
    monkeypatch.setattr(search_module, "log_search_call", noop)
    monkeypatch.setattr(search_module, "_wikipedia_api_search", no_results)
    monkeypatch.setattr(search_module, "_wikidata_api_search", no_results)

    async def allow_engines(config, providers, **kwargs):
        return set(providers)

    monkeypatch.setattr(
        search_module, "providers_allowed_by_circuit_breaker", allow_engines,
    )

    config = Config()
    await web_search("A literal claim sentence.", config)
    await web_search(
        "concise alternate keywords", config,
        include_alternate_query_engines=True,
    )

    assert requested_params[0]["engines"] == ",".join(SEARXNG_BASE_ENGINES)
    assert requested_params[1]["engines"] == ",".join(
        SEARXNG_BASE_ENGINES + SEARXNG_RECOVERED_ENGINES
    )


async def test_web_search_omits_duckduckgo_while_its_circuit_is_open(monkeypatch):
    requested_params = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [], "unresponsive_engines": []}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None, headers=None):
            requested_params.append(params)
            return FakeResponse()

    async def no_results(*args, **kwargs):
        return []

    async def noop(*args, **kwargs):
        pass

    async def block_duckduckgo(config, providers, **kwargs):
        return set() if providers == ("duckduckgo",) else set(providers)

    monkeypatch.setattr(search_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(search_module, "_throttle_searxng", noop)
    monkeypatch.setattr(search_module, "_log_searxng_engines", noop)
    monkeypatch.setattr(search_module, "log_search_call", noop)
    monkeypatch.setattr(search_module, "_wikipedia_api_search", no_results)
    monkeypatch.setattr(search_module, "_wikidata_api_search", no_results)
    monkeypatch.setattr(
        search_module, "providers_allowed_by_circuit_breaker", block_duckduckgo,
    )

    await web_search("literal claim", Config())

    assert requested_params[0]["engines"] == "bing,mojeek"


async def test_serper_is_primary_and_tavily_remains_thin_results_fallback(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": f"PostgreSQL index documentation {i}",
                     "url": f"https://postgresql.example/{i}",
                     "content": "PostgreSQL index documentation", "engine": "bing"}
                    for i in range(3)
                ],
                "unresponsive_engines": [],
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None, headers=None):
            return FakeResponse()

    calls = {"serper": 0, "tavily": 0}

    async def fake_serper(*args, **kwargs):
        calls["serper"] += 1
        return [_result("PostgreSQL primary result", "index documentation")]

    async def fake_tavily(*args, **kwargs):
        calls["tavily"] += 1
        return [_result("PostgreSQL fallback result", "index documentation")]

    async def no_results(*args, **kwargs):
        return []

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(search_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(search_module, "_throttle_searxng", noop)
    monkeypatch.setattr(search_module, "_log_searxng_engines", noop)
    monkeypatch.setattr(search_module, "log_search_call", noop)
    monkeypatch.setattr(search_module, "_wikipedia_api_search", no_results)
    monkeypatch.setattr(search_module, "_wikidata_api_search", no_results)
    monkeypatch.setattr(search_module, "_serper_api_search", fake_serper)
    monkeypatch.setattr(search_module, "_tavily_api_search", fake_tavily)

    config = Config(
        serper=SerperConfig(api_key="serper-key"),
        tavily=TavilyConfig(api_key="tavily-key"),
    )
    await web_search("PostgreSQL index documentation", config)

    assert calls == {"serper": 1, "tavily": 0}
