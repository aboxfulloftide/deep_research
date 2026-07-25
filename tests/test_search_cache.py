import asyncio

import pytest

import deep_research.tools.search as search_module
from deep_research.config import Config, DBConfig, SearchCacheConfig
from deep_research.models import SearchResult
from deep_research.tools.search_cache import (
    get_cached_results,
    normalize_cache_key,
    store_cached_results,
)


def _config(tmp_path, **overrides) -> Config:
    return Config(db=DBConfig(path=str(tmp_path / "research.db")), **overrides)


def _result(title: str) -> SearchResult:
    return SearchResult(title=title, url=f"https://example.test/{title}", snippet="")


def test_normalize_cache_key_ignores_case_and_whitespace():
    a = normalize_cache_key("Python  Packaging", capability=None, include_alternate_query_engines=False, freshness="default")
    b = normalize_cache_key("python packaging", capability=None, include_alternate_query_engines=False, freshness="default")
    assert a == b


def test_normalize_cache_key_differs_by_capability_and_freshness():
    base = normalize_cache_key("query", capability=None, include_alternate_query_engines=False, freshness="default")
    grounding = normalize_cache_key("query", capability="grounding", include_alternate_query_engines=False, freshness="default")
    stable = normalize_cache_key("query", capability=None, include_alternate_query_engines=False, freshness="stable")
    assert len({base, grounding, stable}) == 3


async def test_get_cached_results_is_none_for_a_genuine_miss(tmp_path):
    config = _config(tmp_path)
    assert await get_cached_results(config, "no-such-key") is None


async def test_store_then_get_round_trips_within_ttl(tmp_path):
    config = _config(tmp_path)
    key = normalize_cache_key("query", capability=None, include_alternate_query_engines=False, freshness="default")
    results = [_result("First"), _result("Second")]

    await store_cached_results(config, key, results, "default")
    cached = await get_cached_results(config, key)

    assert cached == results


async def test_get_cached_results_is_none_once_expired(tmp_path):
    import aiosqlite

    from deep_research.tools.search_cache import cache_db_path

    config = _config(tmp_path)
    key = normalize_cache_key("query", capability=None, include_alternate_query_engines=False, freshness="default")
    await store_cached_results(config, key, [_result("First")], "default")

    async with aiosqlite.connect(cache_db_path(config)) as db:
        await db.execute(
            "UPDATE search_cache SET expires_at = ? WHERE cache_key = ?",
            ("2000-01-01T00:00:00+00:00", key),
        )
        await db.commit()

    assert await get_cached_results(config, key) is None


async def test_store_cached_results_upserts_an_existing_key(tmp_path):
    config = _config(tmp_path)
    key = normalize_cache_key("query", capability=None, include_alternate_query_engines=False, freshness="default")

    await store_cached_results(config, key, [_result("Old")], "default")
    await store_cached_results(config, key, [_result("New")], "default")

    cached = await get_cached_results(config, key)
    assert cached == [_result("New")]


class _EmptySearxngResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [], "unresponsive_engines": []}


class _FakeClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, url, params=None, headers=None):
        return _EmptySearxngResponse()


async def _patch_thin_search(monkeypatch, serper_fn):
    async def no_results(*args, **kwargs):
        return []

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(search_module.httpx, "AsyncClient", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(search_module, "_throttle_searxng", noop)
    monkeypatch.setattr(search_module, "_log_searxng_engines", noop)
    monkeypatch.setattr(search_module, "log_search_call", noop)
    monkeypatch.setattr(search_module, "_wikipedia_api_search", no_results)
    monkeypatch.setattr(search_module, "_wikidata_api_search", no_results)
    monkeypatch.setattr(search_module, "_serper_api_search", serper_fn)


async def test_web_search_serves_the_second_identical_call_from_cache(monkeypatch, tmp_path):
    calls = {"serper": 0}

    async def fake_serper(*args, **kwargs):
        calls["serper"] += 1
        return [_result("Serper result")]

    await _patch_thin_search(monkeypatch, fake_serper)
    config = _config(tmp_path, serper={"api_key": "serper-key"})

    first = await search_module.web_search("thin query", config)
    second = await search_module.web_search("thin query", config)

    assert calls["serper"] == 1
    assert first == second


async def test_web_search_cache_disabled_calls_the_provider_every_time(monkeypatch, tmp_path):
    calls = {"serper": 0}

    async def fake_serper(*args, **kwargs):
        calls["serper"] += 1
        return [_result("Serper result")]

    await _patch_thin_search(monkeypatch, fake_serper)
    config = _config(tmp_path, serper={"api_key": "serper-key"}, search_cache=SearchCacheConfig(enabled=False))

    await search_module.web_search("thin query", config)
    await search_module.web_search("thin query", config)

    assert calls["serper"] == 2


async def test_concurrent_identical_searches_are_coalesced_into_one_computation(monkeypatch, tmp_path):
    calls = {"serper": 0}

    async def slow_fake_serper(*args, **kwargs):
        calls["serper"] += 1
        await asyncio.sleep(0.05)
        return [_result("Serper result")]

    await _patch_thin_search(monkeypatch, slow_fake_serper)
    config = _config(tmp_path, serper={"api_key": "serper-key"})

    first, second = await asyncio.gather(
        search_module.web_search("concurrent thin query", config),
        search_module.web_search("concurrent thin query", config),
    )

    assert calls["serper"] == 1
    assert first == second
