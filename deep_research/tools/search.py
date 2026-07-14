import asyncio
import re
import time
import urllib.parse

import httpx

from deep_research.config import Config
from deep_research.models import SearchResult
from deep_research.tools.search_usage import log_search_call, timer

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_API_URL = "https://api.tavily.com/search"
SERPER_API_URL = "https://google.serper.dev/search"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "did", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "say", "the",
    "to", "was", "were", "what", "when", "where", "who", "why", "with",
}
# Below this many combined duckduckgo+brave results, spend a metered Tavily
# query to fill the gap -- keeps Tavily's smaller monthly budget for when
# duckduckgo+brave actually come up short, instead of spending it every query.
MIN_RESULTS_BEFORE_TAVILY_FALLBACK = 3
# Serper's free tier is a one-time 2500-query trial, not a recurring monthly
# allowance like brave/tavily -- only spend it if duckduckgo+brave+tavily
# combined still came up this thin, i.e. genuinely last resort.
MIN_RESULTS_BEFORE_SERPER_FALLBACK = 3

_searxng_throttle_lock = asyncio.Lock()
_searxng_last_call_at: float | None = None


async def _throttle_searxng(min_interval_seconds: float) -> None:
    """Serializes SearXNG calls and enforces a minimum gap between them,
    even across concurrently-running verification tasks -- two claims
    verifying at once (verification_concurrency) would otherwise happily
    fire simultaneous SearXNG requests. Process-local (module-level state),
    so this paces one process's own traffic; it doesn't coordinate across
    separate processes (e.g. a concurrent CLI run and the web server)."""
    global _searxng_last_call_at
    async with _searxng_throttle_lock:
        now = time.monotonic()
        if _searxng_last_call_at is not None:
            wait = min_interval_seconds - (now - _searxng_last_call_at)
            if wait > 0:
                await asyncio.sleep(wait)
        _searxng_last_call_at = time.monotonic()


async def _brave_api_search(query: str, api_key: str) -> list[SearchResult]:
    """Direct call to Brave's official Search API. SearXNG's own brave engine
    is disabled in searxng/settings.yml (it got rate limited under sustained
    query volume, same as google cse/startpage) -- this is now the only
    source of Brave results, called unconditionally rather than only as a
    fallback."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            BRAVE_API_URL,
            params={"q": query, "count": 10},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:10]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=_HTML_TAG_RE.sub("", item.get("description", "")),
        ))
    return results


async def _tavily_api_search(query: str, api_key: str) -> list[SearchResult]:
    """Direct call to the Tavily Search API -- general fallback used when
    SearXNG's combined results are thin, regardless of which engine(s) are
    responsible. search_depth="advanced" is required: the default "basic"
    depth returns results wrapped in an opaque, non-resolving "/goto?url=..."
    redirect instead of the real destination URL."""
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            TAVILY_API_URL,
            json={"api_key": api_key, "query": query, "search_depth": "advanced", "max_results": 10},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:10]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        ))
    return results


async def _serper_api_search(query: str, api_key: str) -> list[SearchResult]:
    """Direct call to Serper's Search API (a Google-results proxy) -- last-
    resort fallback, see MIN_RESULTS_BEFORE_SERPER_FALLBACK."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            SERPER_API_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("organic", [])[:10]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        ))
    return results


def _wikipedia_user_agent(contact: str) -> str:
    """Wikimedia's edge WAF 403s requests that don't follow their
    identification policy (real contact info, not a generic client string)
    -- see foundation.wikimedia.org/wiki/Policy:User-Agent_policy. Verified
    empirically: identical request, only the User-Agent changed, 403 -> 200."""
    contact_part = contact or "no contact configured"
    return f"deep-research-kb-bot/1.0 ({contact_part}) httpx"


async def _wikipedia_api_search(query: str, contact: str = "") -> list[SearchResult]:
    """Direct call to Wikimedia's own public REST search API -- no key, no
    auth, documented and intended for exactly this kind of programmatic
    access (unlike DuckDuckGo/Mojeek/Google's anti-bot walls). Distinct from
    SearXNG's "wikipedia" engine (which shares SearXNG's own rate-limit
    fate) -- called directly and unconditionally alongside duckduckgo/brave,
    logged as "wikipedia_api" to keep the two paths visibly separate."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            WIKIPEDIA_API_URL,
            params={"q": query, "limit": 10},
            headers={"User-Agent": _wikipedia_user_agent(contact)},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for page in data.get("pages", [])[:10]:
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page.get('key', ''))}"
        snippet = _HTML_TAG_RE.sub("", page.get("excerpt", "")) or page.get("description", "")
        results.append(SearchResult(title=page.get("title", ""), url=url, snippet=snippet))
    return results


async def _wikidata_api_search(query: str, contact: str = "") -> list[SearchResult]:
    """Direct call to Wikidata's own entity-search API (wbsearchentities) --
    replaces SearXNG's "wikidata" engine, which failed on 46 of 46 logged
    calls (timeouts + "too many requests", each timeout stalling the whole
    SearXNG response) and is now disabled in searxng/settings.yml. Same
    no-key Wikimedia infrastructure and User-Agent policy as
    _wikipedia_api_search; logged as "wikidata_api".

    wbsearchentities matches entity labels/aliases (returning a proper
    display label + description, unlike Wikidata's REST full-text search,
    whose result titles are opaque Q-ids) -- so entity-name queries get
    clean hits and sentence-shaped queries usually get zero results. That's
    expected and fine: Wikidata is an entity database, and this is a free
    add-on alongside the engines that do full-text search."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbsearchentities", "search": query,
                "language": "en", "format": "json", "limit": 10,
            },
            headers={"User-Agent": _wikipedia_user_agent(contact)},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("search", [])[:10]:
        entity_id = item.get("id", "")
        if not entity_id:
            continue
        results.append(SearchResult(
            title=item.get("label", entity_id),
            url=f"https://www.wikidata.org/wiki/{urllib.parse.quote(entity_id)}",
            snippet=item.get("description") or "",
        ))
    return results


async def _log_searxng_engines(config: Config, data: dict, elapsed_ms: int, query: str) -> None:
    """SearXNG fans one call out to several underlying engines (duckduckgo,
    bing, mojeek, wikipedia, ...) and merges their results -- log each engine
    that actually contributed or failed separately, rather than lumping
    everything under "duckduckgo", so /api/search-usage reflects what's
    really answering queries."""
    contributed: dict[str, int] = {}
    for item in data.get("results", []):
        for engine in item.get("engines") or [item.get("engine")]:
            if engine:
                contributed[engine] = contributed.get(engine, 0) + 1
    for engine, reason in data.get("unresponsive_engines", []):
        if engine not in contributed:
            await log_search_call(
                config, engine, "scrape", "error",
                error_message=reason, elapsed_ms=elapsed_ms, query=query,
            )
    for engine, count in contributed.items():
        await log_search_call(
            config, engine, "scrape", "ok",
            result_count=count, elapsed_ms=elapsed_ms, query=query,
        )
    if not contributed and not data.get("unresponsive_engines"):
        await log_search_call(config, "searxng", "scrape", "empty", result_count=0, elapsed_ms=elapsed_ms, query=query)


def _merge(results: list[SearchResult], new: list[SearchResult]) -> list[SearchResult]:
    seen_urls = {r.url for r in results}
    for r in new:
        if r.url not in seen_urls:
            results.append(r)
            seen_urls.add(r.url)
    return results


def _query_terms(query: str) -> set[str]:
    """Meaningful query terms used to prevent one bad provider from filling
    the entire context window with unrelated results."""
    return {
        token for token in _QUERY_TOKEN_RE.findall(query.lower())
        if len(token) > 2 and token not in _QUERY_STOP_WORDS
    }


def _relevance_score(result: SearchResult, terms: set[str]) -> int:
    """A small, transparent lexical guardrail before an LLM sees results.

    Search APIs occasionally return a stale/cross-query response. Ranking by
    the question's distinctive words is enough to keep those pages from
    crowding out results returned by the other providers. This deliberately
    does not attempt to decide truth or source quality.
    """
    if not terms:
        return 0
    title_terms = set(_QUERY_TOKEN_RE.findall(result.title.lower()))
    content_terms = set(_QUERY_TOKEN_RE.findall(result.snippet.lower()))
    return 2 * len(terms & title_terms) + len(terms & content_terms)


def _is_relevant(result: SearchResult, terms: set[str]) -> bool:
    # A multi-word question needs at least two signals. One-term questions
    # (names, identifiers, product codes) should still be allowed through.
    minimum = 1 if len(terms) <= 1 else 2
    return _relevance_score(result, terms) >= minimum


def _rank_results(results: list[SearchResult], query: str, limit: int = 10) -> list[SearchResult]:
    terms = _query_terms(query)
    return sorted(results, key=lambda result: _relevance_score(result, terms), reverse=True)[:limit]


async def web_search(query: str, config: Config) -> list[SearchResult]:
    """duckduckgo (via SearXNG) + Brave + Tavily, both official APIs. brave/
    google cse/startpage are disabled in searxng/settings.yml -- they got
    rate limited/CAPTCHA'd under sustained query volume and have no documented
    quota to plan around, unlike Brave's and Tavily's metered APIs.

    Every provider call is logged to search_usage.py's SQLite log (best
    effort, never raises) so /api/search-usage can answer "how many searches
    have we used" and "is duckduckgo/brave/tavily currently responding"
    without grepping run logs or curling each provider by hand."""
    url = f"{config.searxng.url.rstrip('/')}/search"
    params = {
        "q": query,
        "format": "json",
    }
    await _throttle_searxng(config.searxng.min_interval_seconds)
    t = timer()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        await log_search_call(
            config, "searxng", "scrape", "error",
            error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
        )
        # SearXNG is one provider, not a single point of failure. Continue to
        # the direct providers below when it is unavailable.
        data = {"results": []}

    await _log_searxng_engines(config, data, t.elapsed_ms, query)

    results = []
    for item in data.get("results", [])[:10]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        ))

    terms = _query_terms(query)

    t = timer()
    try:
        wiki_results = await _wikipedia_api_search(query, config.wikipedia.contact)
        await log_search_call(
            config, "wikipedia_api", "api", "ok" if wiki_results else "empty",
            result_count=len(wiki_results), elapsed_ms=t.elapsed_ms, query=query,
        )
    except httpx.HTTPError as e:
        wiki_results = []
        await log_search_call(
            config, "wikipedia_api", "api", "error",
            error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
        )
    results = _merge(results, wiki_results)

    t = timer()
    try:
        wikidata_results = await _wikidata_api_search(query, config.wikipedia.contact)
        await log_search_call(
            config, "wikidata_api", "api", "ok" if wikidata_results else "empty",
            result_count=len(wikidata_results), elapsed_ms=t.elapsed_ms, query=query,
        )
    except httpx.HTTPError as e:
        wikidata_results = []
        await log_search_call(
            config, "wikidata_api", "api", "error",
            error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
        )
    results = _merge(results, wikidata_results)

    if config.brave.api_key:
        t = timer()
        try:
            brave_results = await _brave_api_search(query, config.brave.api_key)
            await log_search_call(
                config, "brave", "api", "ok" if brave_results else "empty",
                result_count=len(brave_results), elapsed_ms=t.elapsed_ms, query=query,
            )
        except httpx.HTTPError as e:
            # Quota exhausted or the API is unreachable -- fall through with
            # whatever SearXNG's other engines already returned rather than
            # losing the whole search over one engine's fallback failing.
            brave_results = []
            await log_search_call(
                config, "brave", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
            )
        results = _merge(results, brave_results)

    if sum(_is_relevant(result, terms) for result in results) < MIN_RESULTS_BEFORE_TAVILY_FALLBACK and config.tavily.api_key:
        t = timer()
        try:
            tavily_results = await _tavily_api_search(query, config.tavily.api_key)
            await log_search_call(
                config, "tavily", "api", "ok" if tavily_results else "empty",
                result_count=len(tavily_results), elapsed_ms=t.elapsed_ms, query=query,
            )
        except httpx.HTTPError as e:
            # Quota exhausted or the API is unreachable -- fall through with
            # whatever's already been found rather than losing the search.
            tavily_results = []
            await log_search_call(
                config, "tavily", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
            )
        results = _merge(results, tavily_results)

    if sum(_is_relevant(result, terms) for result in results) < MIN_RESULTS_BEFORE_SERPER_FALLBACK and config.serper.api_key:
        t = timer()
        try:
            serper_results = await _serper_api_search(query, config.serper.api_key)
            await log_search_call(
                config, "serper", "api", "ok" if serper_results else "empty",
                result_count=len(serper_results), elapsed_ms=t.elapsed_ms, query=query,
            )
        except httpx.HTTPError as e:
            serper_results = []
            await log_search_call(
                config, "serper", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
            )
        results = _merge(results, serper_results)

    return _rank_results(results, query)


async def check_providers_now(config: Config) -> dict:
    """Fires one lightweight query at each provider right now (unconditionally,
    unlike web_search()'s Tavily gating) and logs each -- gives an
    authoritative live answer to "is X currently responding" instead of
    relying on how recently something happened to call it."""
    probe = "test"
    out = {}

    await _throttle_searxng(config.searxng.min_interval_seconds)
    t = timer()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{config.searxng.url.rstrip('/')}/search", params={"q": probe, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
        await _log_searxng_engines(config, data, t.elapsed_ms, probe)
        contributed: dict[str, int] = {}
        for item in data.get("results", []):
            for engine in item.get("engines") or [item.get("engine")]:
                if engine:
                    contributed[engine] = contributed.get(engine, 0) + 1
        for engine, n in contributed.items():
            out[engine] = {"responding": n > 0, "result_count": n, "error": None}
        for engine, reason in data.get("unresponsive_engines", []):
            if engine not in out:
                out[engine] = {"responding": False, "result_count": 0, "error": reason}
    except httpx.HTTPError as e:
        await log_search_call(config, "searxng", "scrape", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
        out["searxng"] = {"responding": False, "result_count": 0, "error": str(e)}

    t = timer()
    try:
        wiki_results = await _wikipedia_api_search(probe, config.wikipedia.contact)
        await log_search_call(config, "wikipedia_api", "api", "ok" if wiki_results else "empty", result_count=len(wiki_results), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikipedia_api"] = {"responding": len(wiki_results) > 0, "result_count": len(wiki_results), "error": None}
    except httpx.HTTPError as e:
        await log_search_call(config, "wikipedia_api", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikipedia_api"] = {"responding": False, "result_count": 0, "error": str(e)}

    t = timer()
    try:
        wikidata_results = await _wikidata_api_search(probe, config.wikipedia.contact)
        await log_search_call(config, "wikidata_api", "api", "ok" if wikidata_results else "empty", result_count=len(wikidata_results), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikidata_api"] = {"responding": len(wikidata_results) > 0, "result_count": len(wikidata_results), "error": None}
    except httpx.HTTPError as e:
        await log_search_call(config, "wikidata_api", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikidata_api"] = {"responding": False, "result_count": 0, "error": str(e)}

    if config.brave.api_key:
        t = timer()
        try:
            results = await _brave_api_search(probe, config.brave.api_key)
            await log_search_call(config, "brave", "api", "ok" if results else "empty", result_count=len(results), elapsed_ms=t.elapsed_ms, query=probe)
            out["brave"] = {"responding": len(results) > 0, "result_count": len(results), "error": None}
        except httpx.HTTPError as e:
            await log_search_call(config, "brave", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
            out["brave"] = {"responding": False, "result_count": 0, "error": str(e)}
    else:
        out["brave"] = {"responding": None, "result_count": 0, "error": "no api key configured"}

    if config.tavily.api_key:
        t = timer()
        try:
            results = await _tavily_api_search(probe, config.tavily.api_key)
            await log_search_call(config, "tavily", "api", "ok" if results else "empty", result_count=len(results), elapsed_ms=t.elapsed_ms, query=probe)
            out["tavily"] = {"responding": len(results) > 0, "result_count": len(results), "error": None}
        except httpx.HTTPError as e:
            await log_search_call(config, "tavily", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
            out["tavily"] = {"responding": False, "result_count": 0, "error": str(e)}
    else:
        out["tavily"] = {"responding": None, "result_count": 0, "error": "no api key configured"}

    if config.serper.api_key:
        t = timer()
        try:
            results = await _serper_api_search(probe, config.serper.api_key)
            await log_search_call(config, "serper", "api", "ok" if results else "empty", result_count=len(results), elapsed_ms=t.elapsed_ms, query=probe)
            out["serper"] = {"responding": len(results) > 0, "result_count": len(results), "error": None}
        except httpx.HTTPError as e:
            await log_search_call(config, "serper", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe)
            out["serper"] = {"responding": False, "result_count": 0, "error": str(e)}
    else:
        out["serper"] = {"responding": None, "result_count": 0, "error": "no api key configured"}

    return out
