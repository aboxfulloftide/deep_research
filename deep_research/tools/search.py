import asyncio
import math
import re
import time
import urllib.parse

import httpx

from deep_research.config import Config
from deep_research.kb.canonical import normalize_url
from deep_research.models import ProviderObservation, SearchResult
from deep_research.tools.search_cache import (
    get_cached_results,
    normalize_cache_key,
    store_cached_results,
)
from deep_research.tools.search_usage import (
    log_search_call,
    provider_monthly_quota_exhausted,
    providers_allowed_by_circuit_breaker,
    serper_behind_pace,
    timer,
)

def classify_http_error(exc: Exception) -> str:
    """Best-effort category for a provider call failure -- stored alongside
    the log row so later logic (monthly quota detection today, richer
    waterfall/circuit-breaker decisions later) can tell "genuinely down"
    apart from "rate-limited for a few seconds" without re-parsing free-text
    error messages. Deliberately does not attempt a separate
    "quota_exhausted" category here -- that's a *derived* state from several
    rate_limited rows accumulating over a calendar month
    (provider_monthly_quota_exhausted), not something visible from one
    exception."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return "rate_limited"
        if status in (401, 403):
            return "auth_error"
        if status == 404:
            return "not_found"
        if status >= 500:
            return "server_error"
        return "unknown"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)):
        return "network_error"
    return "unknown"


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
_PROPER_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:['’]s)?|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9]*(?:['’]s)?|[A-Z]{2,}))*"
)
_GENERIC_SENTENCE_OPENERS = {
    "a", "an", "as", "at", "by", "for", "from", "in", "it", "on",
    "proof", "that", "the", "these", "this", "those", "when", "while",
}
# Below this many relevant results accumulated so far, the next waterfall
# tier (Serper, then Brave, then Tavily) is worth its call; at or above it,
# that tier is skipped -- "fewer, better provider calls" instead of firing
# every configured provider on every search regardless of whether SearXNG
# and Wikipedia already answered the question.
MIN_SUFFICIENT_RELEVANT_RESULTS = 3

# Google CSE and Startpage are intentionally not part of SearXNG's default
# engine set: literal claim sentences caused sustained, repetitive traffic
# that eventually tripped their bot protection. Verification may opt into
# them for an LLM-generated alternate query, where the wording is both more
# search-like and different from the literal first attempt. Keeping this list
# here (rather than globally enabling the engines in settings.yml) preserves
# that boundary for interactive and other ordinary searches too.
SEARXNG_BASE_ENGINES = ("duckduckgo", "bing")
# Mojeek's yield collapsed to near-zero (1,401 errors of 1,628 calls, 0.3 avg
# results -- see RESEARCH_WORK_HANDOFF.md's provider-yield table) while still
# costing a full SearXNG round-trip on every ordinary search. Removed from
# SEARXNG_BASE_ENGINES so it no longer fires on real traffic, but kept
# reachable here so check_providers_now() can still probe it for a deliberate,
# low-volume health check -- if it shows sustained recovery, promote it back.
SEARXNG_HEALTH_PROBE_ENGINES = SEARXNG_BASE_ENGINES + ("mojeek",)
SEARXNG_RECOVERED_ENGINES = ("google cse", "startpage")
SEARXNG_DUCKDUCKGO_COOLDOWN_HOURS = 3
SEARXNG_ALTERNATE_QUERY_ENGINE_MAX_ATTEMPTS = 20
SEARXNG_ALTERNATE_QUERY_ENGINE_COOLDOWN_HOURS = 48

_searxng_throttle_lock = asyncio.Lock()
_searxng_last_call_at: float | None = None

# In-process-only request coalescing, keyed by the same cache key
# search_cache.py uses. No cross-process coordination -- that would need a
# much heavier mechanism (a durable lease) for a benefit this doesn't need
# yet. A concurrent web_search() call for the same key while one is already
# computing awaits the in-flight Future instead of independently spending
# its own provider calls.
_inflight_searches: dict[str, asyncio.Future] = {}
_inflight_searches_lock = asyncio.Lock()


async def _coalesced(cache_key: str, compute):
    async with _inflight_searches_lock:
        future = _inflight_searches.get(cache_key)
        owner = future is None
        if owner:
            future = asyncio.get_event_loop().create_future()
            _inflight_searches[cache_key] = future
    if not owner:
        return await future
    try:
        result = await compute()
        future.set_result(result)
        return result
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        async with _inflight_searches_lock:
            if _inflight_searches.get(cache_key) is future:
                del _inflight_searches[cache_key]


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


BRAVE_RATE_LIMIT_RETRY_SECONDS = 1.1  # free tier allows 1 request/second


async def _brave_api_search(query: str, api_key: str) -> list[SearchResult]:
    """Direct call to Brave's official Search API. SearXNG's own brave engine
    is disabled in searxng/settings.yml (it got rate limited under sustained
    query volume, same as google cse/startpage) -- this is now the only
    source of Brave results, called unconditionally rather than only as a
    fallback.

    Deliberately returns unstamped results (no provider/canonical_url) --
    _brave_search_layered() stamps them, since it alone knows whether this
    call was the free or paid key ("brave" vs "brave_fallback")."""
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


async def _brave_api_search_with_retry(query: str, api_key: str) -> list[SearchResult]:
    """The free tier allows 1 request/second, so under
    verification_concurrency=2 an occasional per-second 429 is expected and
    not a reason to spend the paid fallback key -- retry once after a beat
    before treating it as a real failure. Monthly-quota exhaustion also
    presents as 429; that one fails the retry too and falls through to the
    fallback key in _brave_search_layered."""
    try:
        return await _brave_api_search(query, api_key)
    except httpx.HTTPStatusError as e:
        if classify_http_error(e) != "rate_limited":
            raise
    await asyncio.sleep(BRAVE_RATE_LIMIT_RETRY_SECONDS)
    return await _brave_api_search(query, api_key)


async def _brave_search_layered(
    query: str, config: Config, search_ids: dict | None = None,
) -> list[SearchResult]:
    """Two Brave subscriptions, spent in order: the free-tier key
    (config.brave.api_key, ~2000 queries/month, 1 req/s) first, and the paid
    key (config.brave.fallback_api_key, 50 req/s, ~3000/month budgeted) only
    when the free key actually *errors* -- quota exhausted, persistent 429,
    subscription problem. Once the primary key's 429 retry also fails, its
    stored error opens a persistent circuit for the rest of the calendar
    month; subsequent searches go directly to the fallback key. A genuinely
    empty result set from the free key is an answer, not a failure, and never
    triggers paid spend. Each key logs under its own provider name ("brave" /
    "brave_fallback") so /search-usage shows exactly when the paid key started
    carrying traffic."""
    search_ids = search_ids or {}
    primary_paused = (
        bool(config.brave.api_key)
        and await provider_monthly_quota_exhausted(config, "brave")
    )
    if config.brave.api_key and not primary_paused:
        t = timer()
        try:
            results = await _brave_api_search_with_retry(query, config.brave.api_key)
            await log_search_call(
                config, "brave", "api", "ok" if results else "empty",
                result_count=len(results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
            )
            return _finalize_provider_results(results, "brave", query)
        except httpx.HTTPError as e:
            await log_search_call(
                config, "brave", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
                error_category=classify_http_error(e), **search_ids,
            )
    if config.brave.fallback_api_key:
        t = timer()
        try:
            results = await _brave_api_search(query, config.brave.fallback_api_key)
            await log_search_call(
                config, "brave_fallback", "api", "ok" if results else "empty",
                result_count=len(results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
            )
            return _finalize_provider_results(results, "brave_fallback", query)
        except httpx.HTTPError as e:
            await log_search_call(
                config, "brave_fallback", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
                error_category=classify_http_error(e), **search_ids,
            )
    return []


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
    return _finalize_provider_results(results, "tavily", query)


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
    return _finalize_provider_results(results, "serper", query)


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
    return _finalize_provider_results(results, "wikipedia_api", query)


async def _wikidata_api_search(query: str, contact: str = "") -> list[SearchResult]:
    """Direct call to Wikidata's own entity-search API (wbsearchentities) --
    replaces SearXNG's "wikidata" engine, which failed on 46 of 46 logged
    calls (timeouts + "too many requests", each timeout stalling the whole
    SearXNG response) and is now disabled in searxng/settings.yml. Same
    no-key Wikimedia infrastructure and User-Agent policy as
    _wikipedia_api_search; logged as "wikidata_api".

    wbsearchentities matches entity labels/aliases (returning a proper
    display label + description, unlike Wikidata's REST full-text search,
    whose result titles are opaque Q-ids). Callers should pass an entity-like
    label, not an entire factual sentence; web_search derives that label from
    the best relevant Wikipedia result."""
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
    return _finalize_provider_results(results, "wikidata_api", query)


async def _log_searxng_engines(
    config: Config, data: dict, elapsed_ms: int, query: str,
    selected_engines: tuple[str, ...] | None = None,
    search_ids: dict | None = None,
) -> None:
    """SearXNG fans one call out to several underlying engines (duckduckgo,
    bing, mojeek, ...) and merges their results -- log each engine
    that actually contributed or failed separately, rather than lumping
    everything under "duckduckgo", so /api/search-usage reflects what's
    really answering queries."""
    search_ids = search_ids or {}
    contributed: dict[str, int] = {}
    for item in data.get("results", []):
        for engine in item.get("engines") or [item.get("engine")]:
            if engine:
                contributed[engine] = contributed.get(engine, 0) + 1
    failed = {engine for engine, _ in data.get("unresponsive_engines", [])}
    for engine, reason in data.get("unresponsive_engines", []):
        if engine not in contributed:
            await log_search_call(
                config, engine, "scrape", "error",
                error_message=reason, elapsed_ms=elapsed_ms, query=query, **search_ids,
            )
    for engine, count in contributed.items():
        await log_search_call(
            config, engine, "scrape", "ok",
            result_count=count, elapsed_ms=elapsed_ms, query=query, **search_ids,
        )
    # An explicitly selected trial engine that returned neither a result nor
    # an error was still attempted and must count toward its rolling cap.
    for engine in selected_engines or ():
        if engine not in contributed and engine not in failed:
            await log_search_call(
                config, engine, "scrape", "empty",
                result_count=0, elapsed_ms=elapsed_ms, query=query, **search_ids,
            )
    if not contributed and not data.get("unresponsive_engines"):
        await log_search_call(
            config, "searxng", "scrape", "empty", result_count=0, elapsed_ms=elapsed_ms,
            query=query, **search_ids,
        )


def _finalize_provider_results(results: list[SearchResult], provider: str, query: str) -> list[SearchResult]:
    """Stamp this provider's own rank and the shared canonical URL onto each
    of its raw results, before _merge() folds cross-provider duplicates --
    without this, _merge() has no canonical_url to key on and no per-provider
    rank to preserve when two providers return the same page."""
    return [
        result.model_copy(update={
            "canonical_url": normalize_url(result.url),
            "observations": [ProviderObservation(provider=provider, rank=rank, query=query)],
        })
        for rank, result in enumerate(results, start=1)
    ]


def _merge(results: list[SearchResult], new: list[SearchResult]) -> list[SearchResult]:
    """Merge by canonical URL identity, preserving every provider's
    observation instead of discarding the losing duplicate's rank/signal.
    Callers must pass `new` results that already carry canonical_url (via
    _finalize_provider_results) -- an unstamped result's empty canonical_url
    would collide with every other unstamped result."""
    by_canonical: dict[str, SearchResult] = {r.canonical_url: r for r in results}
    for r in new:
        existing = by_canonical.get(r.canonical_url)
        if existing is None:
            by_canonical[r.canonical_url] = r
        else:
            existing.observations.extend(r.observations)
    return list(by_canonical.values())


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
    if len(terms) <= 1:
        return _relevance_score(result, terms) >= 1
    # _relevance_score's title weighting (2x) means a flat score threshold of
    # 2 is satisfied by a single incidentally-shared title word, regardless
    # of how many distinct terms the query actually has -- a page titled
    # "Potential" is not evidence for a 7-term query about "potential risks
    # and benefits of intermittent fasting for metabolic health" just
    # because they share that one word. Require distinct-term coverage that
    # scales with query specificity instead, so a genuinely on-topic page
    # (sharing several of the query's words, even reworded) still passes,
    # but a single coincidental match cannot no matter its title placement.
    matched = terms & (
        set(_QUERY_TOKEN_RE.findall(result.title.lower()))
        | set(_QUERY_TOKEN_RE.findall(result.snippet.lower()))
    )
    required = min(len(terms), max(2, math.ceil(len(terms) / 3)))
    return len(matched) >= required


RRF_K = 60  # standard RRF damping constant, same value as kb_search.py's own
            # RRF_K -- not imported from there to keep web search and KB
            # search decoupled; the formula is generic enough to duplicate as
            # a small local constant rather than create a cross-module
            # dependency for one integer.


def _rrf_score(result: SearchResult) -> float:
    """Reciprocal-rank-fusion score summed across every provider that
    surfaced this canonical URL -- rewards cross-provider agreement instead
    of whichever single provider happened to rank a result first. Sourced
    from the per-provider ranks _merge() already preserves on
    SearchResult.observations."""
    return sum(1.0 / (RRF_K + observation.rank) for observation in result.observations)


def _rank_results(results: list[SearchResult], query: str, limit: int = 10) -> list[SearchResult]:
    """Rank by cross-provider agreement (RRF) first, breaking ties with
    lexical relevance -- RRF alone can't distinguish "providers agree on an
    irrelevant stale result" from "providers agree on the right answer," so
    relevance stays a real, transparent secondary signal instead of being
    replaced outright."""
    terms = _query_terms(query)
    return sorted(
        results,
        key=lambda result: (_rrf_score(result), _relevance_score(result, terms)),
        reverse=True,
    )[:limit]


def _wikidata_entity_query(query: str, wikipedia_results: list[SearchResult]) -> str | None:
    """Turn a general web query into the label-shaped input Wikidata expects.

    Wikipedia full-text search is good at mapping a sentence to a page;
    Wikidata's wbsearchentities endpoint is good at mapping that page title
    to an entity. Only use a Wikipedia title with real lexical overlap. A
    naturally short query can already be an entity label and is safe as a
    fallback. Long sentences without a relevant page are skipped rather than
    spending an API call that is structurally guaranteed to return nothing.
    """
    # Prefer an explicit proper name from the original query. Wikipedia can
    # rank a secondary detail above the subject (for example, mapping a claim
    # about Masayoshi Son to UC Berkeley because it mentions university).
    proper_names = []
    for match in _PROPER_NAME_RE.finditer(query):
        candidate = re.sub(r"['’]s$", "", match.group(0)).strip()
        first_word = candidate.split(maxsplit=1)[0].casefold()
        if first_word not in _GENERIC_SENTENCE_OPENERS:
            proper_names.append((match.start(), candidate))
    if proper_names:
        return min(proper_names)[-1]

    terms = _query_terms(query)
    ranked_wikipedia = _rank_results(wikipedia_results, query)
    for result in ranked_wikipedia:
        if _is_relevant(result, terms):
            return result.title.strip() or None
    if 0 < len(terms) <= 4:
        return query.strip() or None
    return None


async def web_search(
    query: str, config: Config, *,
    include_alternate_query_engines: bool = False,
    capability: str | None = None,
    run_id: str | None = None,
    plan_id: str | None = None,
    facet_id: str | None = None,
    attempt_id: str | None = None,
    freshness: str = "default",
) -> list[SearchResult]:
    """SearXNG (Bing/DuckDuckGo) + Wikipedia unconditionally, then an adaptive
    waterfall: Wikidata only when capability="grounding" (its yield is ~67%
    empty on ordinary queries, see RESEARCH_WORK_HANDOFF.md); Serper, then
    Brave, then Tavily each only called when the results accumulated so far
    don't already meet MIN_SUFFICIENT_RELEVANT_RESULTS -- fewer, better
    provider calls rather than firing every configured provider on every
    search. google cse/startpage are disabled in searxng/settings.yml -- they
    got rate limited/CAPTCHA'd under sustained query volume and have no
    documented quota to plan around, unlike Brave's and Tavily's metered APIs.

    Every provider call is logged to search_usage.py's SQLite log (best
    effort, never raises) so /api/search-usage can answer "how many searches
    have we used" and "is duckduckgo/brave/tavily currently responding"
    without grepping run logs or curling each provider by hand.

    capability/run_id/plan_id/facet_id/attempt_id are optional context IDs
    forwarded to every logged call so a search can later be joined back to
    the research run/plan/facet/attempt (or claim verification) that made
    it -- callers with no such concept (e.g. the plain chat tool loop) simply
    omit them. capability="grounding" additionally controls whether Wikidata
    is consulted at all.

    Results are cached (search_cache.py) by normalized query/capability/
    engine-set/freshness for config.search_cache.enabled callers, and
    concurrent identical searches are coalesced into one computation instead
    of each spending their own provider calls -- see _coalesced above."""
    search_ids = {
        "capability": capability, "run_id": run_id, "plan_id": plan_id,
        "facet_id": facet_id, "attempt_id": attempt_id,
    }
    cache_key = normalize_cache_key(
        query, capability=capability,
        include_alternate_query_engines=include_alternate_query_engines,
        freshness=freshness,
    )
    if config.search_cache.enabled:
        cached = await get_cached_results(config, cache_key)
        if cached is not None:
            await log_search_call(
                config, "cache", "cache", "hit",
                result_count=len(cached), query=query, **search_ids,
            )
            return cached

    async def _searxng_call(engines: tuple[str, ...]) -> list[SearchResult]:
        url = f"{config.searxng.url.rstrip('/')}/search"
        params = {"q": query, "format": "json", "engines": ",".join(engines)}
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
                error_category=classify_http_error(e), **search_ids,
            )
            # SearXNG is one provider, not a single point of failure. Continue to
            # the direct providers below when it is unavailable.
            data = {"results": []}

        await _log_searxng_engines(
            config, data, t.elapsed_ms, query, selected_engines=engines,
            search_ids=search_ids,
        )

        engine_results = []
        for rank, item in enumerate(data.get("results", [])[:10], start=1):
            item_url = item.get("url", "")
            # SearXNG already fuses per-engine duplicates before returning this
            # list, so `rank` here is the merged SearXNG position, not each
            # individual engine's own rank -- good enough for observability;
            # true per-engine ranks would require querying engines directly.
            hit_engines = [engine for engine in (item.get("engines") or [item.get("engine")]) if engine]
            engine_results.append(SearchResult(
                title=item.get("title", ""),
                url=item_url,
                snippet=item.get("content", ""),
                canonical_url=normalize_url(item_url),
                observations=[
                    ProviderObservation(provider=f"searxng:{engine}", rank=rank, query=query)
                    for engine in hit_engines
                ] or [ProviderObservation(provider="searxng", rank=rank, query=query)],
            ))
        return engine_results

    async def _compute() -> list[SearchResult]:
        duckduckgo_allowed = await providers_allowed_by_circuit_breaker(
            config, ("duckduckgo",), max_attempts=None,
            cooldown_hours=SEARXNG_DUCKDUCKGO_COOLDOWN_HOURS,
        )
        # Always select the base engines explicitly. Otherwise omitting
        # DuckDuckGo here would merely make SearXNG's default configuration add it
        # back, defeating the cooldown while Bing remains healthy.
        selected_engines = tuple(
            engine for engine in SEARXNG_BASE_ENGINES
            if engine != "duckduckgo" or engine in duckduckgo_allowed
        )
        included_recovered_engines = False
        if include_alternate_query_engines:
            recovered = await providers_allowed_by_circuit_breaker(
                config, SEARXNG_RECOVERED_ENGINES,
                max_attempts=SEARXNG_ALTERNATE_QUERY_ENGINE_MAX_ATTEMPTS,
                cooldown_hours=SEARXNG_ALTERNATE_QUERY_ENGINE_COOLDOWN_HOURS,
            )
            selected_engines += tuple(
                engine for engine in SEARXNG_RECOVERED_ENGINES if engine in recovered
            )
            included_recovered_engines = True

        results = await _searxng_call(selected_engines)
        terms = _query_terms(query)

        t = timer()
        try:
            wiki_results = await _wikipedia_api_search(query, config.wikipedia.contact)
            await log_search_call(
                config, "wikipedia_api", "api", "ok" if wiki_results else "empty",
                result_count=len(wiki_results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
            )
        except httpx.HTTPError as e:
            wiki_results = []
            await log_search_call(
                config, "wikipedia_api", "api", "error",
                error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
                error_category=classify_http_error(e), **search_ids,
            )
        results = _merge(results, wiki_results)

        # Wikidata is an entity/reference lookup, not a general search -- only
        # derive an entity query and call it when a facet/caller explicitly
        # wants grounding (see RESEARCH_WORK_HANDOFF.md: ~67% of past Wikidata
        # calls returned nothing, because most queries were never entity-shaped
        # to begin with). Wikipedia stays unconditional above; its yield doesn't
        # have the same waste problem.
        if capability == "grounding":
            wikidata_query = _wikidata_entity_query(query, wiki_results)
            if wikidata_query:
                t = timer()
                try:
                    wikidata_results = await _wikidata_api_search(
                        wikidata_query, config.wikipedia.contact,
                    )
                    await log_search_call(
                        config, "wikidata_api", "api", "ok" if wikidata_results else "empty",
                        result_count=len(wikidata_results), elapsed_ms=t.elapsed_ms, query=wikidata_query,
                        **search_ids,
                    )
                    results = _merge(results, wikidata_results)
                except httpx.HTTPError as e:
                    await log_search_call(
                        config, "wikidata_api", "api", "error",
                        error_message=str(e), elapsed_ms=t.elapsed_ms, query=wikidata_query,
                        error_category=classify_http_error(e), **search_ids,
                    )

        # SearXNG's base engines (duckduckgo/bing) sometimes degrade silently --
        # not an HTTP error, just irrelevant results (e.g. a scraper serving a
        # dictionary "instant answer" page instead of real organic results) --
        # so this checks actual relevance, not just whether the call succeeded.
        # Only escalate to startpage/google_cse here if the caller didn't
        # already fold them into the first call above (include_alternate_query_
        # engines=True); otherwise this would just repeat that same call.
        # Gated by the same circuit breaker as that opt-in path, so if these
        # engines are already rate-limited/CAPTCHA'd this is a no-op, not a
        # second way to hammer them.
        if (
            not included_recovered_engines
            and sum(_is_relevant(result, terms) for result in results) < MIN_SUFFICIENT_RELEVANT_RESULTS
        ):
            recovered = await providers_allowed_by_circuit_breaker(
                config, SEARXNG_RECOVERED_ENGINES,
                max_attempts=SEARXNG_ALTERNATE_QUERY_ENGINE_MAX_ATTEMPTS,
                cooldown_hours=SEARXNG_ALTERNATE_QUERY_ENGINE_COOLDOWN_HOURS,
            )
            recovered_engines = tuple(
                engine for engine in SEARXNG_RECOVERED_ENGINES if engine in recovered
            )
            if recovered_engines:
                results = _merge(results, await _searxng_call(recovered_engines))

        # Adaptive waterfall from here: Serper, then Brave, then Tavily each only
        # fire when the results accumulated so far aren't already sufficient --
        # "fewer, better provider calls" instead of calling every configured
        # provider on every search regardless of whether SearXNG/Wikipedia/
        # Wikidata already answered the question.
        #
        # Serper is the one exception: its credits expire on a fixed date and
        # are forfeited unused rather than rolling over (unlike Brave/Tavily's
        # monthly-resetting quotas), so serper_behind_pace() can force a call
        # here even when earlier results already look sufficient -- spending
        # the balance down before it expires is strictly better than leaving
        # it unspent. A no-op (never True) until config.serper.quota_expires_at
        # is actually configured.
        if config.serper.api_key and (
            sum(_is_relevant(result, terms) for result in results) < MIN_SUFFICIENT_RELEVANT_RESULTS
            or await serper_behind_pace(config)
        ):
            t = timer()
            try:
                serper_results = await _serper_api_search(query, config.serper.api_key)
                await log_search_call(
                    config, "serper", "api", "ok" if serper_results else "empty",
                    result_count=len(serper_results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
                )
            except httpx.HTTPError as e:
                serper_results = []
                await log_search_call(
                    config, "serper", "api", "error",
                    error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
                    error_category=classify_http_error(e), **search_ids,
                )
            results = _merge(results, serper_results)

        if (
            (config.brave.api_key or config.brave.fallback_api_key)
            and sum(_is_relevant(result, terms) for result in results) < MIN_SUFFICIENT_RELEVANT_RESULTS
        ):
            # Both keys erroring falls through with whatever the other providers
            # already returned rather than losing the whole search -- each
            # attempt is logged inside _brave_search_layered.
            results = _merge(results, await _brave_search_layered(query, config, search_ids))

        if sum(_is_relevant(result, terms) for result in results) < MIN_SUFFICIENT_RELEVANT_RESULTS and config.tavily.api_key:
            t = timer()
            try:
                tavily_results = await _tavily_api_search(query, config.tavily.api_key)
                await log_search_call(
                    config, "tavily", "api", "ok" if tavily_results else "empty",
                    result_count=len(tavily_results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
                )
            except httpx.HTTPError as e:
                # Quota exhausted or the API is unreachable -- fall through with
                # whatever's already been found rather than losing the search.
                tavily_results = []
                await log_search_call(
                    config, "tavily", "api", "error",
                    error_message=str(e), elapsed_ms=t.elapsed_ms, query=query,
                    error_category=classify_http_error(e), **search_ids,
                )
            results = _merge(results, tavily_results)

        final = _rank_results(results, query)
        if config.search_cache.enabled:
            await store_cached_results(config, cache_key, final, freshness)
        return final

    if config.search_cache.enabled:
        return await _coalesced(cache_key, _compute)
    return await _compute()


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
                f"{config.searxng.url.rstrip('/')}/search",
                params={
                    "q": probe,
                    "format": "json",
                    # Do not rely on SearXNG defaults here. Wikipedia and
                    # Wikidata scrape engines are retired; their direct APIs
                    # are checked separately below. Mojeek is included here
                    # (unlike ordinary web_search() traffic) so its health can
                    # still be observed for a possible future recovery.
                    "engines": ",".join(SEARXNG_HEALTH_PROBE_ENGINES),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        await _log_searxng_engines(
            config, data, t.elapsed_ms, probe,
            selected_engines=SEARXNG_HEALTH_PROBE_ENGINES,
        )
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
        await log_search_call(config, "searxng", "scrape", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
        out["searxng"] = {"responding": False, "result_count": 0, "error": str(e)}

    t = timer()
    try:
        wiki_results = await _wikipedia_api_search(probe, config.wikipedia.contact)
        await log_search_call(config, "wikipedia_api", "api", "ok" if wiki_results else "empty", result_count=len(wiki_results), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikipedia_api"] = {"responding": len(wiki_results) > 0, "result_count": len(wiki_results), "error": None}
    except httpx.HTTPError as e:
        await log_search_call(config, "wikipedia_api", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
        out["wikipedia_api"] = {"responding": False, "result_count": 0, "error": str(e)}

    t = timer()
    try:
        wikidata_results = await _wikidata_api_search(probe, config.wikipedia.contact)
        await log_search_call(config, "wikidata_api", "api", "ok" if wikidata_results else "empty", result_count=len(wikidata_results), elapsed_ms=t.elapsed_ms, query=probe)
        out["wikidata_api"] = {"responding": len(wikidata_results) > 0, "result_count": len(wikidata_results), "error": None}
    except httpx.HTTPError as e:
        await log_search_call(config, "wikidata_api", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
        out["wikidata_api"] = {"responding": False, "result_count": 0, "error": str(e)}

    primary_brave_paused = await provider_monthly_quota_exhausted(config, "brave")
    for provider, api_key in (("brave", config.brave.api_key), ("brave_fallback", config.brave.fallback_api_key)):
        if not api_key:
            out[provider] = {"responding": None, "result_count": 0, "error": "no api key configured"}
            continue
        if provider == "brave" and primary_brave_paused:
            out[provider] = {
                "responding": None,
                "result_count": 0,
                "error": "monthly quota exhausted; paused until next month",
            }
            continue
        t = timer()
        try:
            results = await _brave_api_search(probe, api_key)
            await log_search_call(config, provider, "api", "ok" if results else "empty", result_count=len(results), elapsed_ms=t.elapsed_ms, query=probe)
            out[provider] = {"responding": len(results) > 0, "result_count": len(results), "error": None}
        except httpx.HTTPError as e:
            await log_search_call(config, provider, "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
            out[provider] = {"responding": False, "result_count": 0, "error": str(e)}

    if config.tavily.api_key:
        t = timer()
        try:
            results = await _tavily_api_search(probe, config.tavily.api_key)
            await log_search_call(config, "tavily", "api", "ok" if results else "empty", result_count=len(results), elapsed_ms=t.elapsed_ms, query=probe)
            out["tavily"] = {"responding": len(results) > 0, "result_count": len(results), "error": None}
        except httpx.HTTPError as e:
            await log_search_call(config, "tavily", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
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
            await log_search_call(config, "serper", "api", "error", error_message=str(e), elapsed_ms=t.elapsed_ms, query=probe, error_category=classify_http_error(e))
            out["serper"] = {"responding": False, "result_count": 0, "error": str(e)}
    else:
        out["serper"] = {"responding": None, "result_count": 0, "error": "no api key configured"}

    return out
