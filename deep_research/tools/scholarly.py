"""Domain-native scholarly discovery: OpenAlex (broad multi-disciplinary
work search) and arXiv (STEM preprints). Used for facets with
capability="scholarly" to supplement general web search with structured,
keyless, public APIs designed for exactly this kind of programmatic access
-- see RESEARCH_WORK_HANDOFF.md's routing-order section ("scholarly:
OpenAlex, Crossref, arXiv, Semantic Scholar, PubMed"). Neither API requires
authentication.
"""

import xml.etree.ElementTree as ET

import httpx

from deep_research.config import Config
from deep_research.kb.canonical import normalize_url
from deep_research.models import ProviderObservation, SearchResult
from deep_research.tools.search import _merge
from deep_research.tools.search_usage import log_search_call, timer

OPENALEX_API_URL = "https://api.openalex.org/works"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_USER_AGENT = "deep-research-kb-bot/1.0 (scholarly discovery) httpx"


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex returns abstracts as a word -> [positions] inverted index (a
    licensing workaround, not a data-quality choice) -- rebuild the linear
    text from it so it can be used as a snippet."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


async def _openalex_search(query: str, config: Config) -> list[SearchResult]:
    """OpenAlex Works search. Prefers the DOI URL as a result's identity --
    dedup-friendly (doi.org is already tiered "paper" by classify_source())
    -- falling back to the landing/open-access URL, then the OpenAlex work
    ID itself."""
    params = {"search": query, "per_page": 10}
    if config.wikipedia.contact:
        params["mailto"] = config.wikipedia.contact  # OpenAlex's "polite pool" for faster processing
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(OPENALEX_API_URL, params=params, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        data = resp.json()

    results = []
    for rank, work in enumerate(data.get("results", [])[:10], start=1):
        landing_url = (
            (work.get("primary_location") or {}).get("landing_page_url")
            or (work.get("open_access") or {}).get("oa_url")
        )
        url = work.get("doi") or landing_url or work.get("id") or ""
        if not url:
            continue
        title = work.get("title") or work.get("display_name") or ""
        snippet = _reconstruct_abstract(work.get("abstract_inverted_index"))
        results.append(SearchResult(
            title=title, url=url, snippet=snippet[:500],
            canonical_url=normalize_url(url),
            observations=[ProviderObservation(provider="openalex", rank=rank, query=query)],
        ))
    return results


async def _arxiv_search(query: str, config: Config) -> list[SearchResult]:
    """arXiv's Atom-based export API. Each entry's <id> is already an
    /abs/{id} URL -- normalize_url() already folds that against the
    /html/{id} rendering, so cross-tool dedup needs no new logic here."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            ARXIV_API_URL,
            params={"search_query": f"all:{query}", "max_results": 10},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

    results = []
    for rank, entry in enumerate(root.findall("atom:entry", _ARXIV_ATOM_NS), start=1):
        entry_id = (entry.findtext("atom:id", default="", namespaces=_ARXIV_ATOM_NS) or "").strip()
        if not entry_id:
            continue
        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_ATOM_NS) or "").strip()
        results.append(SearchResult(
            title=title, url=entry_id, snippet=summary[:500],
            canonical_url=normalize_url(entry_id),
            observations=[ProviderObservation(provider="arxiv", rank=rank, query=query)],
        ))
    return results


async def scholarly_search(
    query: str, config: Config, *,
    run_id: str | None = None, facet_id: str | None = None, attempt_id: str | None = None,
) -> list[SearchResult]:
    """Query OpenAlex and arXiv directly and merge their results via the
    same canonical-URL _merge() every other provider uses. Each provider's
    failure is isolated and logged like any other provider call; a total
    failure returns [] so the caller's general web_search() results still
    stand on their own -- this is a supplement, not a replacement."""
    search_ids = {
        "run_id": run_id, "facet_id": facet_id, "attempt_id": attempt_id,
        "capability": "scholarly",
    }
    results: list[SearchResult] = []

    t = timer()
    try:
        openalex_results = await _openalex_search(query, config)
        await log_search_call(
            config, "openalex", "api", "ok" if openalex_results else "empty",
            result_count=len(openalex_results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
        )
        results = _merge(results, openalex_results)
    except httpx.HTTPError as e:
        await log_search_call(
            config, "openalex", "api", "error",
            error_message=str(e), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
        )

    t = timer()
    try:
        arxiv_results = await _arxiv_search(query, config)
        await log_search_call(
            config, "arxiv", "api", "ok" if arxiv_results else "empty",
            result_count=len(arxiv_results), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
        )
        results = _merge(results, arxiv_results)
    except httpx.HTTPError as e:
        await log_search_call(
            config, "arxiv", "api", "error",
            error_message=str(e), elapsed_ms=t.elapsed_ms, query=query, **search_ids,
        )

    return results
