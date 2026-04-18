import httpx

from deep_research.config import Config
from deep_research.models import SearchResult


async def web_search(query: str, config: Config) -> list[SearchResult]:
    """Search using SearXNG JSON API."""
    url = f"{config.searxng.url.rstrip('/')}/search"
    params = {
        "q": query,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
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
