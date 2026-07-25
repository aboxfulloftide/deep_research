from pydantic import BaseModel, Field


class ProviderObservation(BaseModel):
    """One provider's own view of a search result: which provider surfaced
    it, at what rank in that provider's own list, for which query. Preserved
    across cross-provider merges instead of discarding the losing duplicate's
    signal (see search.py's _merge())."""

    provider: str
    rank: int
    query: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    canonical_url: str = ""
    observations: list[ProviderObservation] = Field(default_factory=list)


class ScrapedPage(BaseModel):
    url: str
    title: str
    text_content: str
    structured_data: dict | None = None
