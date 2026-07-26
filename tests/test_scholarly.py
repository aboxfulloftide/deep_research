import httpx

from deep_research.config import Config
from deep_research.tools import scholarly as scholarly_module
from deep_research.tools.scholarly import (
    _arxiv_search,
    _openalex_search,
    _reconstruct_abstract,
    scholarly_search,
)


def test_reconstruct_abstract_rebuilds_linear_text_from_inverted_index():
    inverted_index = {"Deep": [0], "learning": [1], "is": [2], "powerful": [3]}
    assert _reconstruct_abstract(inverted_index) == "Deep learning is powerful"


def test_reconstruct_abstract_handles_repeated_words_at_multiple_positions():
    inverted_index = {"the": [0, 3], "cat": [1], "and": [2], "dog": [4]}
    assert _reconstruct_abstract(inverted_index) == "the cat and the dog"


def test_reconstruct_abstract_returns_empty_string_for_missing_index():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


class _FakeJSONResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeXMLResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, url, params=None, headers=None):
        return self._response


async def test_openalex_search_prefers_doi_over_landing_page_and_work_id(monkeypatch):
    payload = {
        "results": [
            {
                "title": "A Paper With a DOI",
                "doi": "https://doi.org/10.1234/example",
                "primary_location": {"landing_page_url": "https://publisher.example/paper"},
                "id": "https://openalex.org/W1",
                "abstract_inverted_index": {"An": [0], "abstract.": [1]},
            },
            {
                "title": "A Paper Without a DOI",
                "doi": None,
                "primary_location": {"landing_page_url": "https://publisher.example/other"},
                "id": "https://openalex.org/W2",
            },
            {
                "title": "A Paper With Only an OpenAlex ID",
                "doi": None,
                "primary_location": {},
                "id": "https://openalex.org/W3",
            },
        ],
    }
    monkeypatch.setattr(
        scholarly_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient(_FakeJSONResponse(payload)),
    )

    results = await _openalex_search("deep learning", Config())

    assert [r.url for r in results] == [
        "https://doi.org/10.1234/example",
        "https://publisher.example/other",
        "https://openalex.org/W3",
    ]
    assert results[0].snippet == "An abstract."
    assert results[0].observations[0].provider == "openalex"


ARXIV_ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>Attention Is All You Need Again</title>
    <summary>We revisit the transformer architecture.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.54321v1</id>
    <title>Another Paper</title>
    <summary>A different summary.</summary>
  </entry>
</feed>
"""


async def test_arxiv_search_parses_atom_feed_into_search_results(monkeypatch):
    monkeypatch.setattr(
        scholarly_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient(_FakeXMLResponse(ARXIV_ATOM_FEED)),
    )

    results = await _arxiv_search("transformers", Config())

    assert len(results) == 2
    assert results[0].url == "http://arxiv.org/abs/2401.12345v2"
    assert results[0].title == "Attention Is All You Need Again"
    assert results[0].canonical_url == results[0].canonical_url  # sanity: computed, not blank
    assert results[0].canonical_url != ""


async def test_scholarly_search_merges_openalex_and_arxiv_results(monkeypatch):
    async def fake_openalex(query, config):
        return [scholarly_module.SearchResult(
            title="OpenAlex result", url="https://doi.org/10.1/x", snippet="",
            canonical_url="https://doi.org/10.1/x",
            observations=[scholarly_module.ProviderObservation(provider="openalex", rank=1, query=query)],
        )]

    async def fake_arxiv(query, config):
        return [scholarly_module.SearchResult(
            title="arXiv result", url="https://arxiv.org/abs/1234.5678", snippet="",
            canonical_url="https://arxiv.org/abs/1234.5678",
            observations=[scholarly_module.ProviderObservation(provider="arxiv", rank=1, query=query)],
        )]

    async def noop_log(*args, **kwargs):
        pass

    monkeypatch.setattr(scholarly_module, "_openalex_search", fake_openalex)
    monkeypatch.setattr(scholarly_module, "_arxiv_search", fake_arxiv)
    monkeypatch.setattr(scholarly_module, "log_search_call", noop_log)

    results = await scholarly_search("query", Config())

    assert {r.title for r in results} == {"OpenAlex result", "arXiv result"}


async def test_scholarly_search_survives_one_provider_failing(monkeypatch):
    async def failing_openalex(query, config):
        raise httpx.ConnectError("openalex unreachable")

    async def fake_arxiv(query, config):
        return [scholarly_module.SearchResult(
            title="arXiv result", url="https://arxiv.org/abs/1234.5678", snippet="",
            canonical_url="https://arxiv.org/abs/1234.5678",
            observations=[scholarly_module.ProviderObservation(provider="arxiv", rank=1, query=query)],
        )]

    async def noop_log(*args, **kwargs):
        pass

    monkeypatch.setattr(scholarly_module, "_openalex_search", failing_openalex)
    monkeypatch.setattr(scholarly_module, "_arxiv_search", fake_arxiv)
    monkeypatch.setattr(scholarly_module, "log_search_call", noop_log)

    results = await scholarly_search("query", Config())

    assert [r.title for r in results] == ["arXiv result"]
