import socket as socket_module

import httpx
import pytest

import deep_research.tools.fetch as fetch_module
from deep_research.config import Config
from deep_research.tools.fetch import (
    FetchTooLargeError,
    UnsafeURLError,
    _is_safe_host,
    safe_fetch,
)

PUBLIC_IP = "8.8.8.8"
_REAL_GETADDRINFO = socket_module.getaddrinfo


def _fake_getaddrinfo(public_hosts: dict[str, str]):
    """Overrides DNS resolution only for the given fake hostnames; anything
    else (e.g. a literal IP address) still resolves for real, so tests that
    want genuine socket-level classification of an IP literal don't need
    this fixture at all."""
    def getaddrinfo(hostname, port):
        ip = public_hosts.get(hostname)
        if ip is not None:
            return [(None, None, None, "", (ip, 0))]
        return _REAL_GETADDRINFO(hostname, port)
    return getaddrinfo


def test_is_safe_host_rejects_literal_private_and_loopback_and_link_local_addresses():
    # Real socket resolution for IP literals is local (no network call) --
    # no DNS mocking needed for these.
    assert _is_safe_host("127.0.0.1") is False
    assert _is_safe_host("10.0.0.5") is False
    assert _is_safe_host("192.168.1.1") is False
    assert _is_safe_host("169.254.169.254") is False


def test_is_safe_host_accepts_a_genuinely_public_address():
    assert _is_safe_host(PUBLIC_IP) is True


def test_is_safe_host_rejects_dns_rebinding_to_a_private_address(monkeypatch):
    """The actual point of resolving the hostname rather than string-matching
    it -- a normal-looking name that resolves to an internal address must
    still be rejected."""
    monkeypatch.setattr(
        fetch_module.socket, "getaddrinfo",
        _fake_getaddrinfo({"evil.example.test": "127.0.0.1"}),
    )
    assert _is_safe_host("evil.example.test") is False


class _FakeStreamResponse:
    def __init__(self, status_code, headers, url, chunks=()):
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self._chunks = list(chunks)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", str(self.url))
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class _FakeClient:
    def __init__(self, responses_by_url):
        self._responses_by_url = responses_by_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, headers=None):
        return _FakeStreamContext(self._responses_by_url[url])


@pytest.fixture(autouse=True)
def _mock_public_dns(monkeypatch):
    monkeypatch.setattr(
        fetch_module.socket, "getaddrinfo",
        _fake_getaddrinfo({"example.test": PUBLIC_IP, "redirect-target.test": PUBLIC_IP, "internal.test": "127.0.0.1"}),
    )


async def test_safe_fetch_returns_the_document_on_a_direct_success(monkeypatch):
    response = _FakeStreamResponse(
        200, {"content-type": "text/html; charset=utf-8"}, "https://example.test/page",
        chunks=[b"<html>", b"hello</html>"],
    )
    monkeypatch.setattr(
        fetch_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient({"https://example.test/page": response}),
    )

    doc = await safe_fetch("https://example.test/page", Config())

    assert doc.status_code == 200
    assert doc.mime_type == "text/html"
    assert doc.content == b"<html>hello</html>"
    assert doc.final_url == "https://example.test/page"


async def test_safe_fetch_follows_a_safe_redirect_chain(monkeypatch):
    redirect = _FakeStreamResponse(
        302, {"location": "https://redirect-target.test/final"}, "https://example.test/page",
    )
    final = _FakeStreamResponse(
        200, {"content-type": "text/html"}, "https://redirect-target.test/final", chunks=[b"final page"],
    )
    monkeypatch.setattr(
        fetch_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient({
            "https://example.test/page": redirect,
            "https://redirect-target.test/final": final,
        }),
    )

    doc = await safe_fetch("https://example.test/page", Config())

    assert doc.final_url == "https://redirect-target.test/final"
    assert doc.content == b"final page"


async def test_safe_fetch_rejects_a_redirect_to_an_unsafe_host(monkeypatch):
    redirect = _FakeStreamResponse(
        302, {"location": "http://internal.test/admin"}, "https://example.test/page",
    )
    monkeypatch.setattr(
        fetch_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient({"https://example.test/page": redirect}),
    )

    with pytest.raises(UnsafeURLError):
        await safe_fetch("https://example.test/page", Config())


async def test_safe_fetch_raises_after_exceeding_max_redirects(monkeypatch):
    responses = {}
    for i in range(10):
        responses[f"https://example.test/{i}"] = _FakeStreamResponse(
            302, {"location": f"https://example.test/{i + 1}"}, f"https://example.test/{i}",
        )
    monkeypatch.setattr(
        fetch_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient(responses),
    )

    with pytest.raises(UnsafeURLError, match="redirects"):
        await safe_fetch("https://example.test/0", Config())


async def test_safe_fetch_raises_when_streamed_content_exceeds_the_byte_cap(monkeypatch):
    response = _FakeStreamResponse(
        200, {"content-type": "text/plain"}, "https://example.test/big",
        chunks=[b"x" * 100, b"y" * 100],
    )
    monkeypatch.setattr(
        fetch_module.httpx, "AsyncClient",
        lambda **kwargs: _FakeClient({"https://example.test/big": response}),
    )
    config = Config()
    config.scraping.max_response_bytes = 150

    with pytest.raises(FetchTooLargeError):
        await safe_fetch("https://example.test/big", config)


async def test_safe_fetch_rejects_non_http_schemes():
    with pytest.raises(UnsafeURLError):
        await safe_fetch("file:///etc/passwd", Config())


async def test_safe_fetch_rejects_a_private_host_before_any_network_call(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("should never construct a client for an unsafe host")

    monkeypatch.setattr(fetch_module.httpx, "AsyncClient", fail_if_called)

    with pytest.raises(UnsafeURLError):
        await safe_fetch("http://127.0.0.1:8080/", Config())


class _HeaderCapturingClient:
    """Same shape as _FakeClient but records the headers each stream() call
    actually used, for the Wikimedia-vs-generic User-Agent tests below."""

    def __init__(self, response):
        self._response = response
        self.seen_headers: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, headers=None):
        self.seen_headers.append(headers or {})
        return _FakeStreamContext(self._response)


# -- Wikimedia's edge WAF 403s a generic browser-spoofing User-Agent on its
# article pages the same way it does its REST search API (see
# tools/search.py's _wikipedia_user_agent) -- confirmed live: identical
# request, only the User-Agent changed, 403 -> 200. safe_fetch() must send
# the compliant bot identity for the whole Wikimedia family, and the
# ordinary browser-spoofing default everywhere else.

async def test_safe_fetch_uses_the_compliant_wikipedia_user_agent(monkeypatch):
    response = _FakeStreamResponse(200, {"content-type": "text/html"}, "https://en.wikipedia.org/wiki/Example", chunks=[b"ok"])
    client = _HeaderCapturingClient(response)
    monkeypatch.setattr(fetch_module.httpx, "AsyncClient", lambda **kwargs: client)

    config = Config()
    config.wikipedia.contact = "me@example.com"
    await safe_fetch("https://en.wikipedia.org/wiki/Example", config)

    assert client.seen_headers[0]["User-Agent"] == "deep-research-kb-bot/1.0 (me@example.com) httpx"


async def test_safe_fetch_uses_the_generic_browser_user_agent_for_non_wikimedia_sites(monkeypatch):
    response = _FakeStreamResponse(200, {"content-type": "text/html"}, "https://example.test/page", chunks=[b"ok"])
    client = _HeaderCapturingClient(response)
    monkeypatch.setattr(fetch_module.httpx, "AsyncClient", lambda **kwargs: client)

    await safe_fetch("https://example.test/page", Config())

    assert "Mozilla" in client.seen_headers[0]["User-Agent"]


async def test_safe_fetch_lets_an_explicit_header_override_the_wikimedia_default(monkeypatch):
    response = _FakeStreamResponse(200, {"content-type": "text/html"}, "https://en.wikipedia.org/wiki/Example", chunks=[b"ok"])
    client = _HeaderCapturingClient(response)
    monkeypatch.setattr(fetch_module.httpx, "AsyncClient", lambda **kwargs: client)

    await safe_fetch("https://en.wikipedia.org/wiki/Example", Config(), headers={"User-Agent": "custom-agent/1.0"})

    assert client.seen_headers[0]["User-Agent"] == "custom-agent/1.0"
