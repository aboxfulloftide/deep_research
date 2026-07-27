"""Shared safe-fetch primitive for anything that retrieves a URL supplied by
a search result or a user, rather than a fixed/deployment-configured
endpoint. `tools/scrape.py`'s `scrape_page()` and `kb/ingest.py`'s
`ingest_web_page()` are the only two such call sites in the codebase
(confirmed by grepping every httpx call site) and don't currently share any
protection: no SSRF validation, no byte cap, no redirect-target
revalidation. httpx's own `follow_redirects=True` does neither of the
latter two, so redirects are followed manually here instead.
"""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from deep_research.config import Config

_ALLOWED_SCHEMES = {"http", "https"}
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Wikimedia's edge WAF 403s a generic browser-spoofing User-Agent on its
# article/file pages the same way it does its REST search API (see
# tools/search.py's _wikipedia_user_agent) -- confirmed live: identical
# request, only the User-Agent changed, 403 -> 200. Every other site gets
# the browser-spoofing default above, which is what most non-Wikimedia
# sites actually expect from an anonymous fetch.
_WIKIMEDIA_DOMAIN_SUFFIXES = (
    "wikipedia.org", "wikimedia.org", "wiktionary.org", "wikidata.org",
    "wikibooks.org", "wikiquote.org", "wikisource.org", "wikinews.org",
    "wikiversity.org", "wikivoyage.org",
)


def _is_wikimedia_domain(hostname: str) -> bool:
    return hostname.lower().endswith(_WIKIMEDIA_DOMAIN_SUFFIXES)


def _default_user_agent(url: str, config: Config) -> str:
    hostname = urlparse(url).hostname or ""
    if _is_wikimedia_domain(hostname):
        contact = config.wikipedia.contact or "no contact configured"
        return f"deep-research-kb-bot/1.0 ({contact}) httpx"
    return _USER_AGENT


class UnsafeURLError(Exception):
    """A URL (the original request or a redirect hop) uses a disallowed
    scheme, or its host resolves to a private/loopback/link-local/reserved
    address."""


class FetchTooLargeError(Exception):
    """The response body exceeded config.scraping.max_response_bytes."""


@dataclass
class FetchedDocument:
    url: str          # originally requested URL
    final_url: str    # URL after following redirects
    status_code: int
    mime_type: str
    content: bytes


def _is_safe_host(hostname: str) -> bool:
    """Resolve the hostname and reject if ANY resolved address is private,
    loopback, link-local, reserved, multicast, or unspecified. Checking the
    literal hostname string alone would miss DNS rebinding -- a
    normal-looking name that resolves to an internal address."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        raw_address = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_address)
        except ValueError:
            return False
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            return False
    return True


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Disallowed URL scheme: {parsed.scheme!r}")
    if not parsed.hostname or not _is_safe_host(parsed.hostname):
        raise UnsafeURLError(f"Disallowed or unresolvable host: {parsed.hostname!r}")


async def safe_fetch(
    url: str, config: Config, *, headers: dict | None = None,
) -> FetchedDocument:
    """Fetch url, validating every redirect hop against the same SSRF rule
    and capping response bytes while streaming (not after fully buffering).
    Raises UnsafeURLError, FetchTooLargeError, httpx.HTTPStatusError, or
    another httpx.HTTPError on failure -- callers keep their existing
    httpx-exception handling and add the two new types."""
    max_bytes = config.scraping.max_response_bytes
    max_redirects = config.scraping.max_redirects
    request_headers = {"User-Agent": _default_user_agent(url, config), **(headers or {})}
    current_url = url
    _validate_url(current_url)  # fail fast before opening any connection

    async with httpx.AsyncClient(timeout=config.scraping.timeout) as client:
        for _ in range(max_redirects + 1):
            async with client.stream("GET", current_url, headers=request_headers) as resp:
                if 300 <= resp.status_code < 400 and "location" in resp.headers:
                    current_url = urljoin(current_url, resp.headers["location"])
                    _validate_url(current_url)
                    continue
                resp.raise_for_status()
                content = bytearray()
                async for chunk in resp.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        raise FetchTooLargeError(
                            f"Response from {current_url} exceeded {max_bytes} byte cap"
                        )
                mime_type = resp.headers.get("content-type", "").split(";")[0].strip()
                return FetchedDocument(
                    url=url, final_url=str(resp.url), status_code=resp.status_code,
                    mime_type=mime_type, content=bytes(content),
                )
    raise UnsafeURLError(f"Too many redirects (> {max_redirects}) fetching {url}")
