import json
import re

import httpx
from bs4 import BeautifulSoup

from deep_research.config import Config
from deep_research.llm import LLMClient
from deep_research.models import ScrapedPage

REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}


def _extract_text(html: str) -> tuple[str, str]:
    """Extract clean text from HTML. Returns (title, text_content)."""
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Remove junk tags
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Prefer main/article content
    main = soup.find("main") or soup.find("article")
    target = main if main else soup.body if soup.body else soup

    text = target.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


async def scrape_page(url: str, config: Config) -> ScrapedPage:
    """Fetch a URL and extract its text content."""
    async with httpx.AsyncClient(
        timeout=config.scraping.timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/0.1)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    title, text = _extract_text(html)

    # Truncate if too long
    max_len = config.scraping.max_content_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[Content truncated]"

    return ScrapedPage(url=url, title=title, text_content=text)


async def scrape_and_extract(
    url: str,
    extraction_prompt: str,
    config: Config,
    llm: LLMClient,
) -> ScrapedPage:
    """Scrape a page and use the LLM to extract structured data."""
    page = await scrape_page(url, config)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data extraction assistant. Extract the requested information "
                "from the provided webpage content. Return valid JSON only, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract the following from this webpage: {extraction_prompt}\n\n"
                f"Page title: {page.title}\n"
                f"Page content:\n{page.text_content}"
            ),
        },
    ]

    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"]

    # Try to parse as JSON
    try:
        structured = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if match:
            try:
                structured = json.loads(match.group(1))
            except json.JSONDecodeError:
                structured = {"raw_extraction": content}
        else:
            structured = {"raw_extraction": content}

    page.structured_data = structured
    return page
