import json
import re

import httpx
from bs4 import BeautifulSoup, Tag

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


def _extract_products(html: str) -> list[dict] | None:
    """Try to extract structured product data from HTML.

    Looks for common e-commerce patterns: product cards with names and prices.
    Returns None if no products detected.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove junk
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Strategy 1: Look for JSON-LD structured data
    for script in BeautifulSoup(html, "lxml").find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                products = [d for d in data if d.get("@type") in ("Product", "Offer")]
                if products:
                    return [{"name": p.get("name", ""), "price": p.get("offers", {}).get("price", p.get("price", "")), "url": p.get("url", "")} for p in products]
            elif isinstance(data, dict) and data.get("@type") == "ItemList":
                items = data.get("itemListElement", [])
                if items:
                    return [{"name": i.get("name", ""), "url": i.get("url", "")} for i in items]
        except (json.JSONDecodeError, AttributeError):
            continue

    # Strategy 2: Find product card patterns
    # Look for repeating elements with product-related classes
    product_containers = []
    for selector in [
        "[class*=product-card]", "[class*=product-item]", "[class*=product_card]",
        "[class*=ProductCard]", "[class*=grid-product]", "[class*=card--product]",
        "[class*=product-grid-item]",
    ]:
        found = soup.select(selector)
        if len(found) > 2:
            product_containers = found
            break

    if not product_containers:
        # Broader search: look for grid items that contain prices
        price_pattern = re.compile(r"\$\s*[\d,]+\.?\d*")
        candidates = soup.find_all(class_=re.compile(r"card|item|product|grid"))
        product_containers = [
            c for c in candidates
            if isinstance(c, Tag) and price_pattern.search(c.get_text())
            and len(c.get_text(strip=True)) < 1000  # Skip huge containers
        ]

    if len(product_containers) < 2:
        return None

    products = []
    price_re = re.compile(r"\$\s*([\d,]+\.?\d*)")

    for container in product_containers:
        # Extract name from headings or product title links
        name = ""
        for tag in container.find_all(["h2", "h3", "h4", "a"]):
            text = tag.get_text(strip=True)
            if len(text) > 10 and not text.startswith("$"):
                name = text
                break

        if not name:
            # Fallback: longest text that looks like a product name
            texts = [t.strip() for t in container.stripped_strings if len(t.strip()) > 10 and not t.startswith("$")]
            if texts:
                name = max(texts, key=len)

        if not name:
            continue

        # Extract price
        price_match = price_re.search(container.get_text())
        price = price_match.group(0) if price_match else ""

        # Extract link
        link = ""
        a_tag = container.find("a", href=True)
        if a_tag:
            href = a_tag["href"]
            if href.startswith("/"):
                link = href
            elif href.startswith("http"):
                link = href

        # Get all text for extra specs
        full_text = container.get_text(separator=" ", strip=True)
        # Clean up repeated whitespace
        full_text = re.sub(r"\s+", " ", full_text)

        # Extract specs from name/text
        specs = {}
        combined = name + " " + full_text
        ram_match = re.search(r"(\d+)\s*GB\s*(?:RAM|DDR)", combined, re.IGNORECASE)
        if ram_match:
            specs["ram_gb"] = int(ram_match.group(1))
        storage_match = re.search(r"(\d+)\s*(?:GB|TB)\s*(?:SSD|HDD|NVMe)", combined, re.IGNORECASE)
        if not storage_match:
            # Try pattern like "512GB" or "1TB" not followed by RAM
            storage_match = re.search(r"(\d+\s*TB)\b", combined, re.IGNORECASE)
        if storage_match:
            specs["storage"] = storage_match.group(0).strip()

        product = {"name": name, "price": price}
        if link:
            product["url"] = link
        if specs:
            product.update(specs)
        if len(full_text) > len(name) + len(price) + 10:
            product["details"] = full_text

        # Deduplicate by name
        if not any(p["name"] == name for p in products):
            products.append(product)

    return products if products else None


async def scrape_page(url: str, config: Config) -> ScrapedPage:
    """Fetch a URL and extract its text content."""
    async with httpx.AsyncClient(
        timeout=config.scraping.timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Try structured product extraction first
    products = _extract_products(html)
    if products:
        title = BeautifulSoup(html, "lxml").title
        title_text = title.string.strip() if title and title.string else ""

        # Build base URL for resolving relative links
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Resolve relative URLs to absolute
        for p in products:
            if p.get("url") and p["url"].startswith("/"):
                p["url"] = base_url + p["url"]

        # Clean product URLs — strip query params for cleaner links
        for p in products:
            if p.get("url") and "?" in p["url"]:
                p["url"] = p["url"].split("?")[0]

        # Build compact product listing — one line per product
        product_text = f"Found {len(products)} products:\n\n"
        for i, p in enumerate(products, 1):
            specs = []
            if p.get("price"):
                specs.append(p["price"])
            if p.get("ram_gb"):
                specs.append(f"{p['ram_gb']}GB RAM")
            if p.get("storage"):
                specs.append(p["storage"])
            specs_str = " | ".join(specs)
            product_text += f"{i}. {p['name']} — {specs_str}\n"
            if p.get("url"):
                product_text += f"   {p['url']}\n"
            product_text += "\n"
        return ScrapedPage(
            url=url,
            title=title_text,
            text_content=product_text,
            structured_data={"products": products},
        )

    # Fallback to plain text extraction
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
