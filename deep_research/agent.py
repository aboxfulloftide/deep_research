import json
import re
from urllib.parse import urlparse

from rich.console import Console

from deep_research.config import Config
from deep_research.db import Database
from deep_research.llm import LLMClient
from deep_research.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS, TOOL_DEFINITIONS
from deep_research.tools.scrape import scrape_and_extract, scrape_page
from deep_research.tools.search import web_search

console = Console()

# Pattern to detect URLs in user queries
_URL_RE = re.compile(r"https?://[^\s]+")


def _analyze_products(products: list[dict], query: str) -> str:
    """Analyze product data based on the query and return a focused summary.

    Does the sorting/filtering in code so the LLM just needs to compose an answer.
    """
    query_lower = query.lower()

    # Determine what the user is looking for
    analysis_lines = []

    # Products with RAM data
    with_ram = [p for p in products if p.get("ram_gb")]
    with_price = [p for p in products if p.get("price")]

    # Most RAM
    if with_ram:
        by_ram = sorted(with_ram, key=lambda x: x["ram_gb"], reverse=True)
        max_ram = by_ram[0]["ram_gb"]
        top_ram = [p for p in by_ram if p["ram_gb"] == max_ram]

        analysis_lines.append(f"## HIGHEST RAM: {max_ram}GB")
        for p in top_ram:
            analysis_lines.append(f"  - {p['name']} — {p.get('price', 'N/A')}")
            if p.get("url"):
                analysis_lines.append(f"    Link: {p['url']}")
        analysis_lines.append("")

    # Cheapest
    if with_price:
        def parse_price(p):
            m = re.search(r"[\d,]+\.?\d*", p.get("price", ""))
            return float(m.group().replace(",", "")) if m else 999999
        by_price = sorted(with_price, key=parse_price)
        cheapest = by_price[0]
        analysis_lines.append(f"## CHEAPEST: {cheapest['name']} — {cheapest.get('price', 'N/A')}")
        if cheapest.get("url"):
            analysis_lines.append(f"    Link: {cheapest['url']}")
        analysis_lines.append("")

    # Best value (most RAM per dollar)
    if with_ram and with_price:
        def ram_per_dollar(p):
            price_match = re.search(r"[\d,]+\.?\d*", p.get("price", ""))
            if not price_match:
                return 0
            price = float(price_match.group().replace(",", ""))
            return p.get("ram_gb", 0) / price if price > 0 else 0
        by_value = sorted(
            [p for p in products if p.get("ram_gb") and p.get("price")],
            key=ram_per_dollar, reverse=True
        )
        if by_value:
            best = by_value[0]
            analysis_lines.append(f"## BEST RAM/PRICE VALUE: {best['name']} — {best.get('price', 'N/A')} ({best.get('ram_gb', '?')}GB RAM)")
            if best.get("url"):
                analysis_lines.append(f"    Link: {best['url']}")
            analysis_lines.append("")

    # RAM breakdown
    if with_ram:
        ram_counts = {}
        for p in with_ram:
            ram = p["ram_gb"]
            ram_counts[ram] = ram_counts.get(ram, 0) + 1
        analysis_lines.append("## RAM BREAKDOWN:")
        for ram in sorted(ram_counts.keys(), reverse=True):
            analysis_lines.append(f"  {ram}GB: {ram_counts[ram]} laptops")
        analysis_lines.append("")

    return "\n".join(analysis_lines)


def _compact_product_list(products: list[dict]) -> str:
    """Create a compact one-line-per-product list."""
    lines = [f"ALL {len(products)} PRODUCTS:"]
    for i, p in enumerate(products, 1):
        parts = [p["name"]]
        if p.get("price"):
            parts.append(p["price"])
        if p.get("ram_gb"):
            parts.append(f"{p['ram_gb']}GB RAM")
        url = p.get("url", "")
        lines.append(f"{i}. {' | '.join(parts)}")
        if url:
            lines.append(f"   {url}")
    return "\n".join(lines)


class ResearchAgent:
    def __init__(self, config: Config, db: Database, llm: LLMClient):
        self.config = config
        self.db = db
        self.llm = llm

    async def run(self, query: str, session_id: str | None = None) -> str:
        """Run the agent loop for a query. Returns the final answer."""
        # Create or resume session
        if session_id:
            existing = await self.db.get_session(session_id)
            if not existing:
                raise ValueError(f"Session {session_id} not found")
        else:
            session_id = await self.db.create_session()

        await self.db.add_message(session_id, "user", content=query)

        # Auto-title from first query
        session = await self.db.get_session(session_id)
        if not session.get("title"):
            title = query[:80] + ("..." if len(query) > 80 else "")
            await self.db.update_session_title(session_id, title)

        # Pre-gather data from URLs in the query
        gathered_data, products = await self._pre_gather(query, session_id)

        # If we have products, do code-side analysis and send only that
        if products:
            analysis = _analyze_products(products, query)
            # Send analysis + compact product list (not the verbose scraped text)
            compact = _compact_product_list(products)
            gathered_data = f"{analysis}\n\n{compact}"

        # Check if this model supports tool calling
        if self.llm.supports_tools is None:
            await self._probe_tool_support()

        if self.llm.supports_tools and not gathered_data:
            # Only use tool loop if we didn't pre-gather (e.g., no URL in query)
            return await self._run_tool_loop(query, session_id, gathered_data)
        else:
            # Pre-gathered data or no tool support → text mode
            return await self._run_text_mode(query, session_id, gathered_data)

    async def _probe_tool_support(self):
        """Quick probe to check if the model supports tool calling."""
        try:
            await self.llm.chat(
                [{"role": "user", "content": "hi"}],
                tools=[TOOL_DEFINITIONS[0]],
            )
        except Exception:
            pass

    async def _pre_gather(self, query: str, session_id: str) -> tuple[str, list[dict]]:
        """Pre-scrape any URLs in the query. Returns (text_data, products_list)."""
        parts = []
        all_products = []

        urls = _URL_RE.findall(query)
        for url in urls:
            console.print(f"  [cyan]Scraping:[/cyan] {url[:80]}...")
            try:
                page = await scrape_page(url, self.config)
                await self.db.save_scraped_page(
                    session_id, page.url, page.title,
                    page.text_content, page.structured_data,
                )
                parts.append(f"=== Scraped: {page.title} ({url}) ===\n{page.text_content}")
                if page.structured_data and page.structured_data.get("products"):
                    all_products.extend(page.structured_data["products"])
            except Exception as e:
                parts.append(f"=== Error scraping {url}: {e} ===")

        return "\n\n".join(parts), all_products

    async def _run_text_mode(
        self, query: str, session_id: str, gathered_data: str
    ) -> str:
        """Pass gathered data to the LLM with a focused prompt."""
        if not gathered_data:
            console.print("[dim]No URL found, searching...[/dim]")
            search_query = query[:200]
            console.print(f"  [cyan]Searching:[/cyan] {search_query[:60]}...")
            try:
                results = await web_search(search_query, self.config)
                if results:
                    gathered_data = "\n".join(
                        f"- {r.title}: {r.url}\n  {r.snippet}" for r in results
                    )
            except Exception as e:
                gathered_data = f"(Search failed: {e})"

        # Extract the actual question from the query
        clean_query = _URL_RE.sub("", query).strip()
        for prefix in ("scrape", "scan", "check", "read", "look at", "go to"):
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):].strip()
        clean_query = clean_query.lstrip("and").strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS},
            {
                "role": "user",
                "content": (
                    f"DATA:\n\n{gathered_data}\n\n"
                    f"---\n\n"
                    f"QUESTION: {clean_query}\n\n"
                    f"Rules:\n"
                    f"1. Answer ONLY this question using the DATA above.\n"
                    f"2. Copy product URLs exactly from the data (do NOT invent URLs).\n"
                    f"3. Start your response with the answer immediately.\n"
                    f"4. Do NOT include any thinking, reasoning, or analysis process.\n"
                    f"5. Keep your response under 200 words."
                ),
            },
        ]

        console.print("\n[dim]Generating answer...[/dim]")
        resp = await self.llm.chat(messages)
        answer = resp["choices"][0]["message"].get("content", "No answer produced.")
        await self.db.add_message(session_id, "assistant", content=answer)
        return answer

    async def _run_tool_loop(
        self, query: str, session_id: str, gathered_data: str
    ) -> str:
        """For models with tool calling — run the standard agent loop."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Load any prior messages for session resumption
        prior = await self.db.get_session_messages(session_id)
        for m in prior:
            if m.get("role") not in ("system",):
                messages.append(m)

        messages.append({"role": "user", "content": query})

        # Agent loop
        for step in range(self.config.agent.max_steps):
            console.print(f"\n[dim]Step {step + 1}...[/dim]")

            resp = await self.llm.chat(messages, tools=TOOL_DEFINITIONS)
            choice = resp["choices"][0]
            msg = choice["message"]

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                answer = msg.get("content", "")
                await self.db.add_message(session_id, "assistant", content=answer)
                return answer

            assistant_msg = {"role": "assistant", "content": msg.get("content")}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            await self.db.add_message(
                session_id, "assistant",
                content=msg.get("content"),
                tool_calls=tool_calls,
            )

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                tool_call_id = tc.get("id", tool_name)
                console.print(f"  [cyan]→ {tool_name}[/cyan]({_format_args(tool_args)})")

                result = await self._execute_tool(tool_name, tool_args, session_id)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
                messages.append(tool_msg)
                await self.db.add_message(
                    session_id, "tool",
                    content=result,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )

                if tool_name == "finish":
                    return tool_args.get("answer", result)

        # Max steps reached
        console.print("[yellow]Max steps reached, summarizing...[/yellow]")
        messages.append({
            "role": "user",
            "content": "Provide your best answer now based on what you've gathered.",
        })
        resp = await self.llm.chat(messages)
        answer = resp["choices"][0]["message"].get("content", "No answer produced.")
        await self.db.add_message(session_id, "assistant", content=answer)
        return answer

    async def _execute_tool(
        self, name: str, args: dict, session_id: str
    ) -> str:
        """Execute a tool and return result as string."""
        try:
            if name == "web_search":
                results = await web_search(args["query"], self.config)
                if not results:
                    return "No search results found."
                lines = []
                for r in results:
                    lines.append(f"**{r.title}**\n{r.url}\n{r.snippet}\n")
                return "\n".join(lines)

            elif name == "scrape_webpage":
                url = args["url"]
                extract = args.get("extract")

                page = await scrape_page(url, self.config)

                if page.structured_data and page.structured_data.get("products"):
                    await self.db.save_scraped_page(
                        session_id, page.url, page.title,
                        page.text_content, page.structured_data,
                    )
                    return f"Title: {page.title}\n\n{page.text_content}"
                elif extract:
                    page = await scrape_and_extract(
                        url, extract, self.config, self.llm
                    )
                    await self.db.save_scraped_page(
                        session_id, page.url, page.title,
                        page.text_content, page.structured_data,
                    )
                    return json.dumps(page.structured_data, indent=2)
                else:
                    await self.db.save_scraped_page(
                        session_id, page.url, page.title, page.text_content,
                    )
                    return f"Title: {page.title}\n\n{page.text_content}"

            elif name == "finish":
                return args.get("answer", "")

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Error executing {name}: {e}"


def _format_args(args: dict) -> str:
    """Format tool args for display."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:60] + "..."
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)
