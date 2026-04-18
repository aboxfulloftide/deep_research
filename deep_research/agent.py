import json

from rich.console import Console

from deep_research.config import Config
from deep_research.db import Database
from deep_research.llm import LLMClient
from deep_research.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
from deep_research.tools.scrape import scrape_and_extract, scrape_page
from deep_research.tools.search import web_search

console = Console()


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
            messages = await self.db.get_session_messages(session_id)
            # Ensure system prompt is first
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        else:
            session_id = await self.db.create_session()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add user message
        user_msg = {"role": "user", "content": query}
        messages.append(user_msg)
        await self.db.add_message(session_id, "user", content=query)

        # Auto-title from first query
        session = await self.db.get_session(session_id)
        if not session.get("title"):
            title = query[:80] + ("..." if len(query) > 80 else "")
            await self.db.update_session_title(session_id, title)

        # Agent loop
        for step in range(self.config.agent.max_steps):
            console.print(f"\n[dim]Step {step + 1}...[/dim]")

            resp = await self.llm.chat(messages, tools=TOOL_DEFINITIONS)
            choice = resp["choices"][0]
            msg = choice["message"]

            # Check for tool calls
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # Plain text response — treat as final answer
                answer = msg.get("content", "")
                await self.db.add_message(session_id, "assistant", content=answer)
                return answer

            # Process tool calls
            # Add assistant message with tool calls to history
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

                # Add tool result to messages
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

                # If finish was called, return the answer
                if tool_name == "finish":
                    return tool_args.get("answer", result)

        # Max steps reached
        console.print("[yellow]Max steps reached, summarizing...[/yellow]")
        messages.append({
            "role": "user",
            "content": "You've reached the maximum number of steps. Please provide your best answer now based on what you've gathered so far.",
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
                if extract:
                    page = await scrape_and_extract(
                        url, extract, self.config, self.llm
                    )
                    await self.db.save_scraped_page(
                        session_id, page.url, page.title,
                        page.text_content, page.structured_data,
                    )
                    return json.dumps(page.structured_data, indent=2)
                else:
                    page = await scrape_page(url, self.config)
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
