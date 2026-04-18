import argparse
import asyncio
import sys

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from deep_research.agent import ResearchAgent
from deep_research.config import load_config
from deep_research.db import Database
from deep_research.llm import LLMClient

console = Console()


async def fetch_models(base_url: str) -> list[str]:
    """Fetch available models from the Ollama server."""
    # Ollama exposes /api/tags for model list
    ollama_url = base_url.replace("/v1", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        console.print(f"[yellow]Could not fetch models: {e}[/yellow]")
        return []


def prompt_model_selection(models: list[str], default: str) -> str:
    """Let the user pick a model from the available list."""
    console.print("\n[bold]Available models:[/bold]")
    for i, name in enumerate(models, 1):
        marker = " [green](default)[/green]" if name == default else ""
        console.print(f"  {i}. {name}{marker}")

    console.print()
    choice = console.input(
        f"[bold]Select model [1-{len(models)}] or Enter for default ({default}): [/bold]"
    ).strip()

    if not choice:
        return default

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        # Treat as a model name typed directly
        if choice in models:
            return choice

    console.print(f"[yellow]Invalid selection, using default: {default}[/yellow]")
    return default


async def setup_config(args):
    """Load config and optionally let user select a model."""
    config = load_config(args.config)

    model = getattr(args, "model", None)
    if model:
        # Explicit --model flag
        config.llm.model = model
    elif not getattr(args, "no_select", False):
        # Fetch and prompt for model selection
        models = await fetch_models(config.llm.base_url)
        if models:
            selected = prompt_model_selection(models, config.llm.model)
            config.llm.model = selected
            console.print(f"[bold]Using model:[/bold] {selected}")
        else:
            console.print(f"[bold]Using model:[/bold] {config.llm.model}")

    return config


async def run_query(args):
    config = await setup_config(args)
    db = Database(config.db_path)
    await db.init()
    llm = LLMClient(config)

    agent = ResearchAgent(config, db, llm)

    try:
        answer = await agent.run(args.query, session_id=args.session)
        console.print()
        console.print(Markdown(answer))
    finally:
        await llm.close()


async def list_sessions(args):
    config = load_config(args.config)
    db = Database(config.db_path)
    await db.init()

    sessions = await db.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions")
    table.add_column("ID", style="cyan", max_width=36)
    table.add_column("Title", style="white")
    table.add_column("Updated", style="dim")

    for s in sessions:
        table.add_row(
            s["id"][:8] + "...",
            s.get("title") or "(untitled)",
            s.get("updated_at", "")[:19],
        )
    console.print(table)


async def interactive_session(args):
    """Interactive mode — keep asking questions in the same session."""
    config = await setup_config(args)
    db = Database(config.db_path)
    await db.init()
    llm = LLMClient(config)

    agent = ResearchAgent(config, db, llm)
    session_id = args.session

    try:
        console.print("[bold]Deep Research[/bold] — interactive mode (Ctrl+C to exit)\n")

        # If a query was provided, run it first
        if args.query:
            answer = await agent.run(args.query, session_id=session_id)
            console.print()
            console.print(Markdown(answer))
            # Use the session from the first run for follow-ups
            if not session_id:
                sessions = await db.list_sessions(limit=1)
                if sessions:
                    session_id = sessions[0]["id"]

        while True:
            console.print()
            try:
                query = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            query = query.strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break

            answer = await agent.run(query, session_id=session_id)
            console.print()
            console.print(Markdown(answer))

            if not session_id:
                sessions = await db.list_sessions(limit=1)
                if sessions:
                    session_id = sessions[0]["id"]
    finally:
        await llm.close()


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research — local LLM-powered research tool"
    )
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument(
        "--model", "-m", help="Model to use (skips selection prompt)"
    )
    parser.add_argument(
        "--session", "-s", help="Resume a session by ID (prefix match supported)"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List past sessions")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode — keep asking questions",
    )
    parser.add_argument("--config", "-c", help="Path to config.yaml")

    args = parser.parse_args()

    if args.list:
        args.no_select = True
        asyncio.run(list_sessions(args))
    elif args.interactive or (not args.query and not args.list):
        asyncio.run(interactive_session(args))
    elif args.query:
        asyncio.run(run_query(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
