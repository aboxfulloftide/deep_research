import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from deep_research.config import load_config
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.db import KBDatabase
from deep_research.kb.ingest import ingest_file, ingest_web_page, ingest_youtube_video
from deep_research.kb.storage import SnapshotStore

console = Console()


def _kb_setup(args):
    config = load_config(args.config)
    kb_db = KBDatabase(config.kb_db_path)
    snapshot_store = SnapshotStore(config.kb_snapshot_dir)
    return config, kb_db, snapshot_store


def _print_result(result, label: str):
    if result.status == "failed":
        console.print(f"[red]Failed to ingest {label}: {result.error}[/red]")
        return
    verb = "Ingested new version of" if result.status == "ingested" else "No change to"
    console.print(f"[green]{verb}[/green] {label}")
    console.print(f"  source_id:  {result.source_id} ({'new' if result.source_created else 'existing'})")
    if result.version_id:
        console.print(f"  version_id: {result.version_id} ({'new' if result.version_created else 'unchanged'})")
    if result.pruned_version_ids:
        console.print(f"  pruned {len(result.pruned_version_ids)} old version(s) per retention policy")


async def cmd_ingest_url(args):
    config, kb_db, snapshot_store = _kb_setup(args)
    await kb_db.init()
    result = await ingest_web_page(args.url, config, kb_db, snapshot_store, trust_tier_code=args.trust_tier)
    _print_result(result, args.url)


async def cmd_ingest_youtube(args):
    config, kb_db, snapshot_store = _kb_setup(args)
    await kb_db.init()
    result = await ingest_youtube_video(args.url, kb_db, snapshot_store, trust_tier_code=args.trust_tier)
    _print_result(result, args.url)


async def cmd_ingest_file(args):
    config, kb_db, snapshot_store = _kb_setup(args)
    await kb_db.init()
    result = await ingest_file(args.path, kb_db, snapshot_store, trust_tier_code=args.trust_tier)
    _print_result(result, args.path)


async def cmd_list_sources(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    sources = await kb_db.list_sources(limit=args.limit)
    if not sources:
        console.print("[dim]No sources ingested yet.[/dim]")
        return

    table = Table(title="Knowledge Base Sources")
    table.add_column("ID", style="cyan", max_width=10)
    table.add_column("Type", style="magenta")
    table.add_column("Title", style="white")
    table.add_column("Trust", style="yellow")
    table.add_column("Updated", style="dim")

    for s in sources:
        table.add_row(
            s["id"][:8] + "...",
            s["source_type_code"],
            s.get("title") or "(untitled)",
            s.get("trust_tier_code") or "-",
            s["updated_at"][:19],
        )
    console.print(table)


async def _resolve_source(kb_db, source_id_prefix: str) -> dict | None:
    sources = await kb_db.list_sources(limit=1000)
    return next((s for s in sources if s["id"].startswith(source_id_prefix)), None)


async def cmd_show_source(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    match = await _resolve_source(kb_db, args.source_id)
    if match is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return

    console.print(f"[bold]{match.get('title') or '(untitled)'}[/bold]")
    console.print(f"  id:            {match['id']}")
    console.print(f"  type:          {match['source_type_code']}")
    console.print(f"  canonical_uri: {match['canonical_uri']}")
    console.print(f"  trust_tier:    {match.get('trust_tier_code') or '(none)'}")

    versions = await kb_db.list_versions(match["id"])
    table = Table(title="Versions")
    table.add_column("#")
    table.add_column("Captured")
    table.add_column("First")
    table.add_column("Latest")
    table.add_column("Locked")
    table.add_column("Bytes")
    for v in versions:
        table.add_row(
            str(v["version_number"]), v["captured_at"][:19],
            "yes" if v["is_first_version"] else "",
            "yes" if v["is_latest"] else "",
            "yes" if v["retention_locked"] else "",
            str(v["byte_size"] or ""),
        )
    console.print(table)

    attempts = await kb_db.list_fetch_attempts(match["id"])
    if attempts:
        table = Table(title="Fetch Attempts")
        table.add_column("When")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Error")
        for a in attempts[:10]:
            table.add_row(
                a["created_at"][:19], a["attempt_type"], a["status"],
                a.get("error_message") or "",
            )
        console.print(table)


async def cmd_chunk_source(args):
    config, kb_db, snapshot_store = _kb_setup(args)
    await kb_db.init()

    match = await _resolve_source(kb_db, args.source_id)
    if match is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return

    version = await kb_db.get_latest_version(match["id"])
    if version is None:
        console.print(f"[red]No ingested version found for source {match['id']}[/red]")
        return

    result = await build_artifact_for_version(kb_db, snapshot_store, match, version, chunk_size=args.chunk_size)

    verb = {
        "chunked": "[green]Chunked[/green]",
        "unchanged": "[yellow]Already chunked (unchanged)[/yellow]",
        "empty": "[yellow]No extractable text found[/yellow]",
    }[result.status]
    console.print(f"{verb} {match.get('title') or match['id']}")
    console.print(f"  artifact_id: {result.artifact_id} ({'new generation' if result.artifact_created else 'existing'})")
    console.print(f"  chunks:      {result.chunk_count}")


async def cmd_search(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    results = await kb_db.search_chunks(args.query, limit=args.limit)
    if not results:
        console.print("[dim]No matching chunks.[/dim]")
        return

    for r in results:
        console.print(f"\n[bold]{r['source_title'] or r['canonical_uri']}[/bold]  [dim]({r['artifact_type']})[/dim]")
        location = f"chunk {r['chunk_index']}"
        if r["page_number"] is not None:
            location += f", page {r['page_number']}"
        if r["time_start_seconds"] is not None:
            location += f", t={r['time_start_seconds']:.0f}s"
        console.print(f"  [dim]{location}[/dim]")
        console.print(f"  {r['snippet']}")


def main():
    parser = argparse.ArgumentParser(description="Deep Research — knowledge base source ingestion")
    parser.add_argument("--config", "-c", help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_url = subparsers.add_parser("ingest-url", help="Ingest a web page")
    p_url.add_argument("url")
    p_url.add_argument("--trust-tier", help="official|reputable_reporting|secondary_analysis|user_generated")
    p_url.set_defaults(func=cmd_ingest_url)

    p_yt = subparsers.add_parser("ingest-youtube", help="Ingest a YouTube video transcript")
    p_yt.add_argument("url", help="YouTube URL or bare video ID")
    p_yt.add_argument("--trust-tier")
    p_yt.set_defaults(func=cmd_ingest_youtube)

    p_file = subparsers.add_parser("ingest-file", help="Ingest a local file (PDF, Markdown, text, HTML, docx)")
    p_file.add_argument("path")
    p_file.add_argument("--trust-tier")
    p_file.set_defaults(func=cmd_ingest_file)

    p_list = subparsers.add_parser("list-sources", help="List ingested sources")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list_sources)

    p_show = subparsers.add_parser("show-source", help="Show a source's versions and fetch history")
    p_show.add_argument("source_id", help="Source ID or prefix")
    p_show.set_defaults(func=cmd_show_source)

    p_chunk = subparsers.add_parser("chunk-source", help="Chunk the latest version of a source")
    p_chunk.add_argument("source_id", help="Source ID or prefix")
    p_chunk.add_argument("--chunk-size", type=int, default=1200)
    p_chunk.set_defaults(func=cmd_chunk_source)

    p_search = subparsers.add_parser("search", help="Full-text search over chunked content")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
