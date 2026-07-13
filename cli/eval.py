import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import yaml
from rich.console import Console
from rich.table import Table

from deep_research.config import load_config
from deep_research.evals import registry
from deep_research.evals.report import compute_stats_for_source
from deep_research.evals.server import start_server, stop_server

console = Console()


def _admin_dsn(dsn: str, admin_db: str = "postgres") -> str:
    """Same host/port/user/password as `dsn`, pointed at the maintenance DB
    -- used for issuing CREATE DATABASE, which can't run inside a
    transaction/against the DB being created."""
    parts = urlsplit(dsn)
    return urlunsplit((parts.scheme, parts.netloc, f"/{admin_db}", parts.query, parts.fragment))


def _db_name(dsn: str) -> str:
    return urlsplit(dsn).path.lstrip("/")


def _eval_dir(*parts: str) -> Path:
    path = Path.cwd() / "evals"
    for p in parts:
        path = path / p
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def cmd_register_model(args):
    config = load_config()
    slug = args.slug

    if args.existing_db:
        postgres_dsn = args.existing_db
        console.print(f"Using existing database (dsn provided) for {slug!r}")
    else:
        postgres_dsn = f"postgresql://deep_research:deep_research@localhost:5432/deep_research_eval_{slug}"
        admin_conn = await asyncpg.connect(_admin_dsn(postgres_dsn))
        try:
            await admin_conn.execute(f'CREATE DATABASE "{_db_name(postgres_dsn)}"')
            console.print(f"[green]Created database[/green] {_db_name(postgres_dsn)}")
        except asyncpg.exceptions.DuplicateDatabaseError:
            console.print(f"[yellow]Database {_db_name(postgres_dsn)} already exists, reusing[/yellow]")
        finally:
            await admin_conn.close()

    snapshot_dir = args.existing_snapshot_dir or str(Path.home() / ".local/share/deep_research" / f"kb_eval_{slug}")
    Path(snapshot_dir).expanduser().mkdir(parents=True, exist_ok=True)

    server_args = {
        "llama_server_bin": args.llama_server_bin,
        "gpu_layers": args.gpu_layers,
        "tensor_split": args.tensor_split,
        "devices": args.devices,
        "split_mode": args.split_mode,
        "parallel": args.parallel,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "flash_attn": not args.no_flash_attn,
    }

    config_path = _eval_dir("configs", f"{slug}.yaml")
    config_yaml = {
        "kb": {
            "postgres_dsn": postgres_dsn,
            "snapshot_dir": snapshot_dir,
            "extraction_llm_base_url": f"http://127.0.0.1:{args.port}/v1",
            "extraction_llm_model": "",
            "embedding_base_url": config.kb.embedding_base_url,
            "embedding_model": config.kb.embedding_model,
            "claim_duplicate_threshold": config.kb.claim_duplicate_threshold,
            "verification_max_web_searches": config.kb.verification_max_web_searches,
            "verification_max_sources_examined": config.kb.verification_max_sources_examined,
            "verification_importance_threshold": config.kb.verification_importance_threshold,
            "verification_max_chunks_per_page": config.kb.verification_max_chunks_per_page,
            "verification_concurrency": config.kb.verification_concurrency,
        },
    }
    config_path.write_text(yaml.safe_dump(config_yaml, sort_keys=False))

    await registry.register_model(
        config, slug, model_path=args.model_path, port=args.port,
        server_args_json=json.dumps(server_args), postgres_dsn=postgres_dsn,
        snapshot_dir=snapshot_dir, config_path=str(config_path), display_name=args.display_name,
    )

    console.print(f"[green]Registered model[/green] {slug!r}")
    console.print(f"  postgres_dsn:  {postgres_dsn}")
    console.print(f"  snapshot_dir:  {snapshot_dir}")
    console.print(f"  config:        {config_path}")
    console.print(f"  port:          {args.port}")


async def cmd_list_models(args):
    config = load_config()
    models = await registry.list_models(config)
    if not models:
        console.print("[dim]No models registered yet.[/dim]")
        return
    table = Table(title="Eval Models")
    table.add_column("Slug", style="cyan")
    table.add_column("Display Name")
    table.add_column("Port")
    table.add_column("Postgres DSN", style="dim")
    table.add_column("Config")
    for m in models:
        table.add_row(m["slug"], m["display_name"], str(m["port"]), m["postgres_dsn"], m["config_path"])
    console.print(table)


async def cmd_start_server(args):
    config = load_config()
    model = await registry.get_model(config, args.slug)
    if model is None:
        console.print(f"[red]No registered model {args.slug!r} — run register-model first[/red]")
        return
    console.print(f"Starting {model['display_name']} on port {model['port']}...")
    ready, log_path = await start_server(model)
    if ready:
        console.print(f"[green]Server ready[/green] (log: {log_path})")
    else:
        console.print(f"[red]Server did not become healthy in time — check {log_path}[/red]")


async def cmd_stop_server(args):
    config = load_config()
    model = await registry.get_model(config, args.slug)
    if model is None:
        console.print(f"[red]No registered model {args.slug!r}[/red]")
        return
    console.print(f"Stopping {model['display_name']}...")
    stopped = await stop_server(model)
    console.print("[green]Stopped[/green]" if stopped else "[red]Did not stop in time[/red]")


async def cmd_add_source(args):
    config = load_config()
    source = await registry.add_source(config, args.slug, url=args.url, title=args.title)
    console.print(f"[green]Registered source[/green] {source['slug']!r} -> {source['url']}")


async def cmd_list_sources(args):
    config = load_config()
    sources = await registry.list_sources(config)
    if not sources:
        console.print("[dim]No sources registered yet.[/dim]")
        return
    table = Table(title="Eval Sources")
    table.add_column("Slug", style="cyan")
    table.add_column("URL")
    table.add_column("Title")
    for s in sources:
        table.add_row(s["slug"], s["url"], s.get("title") or "")
    console.print(table)


async def cmd_report(args):
    config = load_config()
    models = await registry.list_models(config)
    if not models:
        console.print("[dim]No models registered yet.[/dim]")
        return

    if args.source:
        source = await registry.get_source(config, args.source)
        if source is None:
            console.print(f"[red]No registered source {args.source!r}[/red]")
            return
        sources = [source]
    else:
        sources = await registry.list_sources(config)
        if not sources:
            console.print("[dim]No sources registered yet.[/dim]")
            return

    for source in sources:
        table = Table(title=f"{source['slug']} — {source.get('title') or source['url']}")
        table.add_column("Model", style="cyan")
        table.add_column("Total")
        table.add_column("Eligible (>=0.8)")
        table.add_column("Supported")
        table.add_column("Contradicted")
        table.add_column("Mixed")
        table.add_column("Unverified")
        table.add_column("Resolution %")
        for model in models:
            stats = await compute_stats_for_source(
                model["postgres_dsn"], model["display_name"], model["slug"], source["url"],
                config.kb.verification_importance_threshold,
            )
            if not stats.found:
                table.add_row(model["display_name"], "[dim]not ingested[/dim]", "", "", "", "", "", "")
                continue
            rate = f"{stats.resolution_rate * 100:.1f}%" if stats.resolution_rate is not None else "-"
            table.add_row(
                model["display_name"], str(stats.total_claims), str(stats.eligible_claims),
                str(stats.status_counts.get("supported", 0)), str(stats.status_counts.get("contradicted", 0)),
                str(stats.status_counts.get("mixed", 0)), str(stats.status_counts.get("unverified", 0)), rate,
            )
        console.print(table)


async def cmd_backup(args):
    config = load_config()
    model = await registry.get_model(config, args.slug)
    if model is None:
        console.print(f"[red]No registered model {args.slug!r}[/red]")
        return
    db_name = _db_name(model["postgres_dsn"])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    container_path = f"/tmp/{db_name}-{ts}.dump"
    dest = _eval_dir("backups", f"{args.slug}-{ts}.dump")

    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "deep-research-postgres", "pg_dump", "-U", "deep_research",
        "-d", db_name, "-F", "c", "-f", container_path,
    )
    rc = await proc.wait()
    if rc != 0:
        console.print("[red]pg_dump failed[/red]")
        return

    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", f"deep-research-postgres:{container_path}", str(dest),
    )
    await proc.wait()
    console.print(f"[green]Backed up[/green] {db_name} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Deep Research — multi-model eval infrastructure")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_reg = subparsers.add_parser("register-model", help="Provision a DB/config for a model under test")
    p_reg.add_argument("slug", help="Short identifier, e.g. qwen3-14b")
    p_reg.add_argument("--model-path", required=True, help="Path to the .gguf (or model identifier)")
    p_reg.add_argument("--display-name")
    p_reg.add_argument("--port", type=int, default=18080)
    p_reg.add_argument("--llama-server-bin", default="llama-server")
    p_reg.add_argument("--gpu-layers", type=int, default=99)
    p_reg.add_argument("--tensor-split", default="1,1")
    p_reg.add_argument("--devices", default="CUDA0,CUDA1")
    p_reg.add_argument("--split-mode", default="layer")
    p_reg.add_argument("--parallel", type=int, default=2)
    p_reg.add_argument("--context", type=int, default=32768)
    p_reg.add_argument("--batch", type=int, default=4096)
    p_reg.add_argument("--ubatch", type=int, default=512)
    p_reg.add_argument("--no-flash-attn", action="store_true")
    p_reg.add_argument("--existing-db", help="Postgres DSN of an already-provisioned DB to adopt instead of creating one")
    p_reg.add_argument("--existing-snapshot-dir", help="Snapshot dir of an already-provisioned KB to adopt")
    p_reg.set_defaults(func=cmd_register_model)

    p_list = subparsers.add_parser("list-models", help="List registered models")
    p_list.set_defaults(func=cmd_list_models)

    p_start = subparsers.add_parser("start-server", help="Start a registered model's llama-server")
    p_start.add_argument("slug")
    p_start.set_defaults(func=cmd_start_server)

    p_stop = subparsers.add_parser("stop-server", help="Stop a registered model's llama-server")
    p_stop.add_argument("slug")
    p_stop.set_defaults(func=cmd_stop_server)

    p_addsrc = subparsers.add_parser("add-source", help="Register a canonical test source")
    p_addsrc.add_argument("slug", help="Short identifier, e.g. 130years")
    p_addsrc.add_argument("--url", required=True)
    p_addsrc.add_argument("--title")
    p_addsrc.set_defaults(func=cmd_add_source)

    p_listsrc = subparsers.add_parser("list-sources", help="List registered test sources")
    p_listsrc.set_defaults(func=cmd_list_sources)

    p_report = subparsers.add_parser("report", help="Cross-model comparison table")
    p_report.add_argument("--source", help="Limit to one registered source slug")
    p_report.set_defaults(func=cmd_report)

    p_backup = subparsers.add_parser("backup", help="pg_dump a registered model's database")
    p_backup.add_argument("slug")
    p_backup.set_defaults(func=cmd_backup)

    args = parser.parse_args()
    try:
        asyncio.run(args.func(args))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
