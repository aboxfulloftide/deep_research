import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from deep_research.config import load_config
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import run_extraction
from deep_research.kb.ingest import ingest_file, ingest_web_page, ingest_youtube_video
from deep_research.kb.reports import generate_topic_report
from deep_research.kb.resolution import resolve_and_promote
from deep_research.kb.storage import SnapshotStore
from deep_research.kb.timeline import get_topic_timeline
from deep_research.kb.topics import check_claims_against_topics, generate_topic_suggestions
from deep_research.kb.verification import verify_claim

console = Console()


def _fmt_ts(value) -> str:
    """Postgres returns real datetime objects (not the ISO strings SQLite
    stored), so format explicitly instead of the old string[:19] slicing."""
    if value is None:
        return ""
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _kb_setup(args):
    config = load_config(args.config)
    kb_db = KBDatabase(config.kb.postgres_dsn)
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
            _fmt_ts(s["updated_at"]),
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
            str(v["version_number"]), _fmt_ts(v["captured_at"]),
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
                _fmt_ts(a["created_at"]), a["attempt_type"], a["status"],
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


async def cmd_extract_source(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    match = await _resolve_source(kb_db, args.source_id)
    if match is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return

    version = await kb_db.get_latest_version(match["id"])
    if version is None:
        console.print(f"[red]No ingested version found for source {match['id']}[/red]")
        return

    artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
    if not artifacts:
        console.print(f"[red]No chunked artifact found — run `chunk-source {match['id'][:8]}` first[/red]")
        return
    artifact = artifacts[0]

    console.print(f"Extracting from {match.get('title') or match['id']}...")
    extraction_result = await run_extraction(kb_db, config, artifact["id"], force=args.force)

    if extraction_result.status == "empty":
        console.print("[yellow]No chunks to extract from.[/yellow]")
        return
    if extraction_result.status == "unchanged":
        console.print(
            f"[yellow]Already extracted with this model/prompt (run {extraction_result.extraction_run_id})[/yellow]"
        )
        console.print(f"  observations: {extraction_result.observation_count}")
        return

    verb = "[green]Extracted[/green]" if extraction_result.status == "extracted" else "[yellow]Partially extracted[/yellow]"
    console.print(f"{verb} — {extraction_result.observation_count} observation(s) from {extraction_result.chunk_count} chunk(s)")
    if extraction_result.failed_chunk_count:
        console.print(f"  [red]{extraction_result.failed_chunk_count} chunk(s) failed — rerun to retry[/red]")

    console.print("Resolving and promoting...")
    promotion = await resolve_and_promote(kb_db, config, extraction_result.extraction_run_id)
    console.print(f"  promoted:          {promotion.promoted_count}")
    console.print(f"  new claims:        {promotion.new_claim_count}")
    console.print(f"  new entities:      {promotion.new_entity_count}")
    console.print(f"  entity candidates: {promotion.entity_candidate_count}")
    console.print(f"  claim candidates:  {promotion.claim_candidate_count}")

    # Forward-check (decision 27): new claims get checked against every
    # existing topic, not just topics created after this point.
    topic_results = await check_claims_against_topics(kb_db, promotion.new_claim_ids)
    if topic_results:
        console.print("  topic suggestions:")
        for topic_id, result in topic_results.items():
            topic = await kb_db.get_topic(topic_id)
            console.print(
                f"    {topic['name']}: {result.claims_suggested} claim(s), {result.sources_suggested} source(s)"
            )


async def cmd_list_claims(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    claims = await kb_db.list_claims(limit=args.limit)
    if not claims:
        console.print("[dim]No claims yet.[/dim]")
        return

    table = Table(title="Claims")
    table.add_column("ID", style="cyan", max_width=10)
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Conf.")
    table.add_column("Text", style="white", max_width=70)
    for c in claims:
        table.add_row(
            c["id"][:8] + "...", c["claim_type"], c["status"],
            f"{c['confidence']:.2f}" if c["confidence"] is not None else "-",
            c["canonical_text"],
        )
    console.print(table)


async def cmd_show_claim(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    claims = await kb_db.list_claims(limit=5000)
    match = next((c for c in claims if c["id"].startswith(args.claim_id)), None)
    if match is None:
        console.print(f"[red]No claim found matching ID prefix {args.claim_id!r}[/red]")
        return

    console.print(f"[bold]{match['canonical_text']}[/bold]")
    console.print(f"  id:         {match['id']}")
    console.print(f"  type:       {match['claim_type']}")
    console.print(f"  status:     {match['status']}")
    console.print(f"  confidence: {match['confidence']}")
    console.print(f"  importance: {match['importance_score']}")

    evidence = await kb_db.list_claim_evidence(match["id"])
    table = Table(title="Evidence")
    table.add_column("Source")
    table.add_column("Excerpt", max_width=60)
    for e in evidence:
        table.add_row(e.get("source_title") or e["canonical_uri"], e.get("excerpt_text") or "")
    console.print(table)


async def cmd_list_resolution_candidates(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    candidates = await kb_db.list_resolution_candidates(candidate_type=args.type, status=args.status, limit=args.limit)
    if not candidates:
        console.print("[dim]No resolution candidates.[/dim]")
        return

    for c in candidates:
        console.print(f"\n[bold]{c['id'][:8]}...[/bold]  {c['candidate_type']}  score={c['score']:.3f}  method={c['method']}")
        if c["left_entity_id"]:
            left = await kb_db.get_entity(c["left_entity_id"])
            right = await kb_db.get_entity(c["right_entity_id"])
            console.print(f"  {left['name']!r}  <->  {right['name']!r}")
        elif c["left_claim_id"]:
            left = await kb_db.get_claim(c["left_claim_id"])
            right = await kb_db.get_claim(c["right_claim_id"])
            console.print(f"  {left['canonical_text']!r}")
            console.print(f"  {right['canonical_text']!r}")
        if c.get("reason"):
            console.print(f"  [dim]{c['reason']}[/dim]")


async def cmd_review_candidate(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    candidate = await kb_db.get_resolution_candidate(args.candidate_id)
    if candidate is None:
        # allow prefix match
        candidates = await kb_db.list_resolution_candidates(status=None, limit=5000)
        candidate = next((c for c in candidates if c["id"].startswith(args.candidate_id)), None)
    if candidate is None:
        console.print(f"[red]No resolution candidate found matching {args.candidate_id!r}[/red]")
        return

    decision = "accepted" if args.accept else "rejected"
    updated = await kb_db.review_resolution_candidate(candidate["id"], decision)
    console.print(f"Marked {updated['id']} as [bold]{decision}[/bold]")
    if decision == "accepted":
        console.print("[dim]Note: merge execution is not implemented yet — this only records the review decision.[/dim]")


def _print_verification_result(result):
    verb = {
        "supported": "[green]Supported[/green]",
        "contradicted": "[red]Contradicted[/red]",
        "mixed": "[yellow]Mixed[/yellow]",
        "unverified": "[yellow]Still unverified[/yellow] (budget exhausted with no clear signal)",
        "skipped": "[dim]Skipped[/dim] (already verified — use --force to recheck)",
    }[result.status]
    console.print(verb)
    if result.status != "skipped":
        console.print(f"  supports found:     {result.supports_found}")
        console.print(f"  contradicts found:  {result.contradicts_found}")
        console.print(f"  sources examined:   {result.sources_examined}")
        console.print(f"  web searches used:  {result.web_searches_used}")
        if result.contradiction_candidate_ids:
            console.print(
                f"  [red]recorded {len(result.contradiction_candidate_ids)} contradiction(s) "
                f"for review — see list-resolution-candidates --type claim_contradiction[/red]"
            )


async def cmd_verify_claim(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    claims = await kb_db.list_claims(limit=5000)
    match = next((c for c in claims if c["id"].startswith(args.claim_id)), None)
    if match is None:
        console.print(f"[red]No claim found matching ID prefix {args.claim_id!r}[/red]")
        return

    console.print(f"Verifying: {match['canonical_text']}")
    result = await verify_claim(kb_db, config, match["id"], force=args.force)
    _print_verification_result(result)


async def cmd_verify_source(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()

    match = await _resolve_source(kb_db, args.source_id)
    if match is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return

    all_claims = await kb_db.list_claims(limit=5000)
    evidence_by_claim = {}
    for c in all_claims:
        evidence = await kb_db.list_claim_evidence(c["id"])
        if any(e["source_id"] == match["id"] for e in evidence):
            evidence_by_claim[c["id"]] = c

    threshold = args.threshold if args.threshold is not None else config.kb.verification_importance_threshold
    eligible = [
        c for c in evidence_by_claim.values()
        if (c["importance_score"] or 0) >= threshold and c["verification_attempted_at"] is None
    ]
    if not eligible:
        console.print(f"[dim]No unverified claims from this source at or above importance {threshold}.[/dim]")
        return

    console.print(f"Verifying {len(eligible)} claim(s) from {match.get('title') or match['id']}...")
    for claim in eligible:
        console.print(f"\n[bold]{claim['canonical_text']}[/bold]")
        result = await verify_claim(kb_db, config, claim["id"], force=args.force)
        _print_verification_result(result)


async def _resolve_topic(kb_db, topic_id_prefix: str) -> dict | None:
    topics = await kb_db.list_topics(limit=1000)
    return next((t for t in topics if t["id"].startswith(topic_id_prefix) or t["slug"] == topic_id_prefix), None)


async def cmd_create_topic(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await kb_db.create_topic(args.name, description=args.description)
    console.print(f"[green]Created topic[/green] {topic['name']} ({topic['id']})")
    console.print(f"  slug: {topic['slug']}")


async def cmd_list_topics(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topics = await kb_db.list_topics(limit=args.limit)
    if not topics:
        console.print("[dim]No topics yet.[/dim]")
        return
    table = Table(title="Topics")
    table.add_column("ID", style="cyan", max_width=10)
    table.add_column("Name", style="white")
    table.add_column("Slug", style="dim")
    table.add_column("Updated", style="dim")
    for t in topics:
        table.add_row(t["id"][:8] + "...", t["name"], t["slug"], _fmt_ts(t["updated_at"]))
    console.print(table)


async def cmd_attach_source(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return
    source = await _resolve_source(kb_db, args.source_id)
    if source is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return
    await kb_db.attach_source_to_topic(topic["id"], source["id"], link_reason="manual_attach")
    claims = await kb_db.list_topic_claims(topic["id"])
    console.print(f"[green]Attached[/green] {source.get('title') or source['id']} to {topic['name']}")
    console.print(f"  ({len(claims)} claims now attached to this topic in total)")


async def cmd_attach_claim(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return
    claims = await kb_db.list_claims(limit=5000)
    claim = next((c for c in claims if c["id"].startswith(args.claim_id)), None)
    if claim is None:
        console.print(f"[red]No claim found matching ID prefix {args.claim_id!r}[/red]")
        return
    await kb_db.attach_claim_to_topic(topic["id"], claim["id"], link_reason="manual_attach")
    console.print(f"[green]Attached[/green] claim to {topic['name']}: {claim['canonical_text'][:80]}")


async def cmd_backfill_topic(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return
    result = await generate_topic_suggestions(kb_db, topic["id"])
    console.print(f"Backfilled suggestions for {topic['name']}:")
    console.print(f"  claims suggested:  {result.claims_suggested}")
    console.print(f"  sources suggested: {result.sources_suggested}")


async def cmd_show_topic(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return

    console.print(f"[bold]{topic['name']}[/bold]  ({topic['slug']})")
    if topic.get("description"):
        console.print(f"  {topic['description']}")

    timeline = await get_topic_timeline(kb_db, topic["id"])
    if timeline:
        console.print(f"\n[bold]Timeline[/bold] ({len(timeline)} dated events)")
        for entry in timeline:
            console.print(f"\n  [cyan]{entry.event.get('start_at')}[/cyan] — {entry.event['title']}")
            for claim in entry.claims:
                console.print(f"    - {claim['canonical_text']}")

    all_claims = await kb_db.list_topic_claims(topic["id"], link_status="attached")
    console.print(f"\n[bold]Attached claims:[/bold] {len(all_claims)} total")

    suggested_claims = await kb_db.list_topic_claims(topic["id"], link_status="suggested")
    suggested_sources = await kb_db.list_topic_sources(topic["id"], link_status="suggested")
    if suggested_claims or suggested_sources:
        console.print(
            f"\n[yellow]Pending suggestions:[/yellow] {len(suggested_claims)} claim(s), "
            f"{len(suggested_sources)} source(s) — see review-topic-suggestion"
        )


async def cmd_review_topic_suggestion(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return

    decision = "attached" if args.accept else "rejected"
    if args.type == "claim":
        claims = await kb_db.list_topic_claims(topic["id"], link_status="suggested")
        match = next((c for c in claims if c["id"].startswith(args.item_id)), None)
        if match is None:
            console.print(f"[red]No suggested claim matching {args.item_id!r}[/red]")
            return
        updated = await kb_db.review_topic_claim_link(topic["id"], match["id"], decision)
    else:
        sources = await kb_db.list_topic_sources(topic["id"], link_status="suggested")
        match = next((s for s in sources if s["id"].startswith(args.item_id)), None)
        if match is None:
            console.print(f"[red]No suggested source matching {args.item_id!r}[/red]")
            return
        updated = await kb_db.review_topic_source_link(topic["id"], match["id"], decision)
    console.print(f"Marked as [bold]{decision}[/bold]")


async def cmd_generate_report(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    topic = await _resolve_topic(kb_db, args.topic_id)
    if topic is None:
        console.print(f"[red]No topic found matching {args.topic_id!r}[/red]")
        return
    console.print(f"Generating report for {topic['name']}...")
    result = await generate_topic_report(kb_db, config, topic["id"])
    console.print()
    console.print(result.content_markdown)
    if result.suggestion:
        console.print()
        console.print(f"[yellow]{result.suggestion}[/yellow]")


async def cmd_set_preferred_source(args):
    config, kb_db, _ = _kb_setup(args)
    await kb_db.init()
    claims = await kb_db.list_claims(limit=5000)
    claim = next((c for c in claims if c["id"].startswith(args.claim_id)), None)
    if claim is None:
        console.print(f"[red]No claim found matching ID prefix {args.claim_id!r}[/red]")
        return
    source = await _resolve_source(kb_db, args.source_id)
    if source is None:
        console.print(f"[red]No source found matching ID prefix {args.source_id!r}[/red]")
        return
    await kb_db.set_preferred_source_manual(claim["id"], source["id"], reviewed_by="user")
    console.print(f"[green]Set preferred source[/green] for claim to {source.get('title') or source['id']}")


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

    p_extract = subparsers.add_parser("extract-source", help="Extract claims/entities/events from a chunked source")
    p_extract.add_argument("source_id", help="Source ID or prefix")
    p_extract.add_argument("--force", action="store_true", help="Re-extract even if this model/prompt already ran")
    p_extract.set_defaults(func=cmd_extract_source)

    p_claims = subparsers.add_parser("list-claims", help="List canonical claims")
    p_claims.add_argument("--limit", type=int, default=100)
    p_claims.set_defaults(func=cmd_list_claims)

    p_show_claim = subparsers.add_parser("show-claim", help="Show a claim and its evidence")
    p_show_claim.add_argument("claim_id", help="Claim ID or prefix")
    p_show_claim.set_defaults(func=cmd_show_claim)

    p_candidates = subparsers.add_parser("list-resolution-candidates", help="List entity/claim merge candidates for review")
    p_candidates.add_argument(
        "--type", choices=["entity_duplicate", "claim_duplicate", "claim_contradiction"], default=None,
    )
    p_candidates.add_argument("--status", default="open")
    p_candidates.add_argument("--limit", type=int, default=50)
    p_candidates.set_defaults(func=cmd_list_resolution_candidates)

    p_review = subparsers.add_parser("review-candidate", help="Accept or reject a resolution candidate")
    p_review.add_argument("candidate_id", help="Resolution candidate ID or prefix")
    group = p_review.add_mutually_exclusive_group(required=True)
    group.add_argument("--accept", action="store_true")
    group.add_argument("--reject", action="store_true")
    p_review.set_defaults(func=cmd_review_candidate)

    p_verify_claim = subparsers.add_parser("verify-claim", help="Verify a claim against the KB and, if needed, the web")
    p_verify_claim.add_argument("claim_id", help="Claim ID or prefix")
    p_verify_claim.add_argument("--force", action="store_true", help="Re-verify even if already attempted")
    p_verify_claim.set_defaults(func=cmd_verify_claim)

    p_verify_source = subparsers.add_parser(
        "verify-source", help="Verify all unverified claims from a source above the importance threshold",
    )
    p_verify_source.add_argument("source_id", help="Source ID or prefix")
    p_verify_source.add_argument("--threshold", type=float, default=None, help="Overrides kb.verification_importance_threshold")
    p_verify_source.add_argument("--force", action="store_true")
    p_verify_source.set_defaults(func=cmd_verify_source)

    p_create_topic = subparsers.add_parser("create-topic", help="Create a topic")
    p_create_topic.add_argument("name")
    p_create_topic.add_argument("--description")
    p_create_topic.set_defaults(func=cmd_create_topic)

    p_list_topics = subparsers.add_parser("list-topics", help="List topics")
    p_list_topics.add_argument("--limit", type=int, default=50)
    p_list_topics.set_defaults(func=cmd_list_topics)

    p_show_topic = subparsers.add_parser("show-topic", help="Show a topic's timeline, claims, and pending suggestions")
    p_show_topic.add_argument("topic_id", help="Topic ID, ID prefix, or slug")
    p_show_topic.set_defaults(func=cmd_show_topic)

    p_attach_source = subparsers.add_parser("attach-source", help="Attach a source (and its claims) to a topic")
    p_attach_source.add_argument("topic_id")
    p_attach_source.add_argument("source_id")
    p_attach_source.set_defaults(func=cmd_attach_source)

    p_attach_claim = subparsers.add_parser("attach-claim", help="Attach a single claim to a topic")
    p_attach_claim.add_argument("topic_id")
    p_attach_claim.add_argument("claim_id")
    p_attach_claim.set_defaults(func=cmd_attach_claim)

    p_backfill = subparsers.add_parser(
        "backfill-topic-suggestions", help="(Re-)scan the whole KB for entity-overlap suggestions for a topic",
    )
    p_backfill.add_argument("topic_id")
    p_backfill.set_defaults(func=cmd_backfill_topic)

    p_review_topic = subparsers.add_parser("review-topic-suggestion", help="Accept or reject a topic suggestion")
    p_review_topic.add_argument("topic_id")
    p_review_topic.add_argument("item_id", help="Claim or source ID/prefix")
    p_review_topic.add_argument("--type", choices=["claim", "source"], required=True)
    review_group = p_review_topic.add_mutually_exclusive_group(required=True)
    review_group.add_argument("--accept", action="store_true")
    review_group.add_argument("--reject", action="store_true")
    p_review_topic.set_defaults(func=cmd_review_topic_suggestion)

    p_report = subparsers.add_parser("generate-report", help="Generate a timeline report for a topic")
    p_report.add_argument("topic_id")
    p_report.set_defaults(func=cmd_generate_report)

    p_pref_source = subparsers.add_parser("set-preferred-source", help="Manually override a claim's preferred source")
    p_pref_source.add_argument("claim_id")
    p_pref_source.add_argument("source_id")
    p_pref_source.set_defaults(func=cmd_set_preferred_source)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
