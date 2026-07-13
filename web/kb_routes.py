"""Web API routes for the knowledge base: topics, timelines, reports,
suggestion review (build order step 7).

Kept as a separate router module from web/app.py (the research-agent API) so
the two concerns — chat sessions vs. the knowledge base — stay as separate in
the web layer as they already are in storage (SQLite sessions vs. Postgres
KB). init_kb() is called from app.py's lifespan to share one KBDatabase pool.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pgvector import Vector
from pydantic import BaseModel

from deep_research.config import Config
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.conversation import get_topic_conversation_transcript
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import backfill_embeddings, embed_texts
from deep_research.kb.extraction import run_extraction
from deep_research.kb.ingest import ingest_file, ingest_pasted_text, ingest_web_page, ingest_youtube_video
from deep_research.kb.merge import review_and_execute
from deep_research.kb.reports import generate_topic_report
from deep_research.kb.resolution import resolve_and_promote
from deep_research.kb.storage import SnapshotStore
from deep_research.kb.timeline import get_topic_timeline
from deep_research.kb.topics import check_claims_against_topics, generate_topic_suggestions
from deep_research.kb.trust import set_trust_tier_if_missing
from deep_research.kb.verification import (
    claim_check_status,
    is_claim_eligible_for_verification,
    run_verification_sweep,
    verify_claim,
    verify_claims_concurrently,
)

router = APIRouter(prefix="/api/kb")

config: Config | None = None
kb_db: KBDatabase | None = None
snapshot_store: SnapshotStore | None = None


async def init_kb(cfg: Config):
    """Best-effort — the app must still start if Postgres isn't running/
    configured. A missing KB means /api/kb/* routes fail per-request instead
    of the whole app failing to boot, and the research agent's kb_search
    tool/prioritize_kb toggle are simply unavailable rather than fatal."""
    global config, kb_db, snapshot_store
    config = cfg
    try:
        kb_db = KBDatabase(cfg.kb.postgres_dsn)
        await kb_db.init()
        snapshot_store = SnapshotStore(cfg.kb_snapshot_dir)
    except Exception as e:
        print(f"Local knowledge base unavailable ({e}); /api/kb/* routes and kb_search will not work.")
        kb_db = None


async def close_kb():
    if kb_db is not None:
        await kb_db.close()


async def _get_topic_or_404(topic_id: str) -> dict:
    topics = await kb_db.list_topics(limit=1000)
    topic = next((t for t in topics if t["id"] == topic_id or t["id"].startswith(topic_id) or t["slug"] == topic_id), None)
    if topic is None:
        raise HTTPException(404, "Topic not found")
    return topic


class CreateTopicRequest(BaseModel):
    name: str
    description: str | None = None


class ReviewRequest(BaseModel):
    decision: str  # "attached" | "rejected"


class ReviewCandidateRequest(BaseModel):
    decision: str  # "accepted" | "rejected"


class AttachSourceRequest(BaseModel):
    source_id: str


class AttachClaimRequest(BaseModel):
    claim_id: str


class PreferredSourceRequest(BaseModel):
    source_id: str


class VerificationOverrideRequest(BaseModel):
    override: str | None = None  # 'include' | 'exclude' | None


class VerificationContextRequest(BaseModel):
    context: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    trust_tier: str | None = None


class IngestYoutubeRequest(BaseModel):
    url: str
    trust_tier: str | None = None


class IngestConversationRequest(BaseModel):
    text: str
    title: str | None = None
    trust_tier: str | None = None
    threshold: float | None = None
    topic_name: str | None = None


class ChunkSourceRequest(BaseModel):
    chunk_size: int = 1200


class ExtractSourceRequest(BaseModel):
    force: bool = False


class VerifySourceRequest(BaseModel):
    threshold: float | None = None
    force: bool = False


class VerifyClaimRequest(BaseModel):
    force: bool = False


class TriggerVerificationRunRequest(BaseModel):
    threshold: float | None = None
    force: bool = False


def _serialize(obj):
    """FastAPI/pydantic can't serialize asyncpg Record-derived dicts with
    datetime or pgvector.Vector objects out of the box via plain dict returns
    — return plain JSON-safe dicts explicitly wherever those are involved.
    Claims/entities carry a real `embedding` value since step 8 (previously
    always NULL/None during earlier testing, which happens to encode fine —
    this only started raising once real data existed), so any route
    returning a raw claim/entity dict needs this or it 500s. Dropped (to
    None) rather than sent as a raw 768-float list: no frontend view uses the
    embedding, and it would otherwise bloat every claims/timeline response."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, Vector):
        return None
    return obj


def _with_check_status(claims: list[dict]) -> list[dict]:
    """Attaches check_status ('auto_check' | 'auto_skip' | 'manual_include' |
    'manual_exclude') to each claim -- the "will this be auto-verified, and
    why" breakdown, for the paste-a-conversation flow and any other claims
    list that wants to show/toggle it."""
    threshold = config.kb.verification_importance_threshold
    return [{**c, "check_status": claim_check_status(c, threshold)} for c in claims]


@router.get("/topics")
async def list_topics(limit: int = 50):
    topics = await kb_db.list_topics(limit=limit)
    return {"topics": _serialize(topics)}


@router.post("/topics")
async def create_topic(req: CreateTopicRequest):
    topic = await kb_db.create_topic(req.name, description=req.description)
    return {"topic": _serialize(topic)}


@router.get("/topics/{topic_id}")
async def get_topic(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    claims = await kb_db.list_topic_claims(topic["id"], link_status="attached")
    suggested_claims = await kb_db.list_topic_claims(topic["id"], link_status="suggested")
    suggested_sources = await kb_db.list_topic_sources(topic["id"], link_status="suggested")
    attached_sources = await kb_db.list_topic_sources(topic["id"], link_status="attached")
    return {
        "topic": _serialize(topic),
        "claim_count": len(claims),
        "source_count": len(attached_sources),
        "pending_suggestion_count": len(suggested_claims) + len(suggested_sources),
    }


@router.get("/topics/{topic_id}/timeline")
async def get_timeline(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    entries = await get_topic_timeline(kb_db, topic["id"])
    return {
        "entries": [
            {
                "event": _serialize(entry.event),
                "claims": _serialize(entry.claims),
                "sort_date": entry.sort_date.isoformat() if entry.sort_date else None,
            }
            for entry in entries
        ]
    }


@router.get("/topics/{topic_id}/conversation")
async def get_conversation_transcript(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    turns = await get_topic_conversation_transcript(kb_db, topic["id"])

    # Claims embedded per-turn need check_status too (ClaimListItem's badge
    # relies on it) -- annotate once across the deduplicated set rather than
    # per-turn, since the same claim can be evidenced from more than one turn.
    by_id = {c["id"]: c for turn in turns for c in turn["claims"]}
    annotated_by_id = {c["id"]: c for c in _with_check_status(list(by_id.values()))}
    for turn in turns:
        turn["claims"] = [annotated_by_id[c["id"]] for c in turn["claims"]]

    return {"turns": _serialize(turns)}


@router.get("/topics/{topic_id}/claims")
async def list_claims(topic_id: str, status: str = "attached"):
    topic = await _get_topic_or_404(topic_id)
    claims = await kb_db.list_topic_claims(topic["id"], link_status=status)
    return {"claims": _serialize(_with_check_status(claims))}


@router.get("/topics/{topic_id}/sources")
async def list_sources(topic_id: str, status: str = "attached"):
    topic = await _get_topic_or_404(topic_id)
    sources = await kb_db.list_topic_sources(topic["id"], link_status=status)
    return {"sources": _serialize(sources)}


@router.post("/topics/{topic_id}/sources")
async def attach_source(topic_id: str, req: AttachSourceRequest):
    topic = await _get_topic_or_404(topic_id)
    source = await kb_db.get_source(req.source_id)
    if source is None:
        raise HTTPException(404, "Source not found")
    row = await kb_db.attach_source_to_topic(topic["id"], source["id"], link_reason="manual_attach")
    return {"link": _serialize(row)}


@router.post("/topics/{topic_id}/claims")
async def attach_claim(topic_id: str, req: AttachClaimRequest):
    topic = await _get_topic_or_404(topic_id)
    claim = await kb_db.get_claim(req.claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    row = await kb_db.attach_claim_to_topic(topic["id"], claim["id"], link_reason="manual_attach")
    return {"link": _serialize(row)}


@router.post("/topics/{topic_id}/sources/{source_id}/review")
async def review_source(topic_id: str, source_id: str, req: ReviewRequest):
    topic = await _get_topic_or_404(topic_id)
    row = await kb_db.review_topic_source_link(topic["id"], source_id, req.decision)
    return {"link": _serialize(row)}


@router.post("/topics/{topic_id}/claims/{claim_id}/review")
async def review_claim(topic_id: str, claim_id: str, req: ReviewRequest):
    topic = await _get_topic_or_404(topic_id)
    row = await kb_db.review_topic_claim_link(topic["id"], claim_id, req.decision)
    return {"link": _serialize(row)}


@router.post("/topics/{topic_id}/backfill")
async def backfill(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    result = await generate_topic_suggestions(kb_db, config, topic["id"])
    return {"claims_suggested": result.claims_suggested, "sources_suggested": result.sources_suggested}


@router.get("/topics/{topic_id}/report")
async def get_report(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    report = await kb_db.get_latest_report(topic["id"])
    if report is None:
        return {"report": None}
    return {"report": _serialize(report)}


@router.post("/topics/{topic_id}/report")
async def create_report(topic_id: str):
    topic = await _get_topic_or_404(topic_id)
    result = await generate_topic_report(kb_db, config, topic["id"])
    return {
        "report_id": result.report_id,
        "content_markdown": result.content_markdown,
        "suggestion": result.suggestion,
    }


def _enrich_related_claim(other_id: str, claims_by_id: dict, evidence_by_claim: dict) -> dict | None:
    other = claims_by_id.get(other_id)
    if other is None:
        return None
    sources = [
        {"source_id": e["source_id"], "source_title": e["source_title"], "canonical_uri": e["canonical_uri"]}
        for e in evidence_by_claim.get(other_id, [])
    ]
    # de-dupe -- a claim can have multiple evidence rows from the same source
    seen_source_ids = set()
    unique_sources = []
    for s in sources:
        if s["source_id"] in seen_source_ids:
            continue
        seen_source_ids.add(s["source_id"])
        unique_sources.append(s)
    return {"id": other["id"], "canonical_text": other["canonical_text"], "sources": unique_sources}


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: str):
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    evidence = await kb_db.list_claim_evidence(claim_id)

    contradictions = await kb_db.get_claim_contradictions(claim_id)
    supporting_ids = ((claim.get("verification_notes") or {}).get("supporting_claim_ids")) or []
    related_ids = list({c["other_claim_id"] for c in contradictions} | set(supporting_ids))
    claims_by_id = await kb_db.get_claims_bulk(related_ids)
    evidence_by_claim = await kb_db.get_claims_evidence_bulk(related_ids)

    contradicting_claims = []
    for c in contradictions:
        enriched = _enrich_related_claim(c["other_claim_id"], claims_by_id, evidence_by_claim)
        if enriched:
            contradicting_claims.append({
                **enriched,
                "candidate_id": c["candidate_id"],
                "candidate_status": c["candidate_status"],
                "reason": c["reason"],
                "score": c["score"],
            })

    supporting_claims = [
        c for c in (_enrich_related_claim(i, claims_by_id, evidence_by_claim) for i in supporting_ids) if c
    ]

    return {
        "claim": _serialize(claim),
        "evidence": _serialize(evidence),
        "supporting_claims": _serialize(supporting_claims),
        "contradicting_claims": _serialize(contradicting_claims),
    }


@router.put("/claims/{claim_id}/preferred-source")
async def set_preferred_source(claim_id: str, req: PreferredSourceRequest):
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    updated = await kb_db.set_preferred_source_manual(claim_id, req.source_id, reviewed_by="web_ui")
    return {"claim": _serialize(updated)}


@router.put("/claims/{claim_id}/verification-override")
async def set_claim_verification_override(claim_id: str, req: VerificationOverrideRequest):
    """Flag/deflag a claim for auto-verification -- not every extracted
    statement needs a claim check, and this lets a human correct the
    system's importance-based guess either way (force-include something it
    skipped, or force-exclude something it flagged that isn't worth
    checking)."""
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    if req.override not in (None, "include", "exclude"):
        raise HTTPException(400, "override must be 'include', 'exclude', or null")
    updated = await kb_db.set_claim_verification_override(claim_id, req.override)
    return {"claim": _serialize({**updated, "check_status": claim_check_status(
        updated, config.kb.verification_importance_threshold,
    )})}


@router.put("/claims/{claim_id}/verification-context")
async def set_claim_verification_context(claim_id: str, req: VerificationContextRequest):
    """Lets a human expand what a claim's verification pass actually looks
    for beyond the literal claim text (e.g. claim "industrial buildings use
    more electricity than residential" + context "compare specifically
    against datacenter usage") -- see verify_claim's use of
    claims.verification_context. Setting it doesn't itself trigger a
    recheck; force-reverify via the existing verify endpoint afterward to
    have it take effect on a claim that was already checked."""
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    updated = await kb_db.set_claim_verification_context(claim_id, req.context)
    return {"claim": _serialize({**updated, "check_status": claim_check_status(
        updated, config.kb.verification_importance_threshold,
    )})}


def _evidence_summary(rows: list[dict]) -> list[dict]:
    """Trims claim_evidence rows down to what a reviewer actually needs to
    judge a claim_duplicate/claim_contradiction candidate: which source it
    came from and the exact quoted excerpt -- not the full evidence row."""
    return [
        {
            "source_title": r.get("source_title") or r.get("canonical_uri"),
            "canonical_uri": r.get("canonical_uri"),
            "excerpt": r.get("excerpt_text"),
        }
        for r in rows
    ]


def _enrich_candidates(
    candidates: list[dict], entities: dict[str, dict], claims: dict[str, dict],
    claims_evidence: dict[str, list[dict]],
) -> list[dict]:
    """Resolution candidates only store entity/claim IDs (see cli/kb.py's
    list-resolution-candidates for the same pattern) — resolve them to a
    human-readable label, plus enough supporting context (evidence excerpts
    for claims) for a human to actually judge the pair instead of comparing
    two bare sentences with no idea where either came from. `entities`/
    `claims`/`claims_evidence` are pre-fetched in bulk by the caller (one
    query each for the whole list) rather than one lookup per row, which
    turned a single list call into hundreds of sequential round trips once
    the KB had a few hundred candidates."""
    enriched = []
    for c in candidates:
        row = _serialize(c)
        if c.get("left_entity_id"):
            left = entities.get(c["left_entity_id"])
            right = entities.get(c["right_entity_id"])
            row["left_label"] = left["name"] if left else "(deleted entity)"
            row["right_label"] = right["name"] if right else "(deleted entity)"
            row["left_entity_type"] = left["entity_type"] if left else None
            row["right_entity_type"] = right["entity_type"] if right else None
        elif c.get("left_claim_id"):
            left = claims.get(c["left_claim_id"])
            right = claims.get(c["right_claim_id"])
            row["left_label"] = left["canonical_text"] if left else "(deleted claim)"
            row["right_label"] = right["canonical_text"] if right else "(deleted claim)"
            row["left_evidence"] = _evidence_summary(claims_evidence.get(c["left_claim_id"], []))
            row["right_evidence"] = _evidence_summary(claims_evidence.get(c["right_claim_id"], []))
        enriched.append(row)
    return enriched


@router.get("/resolution-candidates")
async def list_resolution_candidates(status: str = "open", type: str | None = None, limit: int = 200):
    candidates = await kb_db.list_resolution_candidates(candidate_type=type, status=status, limit=limit)
    entity_ids = [c[k] for c in candidates for k in ("left_entity_id", "right_entity_id") if c.get(k)]
    claim_ids = [c[k] for c in candidates for k in ("left_claim_id", "right_claim_id") if c.get(k)]
    entities = await kb_db.get_entities_bulk(entity_ids)
    claims = await kb_db.get_claims_bulk(claim_ids)
    claims_evidence = await kb_db.get_claims_evidence_bulk(claim_ids)
    return {"candidates": _enrich_candidates(candidates, entities, claims, claims_evidence)}


@router.post("/resolution-candidates/{candidate_id}/review")
async def review_candidate(candidate_id: str, req: ReviewCandidateRequest):
    candidate = await kb_db.get_resolution_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(404, "Resolution candidate not found")
    result = await review_and_execute(kb_db, candidate_id, req.decision, reviewed_by="web_ui")
    return {
        "candidate_id": result.candidate_id,
        "decision": result.decision,
        "candidate_type": result.candidate_type,
        "action": result.action,
        "winner_id": result.winner_id,
        "loser_id": result.loser_id,
    }


@router.get("/sources")
async def search_sources(q: str = "", limit: int = 50, include_conversations: bool = False):
    """A pasted conversation's underlying 'conversation' source is pipeline
    plumbing (it needs a source to hang chunks/extraction/evidence off of),
    not something meant to be browsed here -- the topic it's attached to is
    the actual thing to look at, so these are hidden by default."""
    sources = await kb_db.list_sources(limit=500)
    if not include_conversations:
        sources = [s for s in sources if s.get("source_type_code") != "conversation"]
    if q:
        q_lower = q.lower()
        sources = [
            s for s in sources
            if q_lower in (s.get("title") or "").lower() or q_lower in s["canonical_uri"].lower()
        ][:limit]
    else:
        sources = sources[:limit]
    return {"sources": _serialize(sources)}


@router.post("/sources/ingest-url")
async def ingest_url_route(req: IngestUrlRequest):
    result = await ingest_web_page(req.url, config, kb_db, snapshot_store, trust_tier_code=req.trust_tier)
    if result.status != "failed" and result.source_id:
        await set_trust_tier_if_missing(kb_db, config, result.source_id)
    return {"result": asdict(result)}


@router.post("/sources/ingest-youtube")
async def ingest_youtube_route(req: IngestYoutubeRequest):
    result = await ingest_youtube_video(req.url, kb_db, snapshot_store, trust_tier_code=req.trust_tier)
    if result.status != "failed" and result.source_id:
        await set_trust_tier_if_missing(kb_db, config, result.source_id)
    return {"result": asdict(result)}


@router.post("/sources/ingest-file")
async def ingest_file_route(file: UploadFile = File(...), trust_tier: str | None = Form(None)):
    """Web uploads are saved under a stable uploads directory keyed by the
    original filename (not a random temp path) so ingest_file's file-path-
    based source identity (decision from step 2: identity is the path, not
    the content hash) behaves sensibly on re-upload — uploading "the same"
    file again is treated as a new version of the same source, matching the
    CLI's ingest-file semantics for a stable local path."""
    uploads_dir = Path("~/.local/share/deep_research/kb_uploads").expanduser()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename
    dest.write_bytes(await file.read())
    result = await ingest_file(dest, kb_db, snapshot_store, trust_tier_code=trust_tier)
    if result.status != "failed" and result.source_id:
        await set_trust_tier_if_missing(kb_db, config, result.source_id)
    return {"result": asdict(result)}


_processing_source_ids: set[str] = set()


async def _process_conversation_source(source_id: str, threshold: float | None, topic_id: str | None) -> None:
    """Runs chunk -> extract -> resolve/promote -> verify in the background
    after a conversation's fast synchronous ingest step returns, so the web
    request doesn't block for however long verification takes (minutes, per
    HARDWARE.md's timing measurements) -- the client gets the source_id back
    immediately and can navigate to the Source Detail page right away, which
    polls /processing to show that work is still happening. Runs to
    completion regardless of whether the client is still around to see it;
    an in-memory set (not a DB row) is enough tracking since this is a
    single-worker dev server, not a durably-scheduled job like the nightly
    verification sweep."""
    _processing_source_ids.add(source_id)
    try:
        source = await kb_db.get_source(source_id)
        version = await kb_db.get_latest_version(source_id)
        chunk_result = await build_artifact_for_version(kb_db, snapshot_store, source, version, config=config)
        if chunk_result.chunk_count == 0:
            return

        artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
        extraction_result = await run_extraction(kb_db, config, artifacts[0]["id"])
        if extraction_result.observation_count == 0:
            return

        promotion = await resolve_and_promote(kb_db, config, extraction_result.extraction_run_id)
        await check_claims_against_topics(kb_db, config, promotion.new_claim_ids)
        if topic_id is not None:
            # attach_source_to_topic was already called once at ingest time
            # (before any claims existed, just to link the topic<->source);
            # calling it again now sweeps the claims resolve_and_promote just
            # created into the topic as 'attached', not merely 'suggested' --
            # the whole point of grouping a pasted conversation under a topic
            # is to see its claims there without a manual review step.
            await kb_db.attach_source_to_topic(topic_id, source_id, link_reason="conversation_paste")

        claims = await kb_db.list_claims_for_source(source_id, limit=500)
        eff_threshold = threshold if threshold is not None else config.kb.verification_importance_threshold
        eligible = [c for c in claims if is_claim_eligible_for_verification(c, eff_threshold)]
        await verify_claims_concurrently(kb_db, config, eligible)
    finally:
        _processing_source_ids.discard(source_id)


@router.post("/sources/ingest-conversation")
async def ingest_conversation_route(req: IngestConversationRequest, background_tasks: BackgroundTasks):
    """Paste a chat conversation, get claims extracted (tagged with who said
    them) and verified against independent sources, grouped under a topic so
    there's context beyond a flat claims list -- a pasted conversation is
    always attached to a topic (named after req.topic_name, or the source's
    own title if not given), reusing an existing topic with the same name if
    one exists so multiple related pastes can be grouped together. Only the
    ingest step (writing the text as a source) happens synchronously --
    chunk/extract/verify can take minutes, so they run as a background task
    instead of blocking the request; the client gets source_id/topic_id back
    immediately and should navigate to the Topic Detail page, which shows
    live progress via GET .../processing rather than leaving the Sources
    page looking frozen with no way to tell whether it's working or whether
    leaving would lose the result (it doesn't -- this keeps running
    server-side either way)."""
    ingest_result = await ingest_pasted_text(
        req.text, kb_db, snapshot_store, title=req.title, trust_tier_code=req.trust_tier,
    )
    if ingest_result.status == "failed":
        raise HTTPException(400, ingest_result.error or "Ingestion failed")

    source = await kb_db.get_source(ingest_result.source_id)
    topic_name = req.topic_name or source["title"] or "Pasted conversation"
    topic, _ = await kb_db.get_or_create_topic(topic_name, topic_type="conversation")
    await kb_db.attach_source_to_topic(topic["id"], ingest_result.source_id, link_reason="conversation_paste")

    background_tasks.add_task(
        _process_conversation_source, ingest_result.source_id, req.threshold, topic["id"],
    )
    return {"source_id": ingest_result.source_id, "topic_id": topic["id"], "status": "processing"}


@router.get("/sources/{source_id}/processing")
async def get_source_processing_status(source_id: str):
    return {"processing": source_id in _processing_source_ids}


_processing_topic_ids: set[str] = set()


async def _process_topic_verification(topic_id: str, threshold: float | None) -> None:
    """Verifies whatever's currently eligible among a topic's attached
    claims -- the manual trigger for claims a human just flagged (or
    unflagged) via verification_override, so they don't have to wait for the
    nightly sweep to pick it up."""
    _processing_topic_ids.add(topic_id)
    try:
        claims = await kb_db.list_topic_claims(topic_id, link_status="attached")
        eff_threshold = threshold if threshold is not None else config.kb.verification_importance_threshold
        eligible = [c for c in claims if is_claim_eligible_for_verification(c, eff_threshold)]
        await verify_claims_concurrently(kb_db, config, eligible)
    finally:
        _processing_topic_ids.discard(topic_id)


@router.post("/topics/{topic_id}/verify")
async def trigger_topic_verification(topic_id: str, background_tasks: BackgroundTasks):
    topic = await _get_topic_or_404(topic_id)
    if topic["id"] in _processing_topic_ids:
        raise HTTPException(409, "Already verifying this topic's claims")
    background_tasks.add_task(_process_topic_verification, topic["id"], None)
    return {"status": "processing"}


@router.get("/topics/{topic_id}/processing")
async def get_topic_processing_status(topic_id: str):
    return {"processing": topic_id in _processing_topic_ids}


async def _resolve_source_or_404(source_id: str) -> dict:
    source = await kb_db.get_source(source_id)
    if source is None:
        sources = await kb_db.list_sources(limit=5000)
        source = next((s for s in sources if s["id"].startswith(source_id)), None)
    if source is None:
        raise HTTPException(404, "Source not found")
    return source


@router.get("/sources/{source_id}")
async def get_source_detail(source_id: str):
    source = await _resolve_source_or_404(source_id)
    versions = await kb_db.list_versions(source["id"])
    fetch_attempts = await kb_db.list_fetch_attempts(source["id"])
    return {
        "source": _serialize(source),
        "versions": _serialize(versions),
        "fetch_attempts": _serialize(fetch_attempts),
    }


@router.post("/sources/{source_id}/chunk")
async def chunk_source_route(source_id: str, req: ChunkSourceRequest):
    source = await _resolve_source_or_404(source_id)
    version = await kb_db.get_latest_version(source["id"])
    if version is None:
        raise HTTPException(400, "No ingested version found for this source")
    result = await build_artifact_for_version(
        kb_db, snapshot_store, source, version, config=config, chunk_size=req.chunk_size,
    )
    return {"result": asdict(result)}


@router.post("/sources/{source_id}/extract")
async def extract_source_route(source_id: str, req: ExtractSourceRequest):
    """Mirrors cli/kb.py's extract-source exactly: extract, then resolve +
    promote, then forward-check the new claims against every existing topic
    (decision 27) — one call from the UI's perspective, same as the CLI."""
    source = await _resolve_source_or_404(source_id)
    version = await kb_db.get_latest_version(source["id"])
    if version is None:
        raise HTTPException(400, "No ingested version found for this source")
    artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
    if not artifacts:
        raise HTTPException(400, "No chunked artifact found — chunk this source first")

    extraction_result = await run_extraction(kb_db, config, artifacts[0]["id"], force=req.force)
    response = {"extraction": asdict(extraction_result)}
    if extraction_result.status in ("extracted", "partial"):
        promotion = await resolve_and_promote(kb_db, config, extraction_result.extraction_run_id)
        response["promotion"] = asdict(promotion)
        topic_results = await check_claims_against_topics(kb_db, config, promotion.new_claim_ids)
        suggestions = []
        for topic_id, result in topic_results.items():
            topic = await kb_db.get_topic(topic_id)
            suggestions.append({
                "topic_id": topic_id,
                "topic_name": topic["name"] if topic else None,
                "claims_suggested": result.claims_suggested,
                "sources_suggested": result.sources_suggested,
            })
        response["topic_suggestions"] = suggestions
    return response


@router.get("/sources/{source_id}/claims")
async def get_source_claims(source_id: str):
    """The "main points" extracted from this source -- every claim backed by
    at least one piece of evidence from it, most important first."""
    source = await _resolve_source_or_404(source_id)
    claims = await kb_db.list_claims_for_source(source["id"])
    return {"claims": _serialize(_with_check_status(claims))}


@router.post("/sources/{source_id}/verify")
async def verify_source_route(source_id: str, req: VerifySourceRequest):
    """Mirrors cli/kb.py's verify-source: verifies every claim backed by this
    source that's at/above the importance threshold and not yet verified,
    concurrently (config.kb.verification_concurrency) rather than one at a
    time -- one claim's failure no longer aborts the whole batch either,
    since verify_claims_concurrently reports it as a 'failed' result instead
    of raising."""
    source = await _resolve_source_or_404(source_id)
    source_claims = await kb_db.list_claims_for_source(source["id"], limit=5000)

    threshold = req.threshold if req.threshold is not None else config.kb.verification_importance_threshold
    eligible = [c for c in source_claims if is_claim_eligible_for_verification(c, threshold, force=req.force)]

    outcomes = await verify_claims_concurrently(kb_db, config, eligible, force=req.force)
    results = []
    for claim, status, result in outcomes:
        if isinstance(result, Exception):
            results.append({"canonical_text": claim["canonical_text"], "status": "failed", "error": str(result)})
        else:
            results.append({"canonical_text": claim["canonical_text"], **asdict(result)})
    return {"verified_count": len(results), "results": results}


@router.get("/claims")
async def list_all_claims(limit: int = 100):
    claims = await kb_db.list_claims(limit=limit)
    return {"claims": _serialize(_with_check_status(claims))}


@router.post("/claims/{claim_id}/verify")
async def verify_claim_route(claim_id: str, req: VerifyClaimRequest):
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    result = await verify_claim(kb_db, config, claim_id, force=req.force)
    return {"result": asdict(result)}


@router.get("/search")
async def search_chunks_route(q: str, semantic: bool = False, limit: int = 20):
    """Raw full-text/semantic search over chunked content -- mirrors
    cli/kb.py's `search` command (distinct from kb_search, the agent tool
    that blends both automatically)."""
    if semantic:
        vectors = await embed_texts([q], config.kb.embedding_base_url, config.kb.embedding_model)
        results = await kb_db.search_chunks_semantic(vectors[0], limit=limit)
    else:
        results = await kb_db.search_chunks(q, limit=limit)
    return {"results": _serialize(results)}


@router.post("/embeddings/backfill")
async def backfill_embeddings_route():
    result = await backfill_embeddings(kb_db, config)
    return {"result": asdict(result)}


# --- Verification runs (nightly cron / manual-trigger status page) ---

@router.get("/verification-runs")
async def list_verification_runs_route(limit: int = 30):
    runs = await kb_db.list_verification_runs(limit=limit)
    return {"runs": _serialize(runs)}


@router.get("/verification-runs/current")
async def get_current_verification_run_route():
    run = await kb_db.get_current_verification_run()
    return {"run": _serialize(run)}


async def _verification_sweep_task(threshold: float | None, force: bool):
    await run_verification_sweep(kb_db, config, trigger="web", threshold=threshold, force=force)


@router.post("/verification-runs/trigger")
async def trigger_verification_run_route(req: TriggerVerificationRunRequest, background_tasks: BackgroundTasks):
    """Lets a user kick off the same KB-wide sweep the nightly cron job runs,
    without needing the CLI. Only one sweep at a time -- verify_claim makes
    real LLM calls against the single local GPU, so stacking sweeps would
    just contend with itself for no benefit."""
    current = await kb_db.get_current_verification_run()
    if current is not None:
        raise HTTPException(409, "A verification run is already in progress")
    background_tasks.add_task(_verification_sweep_task, req.threshold, req.force)
    return {"status": "started"}
