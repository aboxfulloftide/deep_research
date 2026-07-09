"""Web API routes for the knowledge base: topics, timelines, reports,
suggestion review (build order step 7).

Kept as a separate router module from web/app.py (the research-agent API) so
the two concerns — chat sessions vs. the knowledge base — stay as separate in
the web layer as they already are in storage (SQLite sessions vs. Postgres
KB). init_kb() is called from app.py's lifespan to share one KBDatabase pool.
"""

from fastapi import APIRouter, HTTPException
from pgvector import Vector
from pydantic import BaseModel

from deep_research.config import Config
from deep_research.kb.db import KBDatabase
from deep_research.kb.merge import review_and_execute
from deep_research.kb.reports import generate_topic_report
from deep_research.kb.timeline import get_topic_timeline
from deep_research.kb.topics import generate_topic_suggestions

router = APIRouter(prefix="/api/kb")

config: Config | None = None
kb_db: KBDatabase | None = None


async def init_kb(cfg: Config):
    """Best-effort — the app must still start if Postgres isn't running/
    configured. A missing KB means /api/kb/* routes fail per-request instead
    of the whole app failing to boot, and the research agent's kb_search
    tool/prioritize_kb toggle are simply unavailable rather than fatal."""
    global config, kb_db
    config = cfg
    try:
        kb_db = KBDatabase(cfg.kb.postgres_dsn)
        await kb_db.init()
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


@router.get("/topics/{topic_id}/claims")
async def list_claims(topic_id: str, status: str = "attached"):
    topic = await _get_topic_or_404(topic_id)
    claims = await kb_db.list_topic_claims(topic["id"], link_status=status)
    return {"claims": _serialize(claims)}


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
    result = await generate_topic_suggestions(kb_db, topic["id"])
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


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: str):
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    evidence = await kb_db.list_claim_evidence(claim_id)
    return {"claim": _serialize(claim), "evidence": _serialize(evidence)}


@router.put("/claims/{claim_id}/preferred-source")
async def set_preferred_source(claim_id: str, req: PreferredSourceRequest):
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found")
    updated = await kb_db.set_preferred_source_manual(claim_id, req.source_id, reviewed_by="web_ui")
    return {"claim": _serialize(updated)}


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
async def search_sources(q: str = "", limit: int = 50):
    sources = await kb_db.list_sources(limit=500)
    if q:
        q_lower = q.lower()
        sources = [
            s for s in sources
            if q_lower in (s.get("title") or "").lower() or q_lower in s["canonical_uri"].lower()
        ][:limit]
    else:
        sources = sources[:limit]
    return {"sources": _serialize(sources)}
