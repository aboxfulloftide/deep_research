"""Find one strong competing account for a settled, supported claim.

This is intentionally separate from verification. It never changes the
claim's status or creates a human contradiction review item; it gives a
reader a bounded "Counter-view (for balance)" annotation instead.
"""

from datetime import datetime, timedelta, timezone

from deep_research.config import Config, LLMConfig
from deep_research.kb.canonical import is_social_media_domain
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import detect_model
from deep_research.kb.verification import _classify_relationship, _rank_candidates_by_similarity
from deep_research.llm import LLMClient

COUNTER_CLAIM_COOLDOWN_DAYS = 7
COUNTER_CANDIDATE_LIMIT = 6


async def find_strongest_counter_claim(kb_db: KBDatabase, config: Config, claim_id: str, *, force: bool = False) -> dict:
    claim = await kb_db.get_claim(claim_id)
    if not claim:
        raise ValueError(f"No such claim: {claim_id}")
    if claim["status"] != "supported" and not force:
        raise ValueError("Counter-evidence search is only automatic for supported claims")
    checked = claim.get("counter_claim_checked_at")
    if checked and not force and checked > datetime.now(timezone.utc) - timedelta(days=COUNTER_CLAIM_COOLDOWN_DAYS):
        return {"status": "skipped", "reason": "counter evidence was checked recently"}
    own_sources = await kb_db.get_claim_source_ids(claim_id)
    candidates = await kb_db.get_claims_independent_of(list(own_sources), claim_id)
    ranked = await _rank_candidates_by_similarity(config, claim, candidates)
    base_url = config.kb.verification_llm_base_url or config.kb.extraction_llm_base_url
    model = config.kb.verification_llm_model or config.kb.extraction_llm_model or await detect_model(base_url)
    llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))
    strongest: tuple[dict, float, dict] | None = None
    try:
        for other, similarity in ranked[:COUNTER_CANDIDATE_LIMIT]:
            sources = [await kb_db.get_source(sid) for sid in await kb_db.get_claim_source_ids(other["id"])]
            if sources and all(s and is_social_media_domain(s["canonical_uri"]) for s in sources):
                continue
            try:
                verdict = await _classify_relationship(llm, claim["canonical_text"], other["canonical_text"], claim.get("verification_context"))
            except Exception:
                continue
            if verdict.get("relationship") == "contradicts" and (strongest is None or verdict.get("confidence", 0) > strongest[2].get("confidence", 0)):
                strongest = (other, similarity, verdict)
    finally:
        await llm.close()
    await kb_db.mark_counter_claim_checked(claim_id)
    if not strongest:
        return {"status": "no_counter_evidence"}
    other, similarity, verdict = strongest
    candidate, _ = await kb_db.add_counter_evidence_candidate(
        claim_id, other["id"], verdict.get("confidence") or similarity, "counter_evidence_llm", verdict.get("reasoning"),
    )
    await kb_db.record_decision(
        "counter_evidence", "claim", claim_id, f"counter-view recorded from {other['id']}",
        related_ids=[other["id"], candidate["id"]], confidence=verdict.get("confidence"),
        reasoning=verdict.get("reasoning"), model=model, parse_success=True, reversible=False,
    )
    return {"status": "found", "candidate_id": candidate["id"], "other_claim_id": other["id"]}
