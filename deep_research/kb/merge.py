"""Merge execution for accepted resolution_candidates (build order step 4/6/7
follow-up). Accepting an entity_duplicate/claim_duplicate candidate previously
only flipped a status flag — this executes the actual merge: reassigning
evidence/metrics/topic links to one canonical "winner" row and tombstoning
the "loser" (never deleted outright, so evidence/audit trail is preserved —
claims reuse the existing 'deprecated' status from the Claim Status Model,
entities get a parallel merged_into_entity_id pointer).

Contradictions are a different action entirely: accepting a claim_contradiction
candidate is a human confirming two claims genuinely conflict, not that they're
the same fact — no merge happens, just claims.status update on both sides.
"""

from dataclasses import dataclass

from deep_research.kb.db import KBDatabase


def _pick_winner_loser_entities(entity_a: dict, entity_b: dict) -> tuple[dict, dict]:
    """Winner = older (first-seen) entity, ties broken by id for determinism."""
    ordered = sorted([entity_a, entity_b], key=lambda e: (e["created_at"], e["id"]))
    return ordered[0], ordered[1]  # winner, loser


def _pick_winner_loser_claims(claim_a: dict, claim_b: dict) -> tuple[dict, dict]:
    """Winner = higher importance_score, ties broken by older created_at then id."""

    def sort_key(c):
        return (-(c["importance_score"] or 0), c["created_at"], c["id"])

    ordered = sorted([claim_a, claim_b], key=sort_key)
    return ordered[0], ordered[1]


async def _resolve_ultimate_entity(kb_db: KBDatabase, entity: dict) -> dict:
    """Follows merged_into_entity_id chains to the final, non-merged entity —
    guards against a rare case where the "winner" of a prior merge later
    became a "loser" of a different merge."""
    seen = {entity["id"]}
    current = entity
    while current.get("merged_into_entity_id"):
        next_id = current["merged_into_entity_id"]
        if next_id in seen:
            break  # defensive: never loop forever on a corrupt chain
        seen.add(next_id)
        nxt = await kb_db.get_entity(next_id)
        if nxt is None:
            break
        current = nxt
    return current


async def _resolve_ultimate_claim(kb_db: KBDatabase, claim: dict) -> dict:
    seen = {claim["id"]}
    current = claim
    while current.get("merged_into_claim_id"):
        next_id = current["merged_into_claim_id"]
        if next_id in seen:
            break
        seen.add(next_id)
        nxt = await kb_db.get_claim(next_id)
        if nxt is None:
            break
        current = nxt
    return current


async def merge_entities(kb_db: KBDatabase, entity_a_id: str, entity_b_id: str) -> dict:
    """Merges two duplicate entities. Returns {"winner_id", "loser_id"}."""
    entity_a = await kb_db.get_entity(entity_a_id)
    entity_b = await kb_db.get_entity(entity_b_id)
    if entity_a is None or entity_b is None:
        raise ValueError("Both entities must exist to merge")

    entity_a = await _resolve_ultimate_entity(kb_db, entity_a)
    entity_b = await _resolve_ultimate_entity(kb_db, entity_b)
    if entity_a["id"] == entity_b["id"]:
        return {"winner_id": entity_a["id"], "loser_id": None}  # already merged together

    winner, loser = _pick_winner_loser_entities(entity_a, entity_b)
    await kb_db.reassign_metrics_entity(loser["id"], winner["id"])
    await kb_db.reassign_resolution_candidates_entity(loser["id"], winner["id"])
    await kb_db.mark_entity_merged(loser["id"], winner["id"])
    return {"winner_id": winner["id"], "loser_id": loser["id"]}


async def merge_claims(kb_db: KBDatabase, claim_a_id: str, claim_b_id: str) -> dict:
    claim_a = await kb_db.get_claim(claim_a_id)
    claim_b = await kb_db.get_claim(claim_b_id)
    if claim_a is None or claim_b is None:
        raise ValueError("Both claims must exist to merge")

    claim_a = await _resolve_ultimate_claim(kb_db, claim_a)
    claim_b = await _resolve_ultimate_claim(kb_db, claim_b)
    if claim_a["id"] == claim_b["id"]:
        return {"winner_id": claim_a["id"], "loser_id": None}

    winner, loser = _pick_winner_loser_claims(claim_a, claim_b)
    await kb_db.reassign_claim_evidence(loser["id"], winner["id"])
    await kb_db.reassign_metrics_claim(loser["id"], winner["id"])
    await kb_db.reassign_claim_topics(loser["id"], winner["id"])
    await kb_db.reassign_resolution_candidates_claim(loser["id"], winner["id"])
    await kb_db.reassign_observations_claim(loser["id"], winner["id"])
    await kb_db.mark_claim_merged(loser["id"], winner["id"])
    # After merging, the winner has more evidence sources — refresh preferred_source_id
    await kb_db.recompute_preferred_source(winner["id"])
    return {"winner_id": winner["id"], "loser_id": loser["id"]}


async def apply_confirmed_contradiction(kb_db: KBDatabase, claim_a_id: str, claim_b_id: str) -> None:
    """A human confirmed these two claims genuinely conflict — update both
    claims' status (mixed if either already has independent support,
    contradicted otherwise). No merge — contradicting claims are not the
    same fact."""
    for cid in (claim_a_id, claim_b_id):
        claim = await kb_db.get_claim(cid)
        if claim is None:
            continue
        other_id = claim_b_id if cid == claim_a_id else claim_a_id
        new_status = "mixed" if claim["status"] == "supported" else "contradicted"
        notes = dict(claim.get("verification_notes") or {})
        notes["contradiction_confirmed_with"] = other_id
        await kb_db.update_claim_verification(cid, new_status, notes)


@dataclass
class ReviewResult:
    candidate_id: str
    decision: str
    candidate_type: str
    action: str  # "merged" | "contradiction_recorded" | "rejected" | "no_op_already_merged"
    winner_id: str | None = None
    loser_id: str | None = None


async def review_and_execute(
    kb_db: KBDatabase, candidate_id: str, decision: str, reviewed_by: str | None = None,
) -> ReviewResult:
    candidate = await kb_db.get_resolution_candidate(candidate_id)
    if candidate is None:
        raise ValueError(f"No such resolution candidate: {candidate_id}")

    await kb_db.review_resolution_candidate(candidate_id, decision, reviewed_by)

    if decision != "accepted":
        return ReviewResult(candidate_id, decision, candidate["candidate_type"], action="rejected")

    ctype = candidate["candidate_type"]
    if ctype == "entity_duplicate":
        result = await merge_entities(kb_db, candidate["left_entity_id"], candidate["right_entity_id"])
        action = "merged" if result["loser_id"] else "no_op_already_merged"
        return ReviewResult(candidate_id, decision, ctype, action=action, **result)
    elif ctype == "claim_duplicate":
        result = await merge_claims(kb_db, candidate["left_claim_id"], candidate["right_claim_id"])
        action = "merged" if result["loser_id"] else "no_op_already_merged"
        return ReviewResult(candidate_id, decision, ctype, action=action, **result)
    elif ctype == "claim_contradiction":
        await apply_confirmed_contradiction(kb_db, candidate["left_claim_id"], candidate["right_claim_id"])
        return ReviewResult(candidate_id, decision, ctype, action="contradiction_recorded")
    else:
        return ReviewResult(candidate_id, decision, ctype, action="unknown_type")
