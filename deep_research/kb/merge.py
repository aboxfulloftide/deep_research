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
from deep_research.kb.decision_log import record_decision


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


async def merge_entities(
    kb_db: KBDatabase, entity_a_id: str, entity_b_id: str, *, automation: dict | None = None,
) -> dict:
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
    if automation:
        await record_decision(
            kb_db, "entity_merge", "entity", loser["id"], f"merged into {winner['id']}",
            related_ids=[winner["id"]], confidence=automation.get("confidence"),
            reasoning=automation.get("reasoning"), model=automation.get("model"),
            parse_success=automation.get("parse_success"),
            previous_state={"name": loser["name"], "merged_into_entity_id": loser.get("merged_into_entity_id")},
            resulting_state={"merged_into_entity_id": winner["id"]}, reversible=False,
        )
    return {"winner_id": winner["id"], "loser_id": loser["id"]}


async def merge_claims(
    kb_db: KBDatabase, claim_a_id: str, claim_b_id: str, *, automation: dict | None = None,
) -> dict:
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
    if automation:
        await record_decision(
            kb_db, "claim_merge", "claim", loser["id"], f"merged into {winner['id']}",
            related_ids=[winner["id"]], confidence=automation.get("confidence"),
            reasoning=automation.get("reasoning"), model=automation.get("model"),
            parse_success=automation.get("parse_success"),
            previous_state={"canonical_text": loser["canonical_text"], "status": loser["status"]},
            resulting_state={"merged_into_claim_id": winner["id"], "status": "deprecated"}, reversible=False,
        )
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
        has_support = bool(await kb_db.get_claim_support_ids(cid))
        new_status = "mixed" if claim["status"] in {"supported", "mixed"} or has_support else "contradicted"
        notes = dict(claim.get("verification_notes") or {})
        notes["contradiction_confirmed_with"] = other_id
        await kb_db.update_claim_verification(cid, new_status, notes)


async def reconcile_claim_after_rejected_contradiction(kb_db: KBDatabase, claim_id: str) -> None:
    """Remove a rejected conflict from a claim's derived verification state.

    Verification marks a claim contradicted as soon as it queues a proposed
    conflict. If a reviewer later rejects that proposal, leaving the status
    untouched makes the review decision cosmetic. Rebuild the contradiction
    portion from still-open or accepted candidates while preserving any
    independently recorded support.
    """
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        return
    contradiction_rows = await kb_db.get_claim_contradictions(claim_id)
    active_rows = [
        row for row in contradiction_rows
        if row["candidate_status"] in {"open", "accepted"}
    ]
    active_other_ids = [row["other_claim_id"] for row in active_rows]
    accepted_other_ids = [
        row["other_claim_id"] for row in active_rows
        if row["candidate_status"] == "accepted"
    ]
    has_support = bool(await kb_db.get_claim_support_ids(claim_id))
    was_supported = claim["status"] in {"supported", "mixed"}
    if active_rows:
        new_status = "mixed" if has_support or was_supported else "contradicted"
    else:
        new_status = "supported" if has_support or was_supported else "unverified"

    notes = dict(claim.get("verification_notes") or {})
    notes["contradicts_found"] = len(active_rows)
    notes["contradicting_claim_ids"] = active_other_ids
    if accepted_other_ids:
        notes["contradiction_confirmed_with"] = accepted_other_ids[0]
    else:
        notes.pop("contradiction_confirmed_with", None)
    await kb_db.update_claim_verification(claim_id, new_status, notes)


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
        if candidate["candidate_type"] == "claim_contradiction":
            await reconcile_claim_after_rejected_contradiction(kb_db, candidate["left_claim_id"])
            await reconcile_claim_after_rejected_contradiction(kb_db, candidate["right_claim_id"])
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
