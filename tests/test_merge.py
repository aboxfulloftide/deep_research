from datetime import datetime, timedelta, timezone

from deep_research.kb import merge as m


def _entity(id_, created_at):
    return {"id": id_, "created_at": created_at}


def _claim(id_, created_at, importance_score):
    return {"id": id_, "created_at": created_at, "importance_score": importance_score}


# -- pure winner/loser selection ---------------------------------------------

def test_pick_winner_loser_entities_older_wins():
    now = datetime.now(timezone.utc)
    older = _entity("a", now - timedelta(days=1))
    newer = _entity("b", now)
    winner, loser = m._pick_winner_loser_entities(older, newer)
    assert winner["id"] == "a"
    assert loser["id"] == "b"


def test_pick_winner_loser_entities_ties_broken_by_id():
    same_time = datetime.now(timezone.utc)
    e1 = _entity("zzz", same_time)
    e2 = _entity("aaa", same_time)
    winner, loser = m._pick_winner_loser_entities(e1, e2)
    assert winner["id"] == "aaa"  # lexicographically smaller id wins the tie
    assert loser["id"] == "zzz"


def test_pick_winner_loser_claims_higher_importance_wins_even_if_newer():
    now = datetime.now(timezone.utc)
    older_less_important = _claim("a", now - timedelta(days=1), 0.5)
    newer_more_important = _claim("b", now, 0.9)
    winner, loser = m._pick_winner_loser_claims(older_less_important, newer_more_important)
    assert winner["id"] == "b"
    assert loser["id"] == "a"


def test_pick_winner_loser_claims_ties_broken_by_created_at_then_id():
    now = datetime.now(timezone.utc)
    c1 = _claim("b", now, 0.7)
    c2 = _claim("a", now - timedelta(days=1), 0.7)
    winner, loser = m._pick_winner_loser_claims(c1, c2)
    assert winner["id"] == "a"  # same importance, older wins


# -- merge_entities / merge_claims: real DB ----------------------------------

async def test_merge_entities_tombstones_loser_and_reassigns(kb_db):
    e1, _ = await kb_db.get_or_create_entity("product", "data centers")  # older
    e2, _ = await kb_db.get_or_create_entity("product", "data center")   # newer

    result = await m.merge_entities(kb_db, e1["id"], e2["id"])

    assert result["winner_id"] == e1["id"]
    assert result["loser_id"] == e2["id"]

    loser = await kb_db.get_entity(e2["id"])
    assert loser["merged_into_entity_id"] == e1["id"]

    resolved, _ = await kb_db.get_or_create_entity("product", "data center")
    assert resolved["id"] == e1["id"]

    all_entities = await kb_db.list_entities(entity_type="product", limit=1000)
    ids = {e["id"] for e in all_entities}
    assert e2["id"] not in ids
    assert e1["id"] in ids


async def test_merge_entities_is_idempotent_when_already_merged(kb_db):
    e1, _ = await kb_db.get_or_create_entity("product", "widget")
    e2, _ = await kb_db.get_or_create_entity("product", "widgets")
    await m.merge_entities(kb_db, e1["id"], e2["id"])

    result = await m.merge_entities(kb_db, e1["id"], e2["id"])
    assert result["loser_id"] is None  # no-op: both sides already resolve to the same entity


async def test_merge_entities_follows_chain_to_ultimate_winner(kb_db):
    """If the winner of an earlier merge later becomes the loser of a
    different merge, merging into it again must land on the true final
    winner, not the intermediate one."""
    e1, _ = await kb_db.get_or_create_entity("product", "gadget")
    e2, _ = await kb_db.get_or_create_entity("product", "gadgets")
    e3, _ = await kb_db.get_or_create_entity("product", "gadgetz")

    first = await m.merge_entities(kb_db, e1["id"], e2["id"])
    assert first["winner_id"] == e1["id"]

    # Now merge e3 into e1 -- fine. But merge e2 (already a loser) into e3:
    # since e2 chains to e1, this should end up resolving e2 -> e1 and then
    # comparing e1 against e3 by their own winner/loser rule.
    second = await m.merge_entities(kb_db, e2["id"], e3["id"])
    # e2 resolves to e1 first, so this is really e1 vs e3.
    assert second["winner_id"] == e1["id"] or second["winner_id"] == e3["id"]
    # Whichever won, both e2 and e3 (or e1) must ultimately resolve to one root.
    resolved_e2, _ = await kb_db.get_or_create_entity("product", "gadgets")
    resolved_e3, _ = await kb_db.get_or_create_entity("product", "gadgetz")
    assert resolved_e2["id"] == resolved_e3["id"]


async def test_merge_claims_picks_higher_importance_and_reassigns_evidence(kb_db):
    claim_a, _ = await kb_db.get_or_create_claim("fact", "text A", importance_score=0.5)
    claim_b, _ = await kb_db.get_or_create_claim("fact", "text B", importance_score=0.9)

    source, _ = await kb_db.get_or_create_source(source_type_code="web", canonical_uri="http://x.example", canonical_key="x")
    version, _ = await kb_db.add_source_version(source["id"], content_hash="h", snapshot_path="/tmp/x")
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="art-x", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/x.txt", content_hash="h", chunk_params_hash="p",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "chunk text", "chash")
    await kb_db.add_claim_evidence(claim_id=claim_a["id"], artifact_chunk_id=chunk["id"], source_id=source["id"], source_version_id=version["id"])

    result = await m.merge_claims(kb_db, claim_a["id"], claim_b["id"])

    assert result["winner_id"] == claim_b["id"]  # higher importance_score wins
    assert result["loser_id"] == claim_a["id"]

    loser = await kb_db.get_claim(claim_a["id"])
    assert loser["status"] == "deprecated"
    assert loser["merged_into_claim_id"] == claim_b["id"]

    evidence = await kb_db.list_claim_evidence(claim_b["id"])
    assert len(evidence) == 1  # reassigned from the loser


# -- contradictions: status update, never a merge ----------------------------

async def test_apply_confirmed_contradiction_updates_status_without_merging(kb_db):
    claim_a, _ = await kb_db.get_or_create_claim("fact", "Inflation rose 3% in 2025.")
    claim_b, _ = await kb_db.get_or_create_claim("fact", "Inflation fell 1% in 2025.")

    await m.apply_confirmed_contradiction(kb_db, claim_a["id"], claim_b["id"])

    refreshed_a = await kb_db.get_claim(claim_a["id"])
    refreshed_b = await kb_db.get_claim(claim_b["id"])
    assert refreshed_a["status"] == "contradicted"
    assert refreshed_b["status"] == "contradicted"
    assert refreshed_a["merged_into_claim_id"] is None
    assert refreshed_b["merged_into_claim_id"] is None


# -- review_and_execute: dispatch by candidate_type --------------------------

async def test_review_and_execute_rejected_takes_no_action(kb_db):
    e1, _ = await kb_db.get_or_create_entity("product", "thing")
    e2, _ = await kb_db.get_or_create_entity("product", "thingy")
    candidate, _ = await kb_db.add_entity_resolution_candidate(e1["id"], e2["id"], 0.9, "trigram")

    result = await m.review_and_execute(kb_db, candidate["id"], "rejected")

    assert result.action == "rejected"
    refreshed_e2 = await kb_db.get_entity(e2["id"])
    assert refreshed_e2["merged_into_entity_id"] is None


async def test_review_and_execute_entity_duplicate_merges(kb_db):
    e1, _ = await kb_db.get_or_create_entity("product", "server")
    e2, _ = await kb_db.get_or_create_entity("product", "servers")
    candidate, _ = await kb_db.add_entity_resolution_candidate(e1["id"], e2["id"], 0.9, "substring")

    result = await m.review_and_execute(kb_db, candidate["id"], "accepted")

    assert result.action == "merged"
    assert result.candidate_type == "entity_duplicate"
    candidate_after = await kb_db.get_resolution_candidate(candidate["id"])
    assert candidate_after["status"] == "accepted"


async def test_review_and_execute_claim_contradiction_records_no_merge(kb_db):
    claim_a, _ = await kb_db.get_or_create_claim("fact", "Claim X is true.")
    claim_b, _ = await kb_db.get_or_create_claim("fact", "Claim X is false.")
    candidate, _ = await kb_db.add_claim_contradiction_candidate(claim_a["id"], claim_b["id"], 0.8, "llm_comparison")

    result = await m.review_and_execute(kb_db, candidate["id"], "accepted")

    assert result.action == "contradiction_recorded"
    refreshed_a = await kb_db.get_claim(claim_a["id"])
    assert refreshed_a["status"] == "contradicted"
    assert refreshed_a["merged_into_claim_id"] is None
