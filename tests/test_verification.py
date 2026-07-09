from deep_research.kb import verification as v


# -- _Budget: pure state machine, no I/O -------------------------------------

def test_budget_sources_and_searches_remaining():
    budget = v._Budget(max_sources=2, max_searches=1)
    assert budget.sources_remaining() is True
    assert budget.searches_remaining() is True
    budget.sources_examined = 2
    budget.web_searches_used = 1
    assert budget.sources_remaining() is False
    assert budget.searches_remaining() is False


def test_budget_stops_on_any_contradiction():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.contradicts = 1
    assert budget.should_stop() is True


def test_budget_stops_after_two_supports():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 2
    assert budget.should_stop() is True
    budget2 = v._Budget(max_sources=10, max_searches=10)
    budget2.supports = 1
    assert budget2.should_stop() is False


def test_budget_stops_when_source_budget_exhausted():
    budget = v._Budget(max_sources=1, max_searches=10)
    budget.sources_examined = 1
    assert budget.should_stop() is True


def test_budget_final_status_mixed_when_both_present():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports, budget.contradicts = 1, 1
    assert budget.final_status() == "mixed"


def test_budget_final_status_contradicted():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.contradicts = 1
    assert budget.final_status() == "contradicted"


def test_budget_final_status_supported():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 2
    assert budget.final_status() == "supported"


def test_budget_final_status_unverified_by_default():
    budget = v._Budget(max_sources=10, max_searches=10)
    assert budget.final_status() == "unverified"


# -- _examine_candidates resilience (hardening pass) -------------------------
# A transient failure classifying one candidate must not abort examination of
# the rest, and must not propagate out of _examine_candidates at all -- this
# is the exact gap the hardening pass fixed (previously this would raise all
# the way out of verify_claim, discarding every support/contradiction found
# earlier in the same run).

async def test_examine_candidates_survives_one_failing_comparison(kb_db, monkeypatch):
    target, _ = await kb_db.get_or_create_claim("fact", "Target claim for resilience test.")
    other_a, _ = await kb_db.get_or_create_claim("fact", "Other claim A.")
    other_b, _ = await kb_db.get_or_create_claim("fact", "Other claim B.")
    source_a, _ = await kb_db.get_or_create_source(source_type_code="web", canonical_uri="http://a.example", canonical_key="a")
    source_b, _ = await kb_db.get_or_create_source(source_type_code="web", canonical_uri="http://b.example", canonical_key="b")
    version_a, _ = await kb_db.add_source_version(source_a["id"], content_hash="h1", snapshot_path="/tmp/a", http_status=200, mime_type="text/html")
    version_b, _ = await kb_db.add_source_version(source_b["id"], content_hash="h2", snapshot_path="/tmp/b", http_status=200, mime_type="text/html")
    artifact_a, _ = await kb_db.upsert_artifact(artifact_id="art-a", source_version_id=version_a["id"], artifact_type="clean_text", storage_path="/tmp/a.txt", content_hash="h1", chunk_params_hash="p1")
    artifact_b, _ = await kb_db.upsert_artifact(artifact_id="art-b", source_version_id=version_b["id"], artifact_type="clean_text", storage_path="/tmp/b.txt", content_hash="h2", chunk_params_hash="p1")
    chunk_a = await kb_db.add_chunk(artifact_a["id"], 0, "chunk a", "chash-a")
    chunk_b = await kb_db.add_chunk(artifact_b["id"], 0, "chunk b", "chash-b")
    await kb_db.add_claim_evidence(claim_id=other_a["id"], artifact_chunk_id=chunk_a["id"], source_id=source_a["id"], source_version_id=version_a["id"])
    await kb_db.add_claim_evidence(claim_id=other_b["id"], artifact_chunk_id=chunk_b["id"], source_id=source_b["id"], source_version_id=version_b["id"])

    call_count = {"n": 0}

    async def flaky_classify(llm, a, b):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("simulated transient LLM failure")
        return {"relationship": "unrelated", "confidence": 0.1, "reasoning": "test"}

    monkeypatch.setattr(v, "_classify_relationship", flaky_classify)

    budget = v._Budget(max_sources=10, max_searches=1)
    examined_source_ids = set()
    contradiction_ids = []
    ranked = [(other_a, 0.9), (other_b, 0.8)]

    await v._examine_candidates(kb_db, None, None, target, ranked, budget, examined_source_ids, contradiction_ids)

    assert call_count["n"] == 2  # both candidates were attempted despite the first raising
    assert budget.sources_examined == 2  # progress wasn't lost after the failure
