from deep_research.kb import verification as v


# -- _Budget: pure state machine, no I/O -------------------------------------

def test_budget_sources_and_searches_remaining():
    budget = v._Budget(max_sources=2, max_searches=1)
    assert budget.sources_remaining("internal") is True
    assert budget.sources_remaining("external") is True
    assert budget.searches_remaining() is True
    budget.record_source_examined("internal")
    budget.record_source_examined("internal")
    budget.web_searches_used = 1
    assert budget.sources_remaining("internal") is False
    assert budget.searches_remaining() is False


def test_budget_source_phases_are_independent():
    """The bug this guards against: a claim with a few weak internal matches
    burning the whole "sources examined" budget in phase 1 and leaving
    nothing for the web fallback in phase 2. Exhausting one phase's budget
    must not affect the other's."""
    budget = v._Budget(max_sources=1, max_searches=10)
    budget.record_source_examined("internal")
    assert budget.sources_remaining("internal") is False
    assert budget.sources_remaining("external") is True
    budget.record_source_examined("external")
    assert budget.sources_remaining("external") is False
    assert budget.sources_examined == 2


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


def test_budget_does_not_stop_on_source_budget_exhaustion_alone():
    """Exhausting the *internal* source budget must not itself trigger
    should_stop() -- that's what used to block the web fallback from ever
    being tried. Only a contradiction or 2 supports should stop things;
    running out of sources in a given phase is enforced separately via
    sources_remaining(phase), so the other phase still gets its own budget."""
    budget = v._Budget(max_sources=1, max_searches=10)
    budget.record_source_examined("internal")
    assert budget.should_stop() is False


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


async def test_rank_candidates_degrades_gracefully_when_embedding_backend_down(kb_db, monkeypatch):
    """Real bug found while verifying the web UI's verify-claim route with
    Ollama stopped: an unreachable embedding backend raised all the way out
    of verify_claim as an uncaught httpx.ConnectError, discarding the whole
    verification attempt. Every other embed_texts call site in the codebase
    is best-effort; this one should be too."""
    from deep_research.config import load_config

    claim, _ = await kb_db.get_or_create_claim("fact", "Target claim with no persisted embedding.")
    other, _ = await kb_db.get_or_create_claim("fact", "Candidate claim with no persisted embedding either.")

    async def failing_embed_texts(*args, **kwargs):
        raise ConnectionError("simulated Ollama outage")

    monkeypatch.setattr(v, "embed_texts", failing_embed_texts)

    config = load_config()
    ranked = await v._rank_candidates_by_similarity(config, claim, [other])

    assert ranked == []  # degraded to "nothing ranked", not an exception


# -- run_verification_sweep concurrency guard --------------------------------
# verify_claim makes real LLM calls against a single shared GPU (the machine
# this runs on has one, with a second coming later) -- a second sweep starting
# while one is already in progress would double up GPU load for no benefit,
# which is exactly what happened once in practice: the nightly cron fired
# while an orphaned manual-trigger run (from a killed dev server) was still
# marked "running" in the database.

async def test_run_verification_sweep_refuses_concurrent_run(kb_db):
    import pytest

    await kb_db.create_verification_run("manual", claims_total=1)

    with pytest.raises(RuntimeError, match="already in progress"):
        await v.run_verification_sweep(kb_db, None, trigger="cron")


async def test_run_verification_sweep_treats_old_running_run_as_abandoned(kb_db):
    """A "running" row older than the cron job's own 8h timeout can only mean
    the process that owned it died without marking it complete -- it must not
    block new sweeps forever."""
    from deep_research.config import load_config

    stale = await kb_db.create_verification_run("cron", claims_total=1)
    async with kb_db.pool.acquire() as conn:
        await conn.execute(
            "UPDATE verification_runs SET started_at = started_at - INTERVAL '10 hours' WHERE id = $1",
            stale["id"],
        )

    config = load_config()
    summary = await v.run_verification_sweep(kb_db, config, trigger="cron", limit=0)

    stale_after = await kb_db.list_verification_runs(limit=10)
    stale_row = next(r for r in stale_after if r["id"] == stale["id"])
    assert stale_row["status"] == "failed"
    assert "Abandoned" in stale_row["error_message"]
    assert summary["eligible_count"] == 0
