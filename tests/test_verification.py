from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from deep_research.config import Config
from deep_research.kb import verification as v


async def test_batch_verification_uses_the_dedicated_verifier_endpoint(monkeypatch):
    config = Config()
    config.kb.extraction_llm_base_url = "http://extractor/v1"
    config.kb.verification_llm_base_url = "http://verifier/v1"
    seen = []

    async def fake_detect(url):
        seen.append(url)
        return "detected-verifier"

    async def fake_verify(_db, _config, claim_id, **kwargs):
        assert kwargs["extraction_model"] == "detected-verifier"
        return SimpleNamespace(status="unverified")

    monkeypatch.setattr(v, "detect_model", fake_detect)
    monkeypatch.setattr(v, "verify_claim", fake_verify)
    outcomes = await v.verify_claims_concurrently(None, config, [{"id": "claim-1"}])
    assert seen == ["http://verifier/v1"]
    assert outcomes[0][1] == "unverified"


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


def test_budget_stops_after_an_official_weighted_corroboration():
    budget = v._Budget(max_sources=10, max_searches=10)
    budget.supports = 1
    budget.support_weight = 1.0
    assert budget.should_stop() is True


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


# -- eligibility / check_status: settled vs. inconclusive claims -------------
# A settled verdict (supported/contradicted/mixed) is never auto-rechecked.
# An "unverified" (inconclusive) first pass is a different case -- it gets a
# second look automatically once UNVERIFIED_RETRY_COOLDOWN_HOURS has passed,
# rather than being abandoned forever the moment verification_attempted_at is
# set. claim_check_status must always agree with is_claim_eligible_for_verification
# (see the docstring on claim_check_status) -- these tests check both in step.

def _claim(**overrides) -> dict:
    base = {
        "status": "unverified",
        "importance_score": 0.9,
        "verification_attempted_at": None,
        "verification_override": None,
    }
    base.update(overrides)
    return base


def test_never_attempted_claim_above_threshold_is_eligible():
    claim = _claim()
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is True
    assert v.claim_check_status(claim, threshold=0.8) == "auto_check"


def test_never_attempted_claim_below_threshold_is_not_eligible():
    claim = _claim(importance_score=0.5)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "auto_skip"


def test_settled_claim_never_auto_rechecked_even_long_after_attempt():
    old = datetime.now(timezone.utc) - timedelta(days=365)
    claim = _claim(status="supported", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "checked"


def test_contradicted_and_mixed_claims_are_also_settled():
    old = datetime.now(timezone.utc) - timedelta(days=365)
    for status in ("contradicted", "mixed"):
        claim = _claim(status=status, verification_attempted_at=old)
        assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
        assert v.claim_check_status(claim, threshold=0.8) == "checked"


def test_unverified_claim_within_cooldown_is_not_yet_eligible():
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    claim = _claim(status="unverified", verification_attempted_at=recent)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "checked_pending_retry"


def test_unverified_claim_past_cooldown_is_eligible_again():
    old = datetime.now(timezone.utc) - timedelta(hours=v.UNVERIFIED_RETRY_COOLDOWN_HOURS + 1)
    claim = _claim(status="unverified", verification_attempted_at=old)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is True
    assert v.claim_check_status(claim, threshold=0.8) == "auto_check"


def test_unverified_claim_past_cooldown_but_below_threshold_is_auto_skip():
    old = datetime.now(timezone.utc) - timedelta(hours=v.UNVERIFIED_RETRY_COOLDOWN_HOURS + 1)
    claim = _claim(status="unverified", verification_attempted_at=old, importance_score=0.5)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "auto_skip"


def test_manual_exclude_always_wins_regardless_of_attempt_state():
    claim = _claim(status="supported", verification_attempted_at=datetime.now(timezone.utc), verification_override="exclude")
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "manual_exclude"


def test_force_bypasses_the_attempted_and_cooldown_gate():
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    claim = _claim(status="unverified", verification_attempted_at=recent)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8, force=True) is True


def test_deprecated_claim_is_never_eligible_even_if_never_attempted():
    # The losing side of a claim merge -- just a pointer to the real claim
    # now, not a live fact worth a verification pass, regardless of
    # importance score or whether it was ever individually checked.
    claim = _claim(status="deprecated", verification_attempted_at=None, importance_score=0.99)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8) is False
    assert v.claim_check_status(claim, threshold=0.8) == "deprecated"


def test_deprecated_claim_is_never_eligible_even_with_force():
    claim = _claim(status="deprecated", verification_attempted_at=None, importance_score=0.99)
    assert v.is_claim_eligible_for_verification(claim, threshold=0.8, force=True) is False


# -- _classify_relationship: verification_context steers the comparison -----

async def test_classify_relationship_includes_context_when_given():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"relationship": "supports", "confidence": 0.9, "reasoning": "ok"}'}}]}

    await v._classify_relationship(
        FakeLLM(), "Industrial buildings use more electricity than residential.",
        "Datacenters use far more electricity per square foot than industrial buildings.",
        context="compare specifically against datacenter usage",
    )

    assert "compare specifically against datacenter usage" in seen["content"]


async def test_classify_relationship_omits_context_line_when_absent():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"relationship": "supports", "confidence": 0.9, "reasoning": "ok"}'}}]}

    await v._classify_relationship(FakeLLM(), "Claim A text.", "Claim B text.")

    assert "Additional context" not in seen["content"]


# -- _suggest_search_query: repeating a failed query wastes the retry --------

async def test_suggest_search_query_uses_llm_suggestion():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": "specific alternate query"}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "specific alternate query"


async def test_suggest_search_query_falls_back_to_claim_text_on_garbage_response():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "not json"}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "Some claim text."


async def test_suggest_search_query_falls_back_to_claim_text_on_empty_query():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": ""}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", ["Some claim text."])
    assert result == "Some claim text."


async def test_suggest_search_query_includes_context_in_the_prompt():
    seen = {}

    class FakeLLM:
        async def chat(self, messages):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"message": {"content": '{"query": "datacenter electricity usage vs industrial"}'}}]}

    result = await v._suggest_search_query(
        FakeLLM(), "Industrial buildings use more electricity than residential.", [],
        context="compare specifically against datacenter usage",
    )

    assert result == "datacenter electricity usage vs industrial"
    assert "compare specifically against datacenter usage" in seen["content"]


async def test_suggest_search_query_uses_llm_even_on_first_attempt_when_context_given():
    # Normally the very first search just uses the raw claim text (see
    # verify_claim) -- this only tests that _suggest_search_query itself
    # produces a sensible result when called with no tried_queries yet but
    # context present, which verify_claim does specifically to cover that case.
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"query": "context-aware query"}'}}]}

    result = await v._suggest_search_query(FakeLLM(), "Some claim text.", [], context="a specific angle")
    assert result == "context-aware query"


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

    async def flaky_classify(llm, a, b, context=None):
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


# -- social-media-only sources can't settle a verification -------------------
# Reddit/Instagram/Facebook are unvetted user-generated content -- fine to
# read, but if that's the *only* evidence a candidate claim has, it must
# never single-handedly mark the target claim supported/contradicted.

async def test_examine_candidates_skips_llm_call_for_social_media_only_source(kb_db, monkeypatch):
    target, _ = await kb_db.get_or_create_claim("fact", "Target claim for social-media test.")
    reddit_claim, _ = await kb_db.get_or_create_claim("fact", "A claim only backed by a Reddit post.")
    real_claim, _ = await kb_db.get_or_create_claim("fact", "A claim backed by a real news source.")

    reddit_source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://www.reddit.com/r/test/comments/abc", canonical_key="reddit-abc",
    )
    real_source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://www.example-news.example/article", canonical_key="real-article",
    )
    reddit_version, _ = await kb_db.add_source_version(reddit_source["id"], content_hash="h1", snapshot_path="/tmp/r", http_status=200, mime_type="text/html")
    real_version, _ = await kb_db.add_source_version(real_source["id"], content_hash="h2", snapshot_path="/tmp/n", http_status=200, mime_type="text/html")
    reddit_artifact, _ = await kb_db.upsert_artifact(artifact_id="art-reddit", source_version_id=reddit_version["id"], artifact_type="clean_text", storage_path="/tmp/r.txt", content_hash="h1", chunk_params_hash="p1")
    real_artifact, _ = await kb_db.upsert_artifact(artifact_id="art-real", source_version_id=real_version["id"], artifact_type="clean_text", storage_path="/tmp/n.txt", content_hash="h2", chunk_params_hash="p1")
    reddit_chunk = await kb_db.add_chunk(reddit_artifact["id"], 0, "reddit chunk", "chash-r")
    real_chunk = await kb_db.add_chunk(real_artifact["id"], 0, "real chunk", "chash-n")
    await kb_db.add_claim_evidence(claim_id=reddit_claim["id"], artifact_chunk_id=reddit_chunk["id"], source_id=reddit_source["id"], source_version_id=reddit_version["id"])
    await kb_db.add_claim_evidence(claim_id=real_claim["id"], artifact_chunk_id=real_chunk["id"], source_id=real_source["id"], source_version_id=real_version["id"])

    classified = []

    async def fake_classify(llm, a, b, context=None):
        classified.append(b)
        return {"relationship": "supports", "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(v, "_classify_relationship", fake_classify)

    budget = v._Budget(max_sources=10, max_searches=1)
    ranked = [(reddit_claim, 0.9), (real_claim, 0.8)]

    await v._examine_candidates(kb_db, None, None, target, ranked, budget, set(), [], supporting_ids=[])

    assert real_claim["canonical_text"] in classified
    assert reddit_claim["canonical_text"] not in classified  # never sent to the LLM at all
    assert budget.supports == 1  # only the real source's support counted
    assert budget.sources_examined == 2  # the reddit source still cost budget -- it was looked at


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


async def test_run_search_budget_is_atomic_and_bounded():
    budget = v._RunSearchBudget(2)

    assert await budget.reserve() is True
    assert await budget.reserve() is True
    assert await budget.reserve() is False
    assert budget.used == 2


async def test_run_search_budget_can_be_unbounded_for_explicit_actions():
    budget = v._RunSearchBudget(None)

    assert await budget.reserve() is True
    assert await budget.reserve() is True
    assert budget.used == 0


async def test_batch_verification_detects_the_model_once(monkeypatch):
    calls = {"detect": 0, "models": []}

    async def fake_detect(_url):
        calls["detect"] += 1
        return "shared-model"

    async def fake_verify(_db, _config, claim_id, **kwargs):
        calls["models"].append(kwargs["extraction_model"])
        return SimpleNamespace(status="supported")

    monkeypatch.setattr(v, "detect_model", fake_detect)
    monkeypatch.setattr(v, "verify_claim", fake_verify)

    outcomes = await v.verify_claims_concurrently(
        None, Config(), [{"id": "claim-1"}, {"id": "claim-2"}], concurrency=2,
    )

    assert len(outcomes) == 2
    assert calls == {"detect": 1, "models": ["shared-model", "shared-model"]}
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


# -- keep-or-delete claims discovered during web-fallback verification ------
# The bug this guards against: extracting a scraped page's top chunks can
# promote several new claims, but only the one an LLM comparison pass
# actually classifies as supports/contradicts is worth keeping. Without
# discarding the rest, verifying claim A pulls in claims B/C/D from other
# pages, and if any of them also meet the importance threshold, the next
# sweep verifies *them* too -- unbounded compounding growth with no way to
# ever finish. Found in practice: a real KB had 1500+ such claims after one
# night, almost none of them ever established as relevant to anything, all
# floating with no topic and all eligible to keep the chain going.

async def _make_source_with_evidenced_claim(kb_db, claim_id, canonical_uri):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri=canonical_uri, canonical_key=canonical_uri,
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/x", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id=f"art-{canonical_uri}", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/x.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "text", "chash")
    await kb_db.add_claim_evidence(
        claim_id=claim_id, artifact_chunk_id=chunk["id"], source_id=source["id"], source_version_id=version["id"],
    )
    return source


async def test_resolve_new_verification_claims_keeps_and_tags_the_proving_claim(kb_db):
    topic = await kb_db.create_topic("Data Centers")
    kept, _ = await kb_db.get_or_create_claim("fact", "Groundwater supply could soon come under pressure.")
    discarded, _ = await kb_db.get_or_create_claim("fact", "Tangential fact from the same scraped page.")
    source = await _make_source_with_evidenced_claim(kb_db, kept["id"], "http://kept-claim-source.example")

    await v._resolve_new_verification_claims(
        kb_db, [kept["id"], discarded["id"]], kept_ids={kept["id"]}, topics=[topic], source_id=source["id"],
    )

    kept_after = await kb_db.get_claim(kept["id"])
    assert kept_after["verification_override"] == "exclude"
    assert v.claim_check_status(kept_after, threshold=0.8) == "manual_exclude"
    linked_topics = await kb_db.get_topics_for_claim(kept["id"])
    assert [t["id"] for t in linked_topics] == [topic["id"]]

    # the source itself is tied to the topic too, not just the claim -- so
    # it shows up in context on the Sources page instead of floating
    # unexplained.
    source_topics = await kb_db.list_topic_sources(topic["id"])
    assert [s["id"] for s in source_topics] == [source["id"]]


async def test_resolve_new_verification_claims_deletes_the_non_proving_claims(kb_db):
    kept, _ = await kb_db.get_or_create_claim("fact", "The proving claim.")
    discarded, _ = await kb_db.get_or_create_claim("fact", "A tangential fact never established as relevant.")
    source = await _make_source_with_evidenced_claim(kb_db, kept["id"], "http://another-kept-claim-source.example")

    await v._resolve_new_verification_claims(
        kb_db, [kept["id"], discarded["id"]], kept_ids={kept["id"]}, topics=[], source_id=source["id"],
    )

    assert await kb_db.get_claim(kept["id"]) is not None
    assert await kb_db.get_claim(discarded["id"]) is None


# -- deleting sources that contributed nothing -------------------------------
# A page scraped/chunked/extracted purely to check one claim can end up with
# no surviving claim at all (empty page, nothing extractable, or the one
# claim it produced wasn't the prover/disprover) -- just as much dead weight
# as the discarded claims, and left unaddressed it accumulates the same way
# (187 sources found in a real KB, 166 with zero surviving claim_evidence).

async def test_source_has_claim_evidence_false_for_untouched_source(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://contributed-nothing.example", canonical_key="nothing",
    )
    assert await kb_db.source_has_claim_evidence(source["id"]) is False


async def test_delete_source_cascade_removes_source_and_its_artifacts(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://to-delete.example", canonical_key="to-delete",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/to-delete", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="art-to-delete", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/to-delete.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "some text", "chash-1")

    assert await kb_db.source_has_claim_evidence(source["id"]) is False
    await kb_db.delete_source_cascade(source["id"])

    assert await kb_db.get_source(source["id"]) is None
    assert await kb_db.list_chunks(artifact["id"]) == []


# -- claims.verification_context: expands what verify_claim looks for --------

async def test_set_claim_verification_context_sets_and_clears(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Industrial buildings use more electricity than residential.")

    updated = await kb_db.set_claim_verification_context(claim["id"], "compare against datacenter usage")
    assert updated["verification_context"] == "compare against datacenter usage"

    cleared = await kb_db.set_claim_verification_context(claim["id"], None)
    assert cleared["verification_context"] is None


async def test_set_claim_verification_context_strips_and_treats_blank_as_clear(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Some claim.")

    updated = await kb_db.set_claim_verification_context(claim["id"], "  padded context  ")
    assert updated["verification_context"] == "padded context"

    blanked = await kb_db.set_claim_verification_context(claim["id"], "   ")
    assert blanked["verification_context"] is None


async def test_supporting_claims_are_durable_first_class_evidence(kb_db):
    claim, _ = await kb_db.get_or_create_claim("fact", "Main claim")
    support, _ = await kb_db.get_or_create_claim("fact", "Independent supporting claim")

    created = await kb_db.record_claim_supports(claim["id"], [support["id"], support["id"]])

    assert len(created) == 1
    assert await kb_db.get_claim_support_ids(claim["id"]) == [support["id"]]
