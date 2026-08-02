from deep_research.config import load_config
from deep_research.kb import claim_quality as cq


# -- classify_claim_specificity: parsing and fail-safe behavior -------------

async def test_classify_claim_specificity_parses_true():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {
                "content": '{"specific": true, "reason": "names a precise dollar figure and year"}',
            }}]}

    specific, reason = await cq.classify_claim_specificity(
        FakeLLM(), "Military spending in 1800 was roughly $2.5 million.",
    )
    assert specific is True
    assert "dollar" in reason


async def test_classify_claim_specificity_parses_false():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {
                "content": '{"specific": false, "reason": "too generic, no identifying detail"}',
            }}]}

    specific, reason = await cq.classify_claim_specificity(FakeLLM(), "Businesses closed in 1837")
    assert specific is False
    assert "generic" in reason


async def test_classify_claim_specificity_fails_safe_on_garbage_response():
    """An unparseable classifier response must keep the claim, not delete
    it -- a false "keep" just leaves a claim unverified a little longer; a
    false "delete" destroys data with no way back."""
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": "not json"}}]}

    specific, reason = await cq.classify_claim_specificity(FakeLLM(), "Some claim.")
    assert specific is True
    assert "kept by default" in reason


async def test_classify_claim_specificity_fails_safe_on_non_bool_specific_field():
    class FakeLLM:
        async def chat(self, messages):
            return {"choices": [{"message": {"content": '{"specific": "yes", "reason": "oops"}'}}]}

    specific, reason = await cq.classify_claim_specificity(FakeLLM(), "Some claim.")
    assert specific is True
    assert "kept by default" in reason


# -- sweep_claim_specificity: scope, deletion, dry_run ----------------------

async def test_sweep_claim_specificity_only_touches_unverified_claims(kb_db, monkeypatch):
    """A claim that already settled (supported/contradicted) represents
    real completed verification work regardless of how generic its text
    reads -- the sweep must never reconsider it."""
    generic_unverified, _ = await kb_db.get_or_create_claim("fact", "Businesses closed in 1837")
    generic_supported, _ = await kb_db.get_or_create_claim("fact", "A generic-sounding but already-settled claim.")
    async with kb_db.pool.acquire() as conn:
        await conn.execute("UPDATE claims SET status = 'supported' WHERE id = $1", generic_supported["id"])

    seen_claim_ids = []

    async def fake_classify(llm, claim_text):
        seen_claim_ids.append(claim_text)
        return False, "too generic"

    monkeypatch.setattr(cq, "classify_claim_specificity", fake_classify)

    result = await cq.sweep_claim_specificity(kb_db, load_config())

    assert result["total"] == 1
    assert seen_claim_ids == ["Businesses closed in 1837"]
    remaining = await kb_db.get_claim(generic_supported["id"])
    assert remaining is not None
    assert remaining["status"] == "supported"


async def test_sweep_claim_specificity_deletes_claims_judged_not_specific(kb_db, monkeypatch):
    generic, _ = await kb_db.get_or_create_claim("fact", "Businesses closed in 1837")
    specific, _ = await kb_db.get_or_create_claim("fact", "eBay went public on 24th September 1998.")

    async def fake_classify(llm, claim_text):
        if "Businesses" in claim_text:
            return False, "too generic"
        return True, "specific IPO date"

    monkeypatch.setattr(cq, "classify_claim_specificity", fake_classify)

    result = await cq.sweep_claim_specificity(kb_db, load_config())

    assert result == {"total": 2, "dry_run": False, "kept": 1, "removed": 1, "failed": 0}
    assert await kb_db.get_claim(generic["id"]) is None
    remaining = await kb_db.get_claim(specific["id"])
    assert remaining is not None


async def test_sweep_claim_specificity_dry_run_classifies_without_deleting(kb_db, monkeypatch):
    generic, _ = await kb_db.get_or_create_claim("fact", "Businesses closed in 1837")

    async def fake_classify(llm, claim_text):
        return False, "too generic"

    monkeypatch.setattr(cq, "classify_claim_specificity", fake_classify)

    result = await cq.sweep_claim_specificity(kb_db, load_config(), dry_run=True)

    assert result == {"total": 1, "dry_run": True, "kept": 0, "removed": 1, "failed": 0}
    remaining = await kb_db.get_claim(generic["id"])
    assert remaining is not None


async def test_sweep_claim_specificity_reports_failed_without_deleting(kb_db, monkeypatch):
    claim, _ = await kb_db.get_or_create_claim("fact", "Some claim that errors during classification.")

    async def fake_classify(llm, claim_text):
        raise RuntimeError("LLM unreachable")

    monkeypatch.setattr(cq, "classify_claim_specificity", fake_classify)

    reported = []
    result = await cq.sweep_claim_specificity(
        kb_db, load_config(), on_result=lambda c, status, reason: reported.append((status, reason)),
    )

    assert result == {"total": 1, "dry_run": False, "kept": 0, "removed": 0, "failed": 1}
    assert reported == [("failed", "LLM unreachable")]
    remaining = await kb_db.get_claim(claim["id"])
    assert remaining is not None


async def test_sweep_claim_specificity_bounds_concurrency(kb_db, monkeypatch):
    claims = [
        await kb_db.get_or_create_claim("fact", f"Generic claim number {i}.")
        for i in range(6)
    ]
    in_flight = 0
    max_in_flight = 0

    async def fake_classify(llm, claim_text):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        import asyncio
        await asyncio.sleep(0.02)
        in_flight -= 1
        return True, "ok"

    monkeypatch.setattr(cq, "classify_claim_specificity", fake_classify)

    result = await cq.sweep_claim_specificity(kb_db, load_config(), concurrency=2)

    assert result["total"] == 6
    assert max_in_flight == 2
