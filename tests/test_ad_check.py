from deep_research.config import Config
from deep_research.kb import ad_check as a


class _FakeLLMClient:
    """Stands in for the real LLMClient check_claims_for_ads builds
    internally -- these tests monkeypatch classify_claim_is_ad itself, so
    the fake client's chat() is never actually called, only close()."""

    def __init__(self, config):
        pass

    async def close(self):
        pass


async def _fake_detect_model(base_url):
    return "fake-model"


# -- _parse_ad_check: pure function, no I/O ----------------------------------

def test_parse_ad_check_handles_plain_json():
    result = a._parse_ad_check('{"is_ad": true, "confidence": 0.9, "reasoning": "sponsor read"}')
    assert result == {"is_ad": True, "confidence": 0.9, "reasoning": "sponsor read"}


def test_parse_ad_check_extracts_json_from_surrounding_text():
    content = 'Here you go:\n{"is_ad": false, "confidence": 0.8, "reasoning": "genuine fact"}\nDone.'
    result = a._parse_ad_check(content)
    assert result["is_ad"] is False


def test_parse_ad_check_falls_back_safely_on_garbage():
    result = a._parse_ad_check("not json at all")
    assert result == {"is_ad": False, "confidence": 0.0, "reasoning": "could not parse model output"}


# -- classify_claim_is_ad: LLM call + parse ----------------------------------

async def test_classify_claim_is_ad_returns_parsed_verdict():
    class FakeLLM:
        async def chat(self, messages):
            assert "code SAVE20" in messages[1]["content"]
            return {"choices": [{"message": {"content": '{"is_ad": true, "confidence": 0.95, "reasoning": "discount code"}'}}]}

    result = await a.classify_claim_is_ad(FakeLLM(), "The sponsor offers 20% off with code SAVE20.")
    assert result["is_ad"] is True


async def test_classify_claim_is_ad_includes_surrounding_source_context():
    class FakeLLM:
        async def chat(self, messages):
            content = messages[1]["content"]
            assert "The host drinks a meal replacement shake" in content
            assert "link in the description" in content
            assert "30% off" in content
            return {"choices": [{"message": {"content": '{"is_ad": true, "confidence": 0.99, "reasoning": "sponsor CTA"}'}}]}

    result = await a.classify_claim_is_ad(
        FakeLLM(),
        "The host drinks a meal replacement shake for breakfast.",
        source_context="Use the link in the description and code TEST for 30% off.",
    )

    assert result["is_ad"] is True


# -- check_claims_for_ads: needs a real DB -----------------------------------

async def test_check_claims_for_ads_excludes_a_confident_ad_verdict(kb_db, monkeypatch):
    monkeypatch.setattr(a, "detect_model", _fake_detect_model)
    monkeypatch.setattr(a, "LLMClient", _FakeLLMClient)
    claim, _ = await kb_db.get_or_create_claim("fact", "This video is sponsored by the book from Hungry Minds.")

    async def fake_classify(llm, claim_text, source_context=None):
        return {"is_ad": True, "confidence": 0.9, "reasoning": "sponsor read"}

    monkeypatch.setattr(a, "classify_claim_is_ad", fake_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == [claim["id"]]
    refreshed = await kb_db.get_claim(claim["id"])
    assert refreshed["verification_override"] == "exclude"


async def test_check_claims_for_ads_passes_evidence_context_to_classifier(kb_db, monkeypatch):
    monkeypatch.setattr(a, "detect_model", _fake_detect_model)
    monkeypatch.setattr(a, "LLMClient", _FakeLLMClient)
    claim, _ = await kb_db.get_or_create_claim(
        "fact", "The host drinks a meal replacement shake for breakfast."
    )
    source, _ = await kb_db.get_or_create_source(
        source_type_code="youtube_video",
        canonical_uri="https://www.youtube.com/watch?v=sponsor-test",
        canonical_key="youtube:sponsor-test",
        title="Sponsor context test",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="sponsor-context-hash",
        snapshot_path="/tmp/sponsor-context", http_status=200,
        mime_type="text/plain",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="sponsor-context-artifact", source_version_id=version["id"],
        artifact_type="clean_text", storage_path="/tmp/sponsor-context.txt",
        content_hash="sponsor-context-hash", chunk_params_hash="sponsor-context-params",
    )
    quote = "The host drinks a meal replacement shake for breakfast."
    chunk_text = (
        f"{quote} It saves time in the morning. Use the link in the description "
        "and code TEST to get 30% off your first subscription."
    )
    chunk = await kb_db.add_chunk(
        artifact["id"], 0, chunk_text, "sponsor-context-chunk-hash",
    )
    await kb_db.add_claim_evidence(
        claim_id=claim["id"], artifact_chunk_id=chunk["id"], source_id=source["id"],
        source_version_id=version["id"], excerpt_text=quote,
        char_start=0, char_end=len(quote),
    )
    seen = {}

    async def fake_classify(llm, claim_text, source_context=None):
        seen["context"] = source_context
        return {"is_ad": True, "confidence": 0.99, "reasoning": "sponsor CTA"}

    monkeypatch.setattr(a, "classify_claim_is_ad", fake_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == [claim["id"]]
    assert "link in the description" in seen["context"]
    assert "30% off" in seen["context"]


async def test_check_claims_for_ads_leaves_a_genuine_claim_untouched(kb_db, monkeypatch):
    monkeypatch.setattr(a, "detect_model", _fake_detect_model)
    monkeypatch.setattr(a, "LLMClient", _FakeLLMClient)
    claim, _ = await kb_db.get_or_create_claim("fact", "The Taft-Hartley Act outlawed closed shops.")

    async def fake_classify(llm, claim_text, source_context=None):
        return {"is_ad": False, "confidence": 0.95, "reasoning": "genuine historical fact"}

    monkeypatch.setattr(a, "classify_claim_is_ad", fake_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == []
    refreshed = await kb_db.get_claim(claim["id"])
    assert refreshed["verification_override"] is None


async def test_check_claims_for_ads_leaves_claim_untouched_on_low_confidence(kb_db, monkeypatch):
    monkeypatch.setattr(a, "detect_model", _fake_detect_model)
    monkeypatch.setattr(a, "LLMClient", _FakeLLMClient)
    claim, _ = await kb_db.get_or_create_claim("fact", "This might be a sponsor mention, unclear.")

    async def fake_classify(llm, claim_text, source_context=None):
        return {"is_ad": True, "confidence": 0.4, "reasoning": "not sure"}

    monkeypatch.setattr(a, "classify_claim_is_ad", fake_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == []
    refreshed = await kb_db.get_claim(claim["id"])
    assert refreshed["verification_override"] is None


async def test_check_claims_for_ads_leaves_claim_untouched_when_llm_call_raises(kb_db, monkeypatch):
    monkeypatch.setattr(a, "detect_model", _fake_detect_model)
    monkeypatch.setattr(a, "LLMClient", _FakeLLMClient)
    claim, _ = await kb_db.get_or_create_claim("fact", "Some claim text.")

    async def raising_classify(llm, claim_text, source_context=None):
        raise ConnectionError("simulated transient LLM failure")

    monkeypatch.setattr(a, "classify_claim_is_ad", raising_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == []
    refreshed = await kb_db.get_claim(claim["id"])
    assert refreshed["verification_override"] is None


async def test_check_claims_for_ads_skips_a_claim_that_already_has_an_override(kb_db, monkeypatch):
    claim, _ = await kb_db.get_or_create_claim("fact", "A claim a human already marked include.")
    await kb_db.set_claim_verification_override(claim["id"], "include")

    calls = {"n": 0}

    async def fake_classify(llm, claim_text, source_context=None):
        calls["n"] += 1
        return {"is_ad": True, "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(a, "classify_claim_is_ad", fake_classify)

    flagged = await a.check_claims_for_ads(kb_db, Config(), [claim["id"]])

    assert flagged == []
    assert calls["n"] == 0  # never even called the LLM -- already has an override
    refreshed = await kb_db.get_claim(claim["id"])
    assert refreshed["verification_override"] == "include"


async def test_check_claims_for_ads_returns_empty_for_empty_input(kb_db):
    flagged = await a.check_claims_for_ads(kb_db, Config(), [])
    assert flagged == []


async def test_delete_claim_cascade_removes_review_and_support_edges(kb_db):
    bad_claim, _ = await kb_db.get_or_create_claim("fact", "A slipped sponsor claim.")
    other_claim, _ = await kb_db.get_or_create_claim("fact", "A genuine comparison claim.")
    candidate, _ = await kb_db.add_claim_contradiction_candidate(
        bad_claim["id"], other_claim["id"], 0.9, "llm_comparison",
    )
    await kb_db.record_claim_supports(bad_claim["id"], [other_claim["id"]])
    await kb_db.record_claim_supports(other_claim["id"], [bad_claim["id"]])

    await kb_db.delete_claim_cascade(bad_claim["id"])

    assert await kb_db.get_claim(bad_claim["id"]) is None
    assert await kb_db.get_resolution_candidate(candidate["id"]) is None
    assert await kb_db.get_claim_support_ids(other_claim["id"]) == []
