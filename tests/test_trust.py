from deep_research.config import Config
from deep_research.kb import trust as t


class _FakeLLMClient:
    """Stands in for the real LLMClient set_trust_tier_if_missing builds
    internally -- these tests monkeypatch classify_source_trust_tier itself,
    so the fake client's chat() is never actually called, only close()."""

    def __init__(self, config):
        pass

    async def close(self):
        pass


async def _fake_detect_model(base_url):
    return "fake-model"


# -- _parse_trust_tier_classification: pure function, no I/O -----------------

def test_parse_trust_tier_classification_handles_plain_json():
    result = t._parse_trust_tier_classification('{"tier": "official", "confidence": 0.9, "reasoning": "ok"}')
    assert result == {"tier": "official", "confidence": 0.9, "reasoning": "ok"}


def test_parse_trust_tier_classification_extracts_json_from_surrounding_text():
    content = 'Here you go:\n{"tier": "user_generated", "confidence": 0.8, "reasoning": "ok"}\nDone.'
    result = t._parse_trust_tier_classification(content)
    assert result["tier"] == "user_generated"


def test_parse_trust_tier_classification_falls_back_safely_on_garbage():
    result = t._parse_trust_tier_classification("not json at all")
    assert result == {"tier": None, "confidence": 0.0, "reasoning": "could not parse model output"}


# -- classify_source_trust_tier: LLM call + parse -----------------------------

async def test_classify_source_trust_tier_returns_parsed_verdict():
    class FakeLLM:
        async def chat(self, messages):
            assert "nytimes.com" in messages[1]["content"]
            return {"choices": [{"message": {"content": '{"tier": "reputable_reporting", "confidence": 0.9, "reasoning": "major outlet"}'}}]}

    result = await t.classify_source_trust_tier(FakeLLM(), "https://www.nytimes.com/some-article", "Some Article")
    assert result["tier"] == "reputable_reporting"


# -- set_trust_tier_if_missing: needs a real DB -------------------------------

async def test_set_trust_tier_if_missing_sets_a_confident_verdict(kb_db, monkeypatch):
    monkeypatch.setattr(t, "detect_model", _fake_detect_model)
    monkeypatch.setattr(t, "LLMClient", _FakeLLMClient)
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://www.reuters.com/x", canonical_key="reuters:x",
        title="Some Reuters Story",
    )

    async def fake_classify(llm, canonical_uri, title):
        return {"tier": "reputable_reporting", "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(t, "classify_source_trust_tier", fake_classify)

    tier = await t.set_trust_tier_if_missing(kb_db, Config(), source["id"])

    assert tier == "reputable_reporting"
    refreshed = await kb_db.get_source(source["id"])
    assert refreshed["trust_tier_id"] is not None


async def test_set_trust_tier_if_missing_leaves_untiered_on_low_confidence(kb_db, monkeypatch):
    monkeypatch.setattr(t, "detect_model", _fake_detect_model)
    monkeypatch.setattr(t, "LLMClient", _FakeLLMClient)
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://ambiguous.example/x", canonical_key="ambiguous:x",
    )

    async def fake_classify(llm, canonical_uri, title):
        return {"tier": "reputable_reporting", "confidence": 0.2, "reasoning": "not sure"}

    monkeypatch.setattr(t, "classify_source_trust_tier", fake_classify)

    tier = await t.set_trust_tier_if_missing(kb_db, Config(), source["id"])

    assert tier is None
    refreshed = await kb_db.get_source(source["id"])
    assert refreshed["trust_tier_id"] is None


async def test_set_trust_tier_if_missing_leaves_untiered_when_llm_call_raises(kb_db, monkeypatch):
    monkeypatch.setattr(t, "detect_model", _fake_detect_model)
    monkeypatch.setattr(t, "LLMClient", _FakeLLMClient)
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/x", canonical_key="example:x",
    )

    async def raising_classify(llm, canonical_uri, title):
        raise ConnectionError("simulated transient LLM failure")

    monkeypatch.setattr(t, "classify_source_trust_tier", raising_classify)

    tier = await t.set_trust_tier_if_missing(kb_db, Config(), source["id"])

    assert tier is None
    refreshed = await kb_db.get_source(source["id"])
    assert refreshed["trust_tier_id"] is None


async def test_set_trust_tier_if_missing_skips_a_source_that_already_has_a_tier(kb_db, monkeypatch):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/y", canonical_key="example:y",
        trust_tier_code="official",
    )

    calls = {"n": 0}

    async def fake_classify(llm, canonical_uri, title):
        calls["n"] += 1
        return {"tier": "user_generated", "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(t, "classify_source_trust_tier", fake_classify)

    tier = await t.set_trust_tier_if_missing(kb_db, Config(), source["id"])

    assert tier is None
    assert calls["n"] == 0  # never even called the LLM -- already tiered
