from deep_research.kb import topics as tp


# -- _parse_relevance_classification: pure function, no I/O ------------------

def test_parse_relevance_classification_handles_plain_json():
    result = tp._parse_relevance_classification('{"relationship": "relevant", "confidence": 0.9, "reasoning": "ok"}')
    assert result == {"relationship": "relevant", "confidence": 0.9, "reasoning": "ok"}


def test_parse_relevance_classification_extracts_json_from_surrounding_text():
    content = 'Sure:\n{"relationship": "not_relevant", "confidence": 0.9, "reasoning": "ok"}\nDone.'
    result = tp._parse_relevance_classification(content)
    assert result["relationship"] == "not_relevant"


def test_parse_relevance_classification_falls_back_to_relevant_on_garbage():
    # Unlike entity/claim-duplicate parsing (safe default "different"), the
    # safe default here is "relevant" -- an unparseable response must never
    # cause a suggestion to be silently dropped.
    result = tp._parse_relevance_classification("not json at all")
    assert result == {"relationship": "relevant", "confidence": 0.0, "reasoning": "could not parse model output"}


# -- _suggest_claims_and_sources: needs a real DB -----------------------------

async def test_suggest_claims_and_sources_without_llm_preserves_old_behavior(kb_db):
    topic = await kb_db.create_topic("Data Centers")
    claim, _ = await kb_db.get_or_create_claim("fact", "Nvidia GPUs power most modern data centers.")

    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])])

    assert result.claims_suggested == 1
    candidates = await kb_db.list_topic_claims(topic["id"], link_status="suggested")
    assert [c["id"] for c in candidates] == [claim["id"]]


async def test_suggest_claims_and_sources_skips_confidently_not_relevant_claim(kb_db, monkeypatch):
    topic = await kb_db.create_topic("Data Centers")
    claim, _ = await kb_db.get_or_create_claim("fact", "Nvidia's CEO gave a speech about immigration policy.")

    async def fake_classify(llm, topic_name, topic_description, claim_text, matched_names):
        return {"relationship": "not_relevant", "confidence": 0.95, "reasoning": "test"}

    monkeypatch.setattr(tp, "_classify_claim_topic_relevance", fake_classify)

    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])], llm=object())

    assert result.claims_suggested == 0
    candidates = await kb_db.list_topic_claims(topic["id"], link_status="suggested")
    assert candidates == []


async def test_suggest_claims_and_sources_keeps_relevant_claim(kb_db, monkeypatch):
    topic = await kb_db.create_topic("Data Centers")
    claim, _ = await kb_db.get_or_create_claim("fact", "Nvidia GPUs power most modern data centers.")

    async def fake_classify(llm, topic_name, topic_description, claim_text, matched_names):
        return {"relationship": "relevant", "confidence": 0.8, "reasoning": "test"}

    monkeypatch.setattr(tp, "_classify_claim_topic_relevance", fake_classify)

    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])], llm=object())

    assert result.claims_suggested == 1


async def test_suggest_claims_and_sources_auto_attaches_high_confidence_relevance(kb_db, monkeypatch):
    topic = await kb_db.create_topic("Nvidia")
    claim, _ = await kb_db.get_or_create_claim("fact", "Nvidia announced a new product.")

    async def fake_classify(*args, **kwargs):
        return {"relationship": "relevant", "confidence": 0.95, "reasoning": "Directly about the topic."}

    monkeypatch.setattr(tp, "_classify_claim_topic_relevance", fake_classify)
    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])], llm=object())

    assert result.claims_auto_attached == 1
    assert result.claims_suggested == 0
    assert [row["id"] for row in await kb_db.list_topic_claims(topic["id"])] == [claim["id"]]
    decisions = await kb_db.list_decisions(subject_type="claim", subject_id=claim["id"])
    assert decisions[0]["decision_type"] == "topic_auto_attach"
    assert decisions[0]["reversible"] is True


async def test_suggest_claims_and_sources_keeps_claim_on_low_confidence_not_relevant(kb_db, monkeypatch):
    # Bias toward keeping when unsure -- a false suppression here is a
    # silent, unrecoverable loss (see TOPIC_RELEVANCE_SUPPRESS_THRESHOLD).
    topic = await kb_db.create_topic("Data Centers")
    claim, _ = await kb_db.get_or_create_claim("fact", "Ambiguous claim mentioning Nvidia.")

    async def fake_classify(llm, topic_name, topic_description, claim_text, matched_names):
        return {"relationship": "not_relevant", "confidence": 0.5, "reasoning": "not sure"}

    monkeypatch.setattr(tp, "_classify_claim_topic_relevance", fake_classify)

    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])], llm=object())

    assert result.claims_suggested == 1


async def test_suggest_claims_and_sources_keeps_claim_when_llm_call_raises(kb_db, monkeypatch):
    topic = await kb_db.create_topic("Data Centers")
    claim, _ = await kb_db.get_or_create_claim("fact", "A claim mentioning Nvidia.")

    async def raising_classify(llm, topic_name, topic_description, claim_text, matched_names):
        raise ConnectionError("simulated transient LLM failure")

    monkeypatch.setattr(tp, "_classify_claim_topic_relevance", raising_classify)

    result = await tp._suggest_claims_and_sources(kb_db, topic["id"], [(claim, 1.0, ["nvidia"])], llm=object())

    assert result.claims_suggested == 1  # a broken LLM call must never block the safe fallback
