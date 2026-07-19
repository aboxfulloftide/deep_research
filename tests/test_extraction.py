from deep_research.config import load_config
from deep_research.kb.extraction import (
    is_assessment_content,
    has_unresolved_subject,
    repair_lifespan_date_misattribution,
    run_extraction,
)


def test_assessment_content_detector_requires_answer_bank_structure():
    exam = """Question
36
Multiple Choice
Review Later
A) Total assets $350,000; total capital $275,000.
B) Total assets $305,000; total capital $230,000.
E) Total assets $405,000; total capital $305,000.
Correct Answer
Show Answer"""

    assert is_assessment_content(exam) is True
    assert is_assessment_content(
        "The bar exam has multiple choice questions, according to the report."
    ) is False


def test_lifespan_parenthesis_is_not_converted_to_role_tenure():
    claim = {
        "claim_text": "Benito Mussolini was the Italian dictator from 1883 to 1945",
        "supporting_quote": "Italian dictator Benito Mussolini (1883 - 1945) (centre), leading the blackshirts.",
        "entities": [{"name": "Benito Mussolini", "type": "person"}],
        "event": {
            "title": "Benito Mussolini's dictatorship",
            "date": "1883-1945",
            "date_precision": "exact",
        },
    }

    repaired = repair_lifespan_date_misattribution(claim)

    assert repaired["claim_text"] == "Benito Mussolini was the Italian dictator."
    assert repaired["event"] is None


def test_explicit_lifespan_claim_keeps_parenthetical_years():
    claim = {
        "claim_text": "Benito Mussolini lived from 1883 to 1945.",
        "supporting_quote": "Benito Mussolini (1883-1945) was an Italian politician.",
        "entities": [{"name": "Benito Mussolini", "type": "person"}],
        "event": None,
    }

    repaired = repair_lifespan_date_misattribution(claim)

    assert repaired["claim_text"] == "Benito Mussolini lived from 1883 to 1945."


def test_explicit_role_tenure_without_lifespan_parenthesis_is_unchanged():
    claim = {
        "claim_text": "The official served as prime minister from 1922 to 1943.",
        "supporting_quote": "He served as prime minister from 1922 to 1943.",
        "entities": [{"name": "The official", "type": "person"}],
        "event": {"title": "Term as prime minister", "date": "1922-1943"},
    }

    repaired = repair_lifespan_date_misattribution(claim)

    assert repaired["claim_text"] == "The official served as prime minister from 1922 to 1943."
    assert repaired["event"]["date"] == "1922-1943"


def test_subjectless_measurement_claims_are_rejected():
    assert has_unresolved_subject("The total cost had risen to 23 million.") is True
    assert has_unresolved_subject("It cost $1.4 billion") is True
    assert has_unresolved_subject("Total assets are $405,000.") is True
    assert has_unresolved_subject(
        "The total cost of the Louisiana Purchase had risen to $23 million."
    ) is False
    assert has_unresolved_subject(
        "The acquisition of United States Shoes cost $1.4 billion."
    ) is False


def test_leading_personal_pronouns_are_unresolved_subjects():
    assert has_unresolved_subject(
        "He retired with a $417 million severance package."
    ) is True
    assert has_unresolved_subject(
        "His severance package was the largest at the time."
    ) is True
    assert has_unresolved_subject(
        "Jack Welch retired with a $417 million severance package."
    ) is False


async def _make_artifact_with_chunks(kb_db, chunk_texts):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="http://scoped-extraction.example", canonical_key="scoped",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="h1", snapshot_path="/tmp/scoped", http_status=200, mime_type="text/html",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="art-scoped", source_version_id=version["id"], artifact_type="clean_text",
        storage_path="/tmp/scoped.txt", content_hash="h1", chunk_params_hash="p1",
    )
    chunks = [
        await kb_db.add_chunk(artifact["id"], i, text, f"chash-{i}")
        for i, text in enumerate(chunk_texts)
    ]
    return artifact, chunks


async def test_run_extraction_with_chunk_ids_only_processes_those_chunks(kb_db, monkeypatch):
    """The bug this guards against: verify_claim's web-fallback only wants
    extraction run on the 1-3 chunks relevant to the claim being checked, not
    an entire page -- a real page extracted in full during this session
    produced 1072 tangential claims from one source. chunk_ids must actually
    restrict which chunks get sent to the extraction LLM."""
    import deep_research.kb.extraction as extraction_module

    artifact, chunks = await _make_artifact_with_chunks(kb_db, ["chunk zero", "chunk one", "chunk two"])

    seen_chunk_texts = []

    async def fake_chat(self, messages):
        seen_chunk_texts.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)

    config = load_config()
    result = await run_extraction(kb_db, config, artifact["id"], chunk_ids=[chunks[1]["id"]])

    assert result.chunk_count == 1
    assert len(seen_chunk_texts) == 1
    assert "chunk one" in seen_chunk_texts[0]


async def test_run_extraction_skips_assessment_chunks_without_calling_llm(kb_db, monkeypatch):
    artifact, _ = await _make_artifact_with_chunks(kb_db, [
        """Question
12
Multiple Choice
A) $100,000.
B) $200,000.
C) $300,000.
Correct Answer
Show Answer""",
    ])
    calls = 0

    async def fake_chat(self, messages):
        nonlocal calls
        calls += 1
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)

    result = await run_extraction(kb_db, load_config(), artifact["id"])

    assert calls == 0
    assert result.observation_count == 0


async def test_partial_extraction_run_does_not_satisfy_full_extraction_cache(kb_db, monkeypatch):
    """A chunk-scoped (partial) run must never be mistaken later for "this
    artifact was already fully extracted" -- otherwise a real extract-source
    call could silently skip most of a page's chunks forever."""
    async def fake_chat(self, messages):
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)

    artifact, chunks = await _make_artifact_with_chunks(kb_db, ["chunk zero", "chunk one"])
    config = load_config()

    partial = await run_extraction(kb_db, config, artifact["id"], chunk_ids=[chunks[0]["id"]])
    assert partial.chunk_count == 1

    full = await run_extraction(kb_db, config, artifact["id"])
    assert full.status != "unchanged"
    assert full.chunk_count == 2
