from deep_research.config import load_config
from deep_research.kb.extraction import (
    is_assessment_content,
    has_unresolved_subject,
    propagate_named_events,
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


# -- propagate_named_events: mechanical fix for a prompt that didn't stick --
# Live-confirmed twice: told to name a chunk's specific event instead of a
# bare year, the model reliably extracts the naming sentence itself
# ("This crisis became known as the Panic of 1837") as its own claim, but
# does not reliably fold that name into sibling claims about the same year
# ("Businesses closed in 1837"), even after two rounds of strengthening the
# prompt's wording. This repairs it deterministically after the fact.

def test_propagate_named_events_appends_name_to_sibling_claims():
    claims = [
        {"claim_text": "Businesses closed in 1837."},
        {"claim_text": "Workers lost jobs in 1837."},
        {"claim_text": "This crisis became known as the Panic of 1837."},
    ]

    result = propagate_named_events(claims)

    assert result[0]["claim_text"] == "Businesses closed in 1837 (the Panic of 1837)."
    assert result[1]["claim_text"] == "Workers lost jobs in 1837 (the Panic of 1837)."
    assert result[2]["claim_text"] == "This crisis became known as the Panic of 1837."


def test_propagate_named_events_preserves_names_without_the_article():
    claims = [
        {"claim_text": "Stock prices fell sharply on October 19, 1987."},
        {"claim_text": "The 1987 crash became known as Black Monday."},
    ]

    result = propagate_named_events(claims)

    assert "(Black Monday)" in result[0]["claim_text"]
    assert "(the Black Monday)" not in result[0]["claim_text"]


def test_propagate_named_events_does_not_duplicate_an_already_present_name():
    claims = [
        {"claim_text": "Businesses closed during the Panic of 1837."},
        {"claim_text": "This crisis became known as the Panic of 1837."},
    ]

    result = propagate_named_events(claims)

    assert result[0]["claim_text"] == "Businesses closed during the Panic of 1837."


def test_propagate_named_events_only_applies_to_the_matching_year():
    claims = [
        {"claim_text": "Businesses closed in 1837."},
        {"claim_text": "A different recession happened in 1929."},
        {"claim_text": "This crisis became known as the Panic of 1837."},
    ]

    result = propagate_named_events(claims)

    assert "(the Panic of 1837)" in result[0]["claim_text"]
    assert result[1]["claim_text"] == "A different recession happened in 1929."


def test_propagate_named_events_is_a_noop_without_any_naming_claim():
    claims = [
        {"claim_text": "Businesses closed in 1837."},
        {"claim_text": "Workers lost jobs in 1837."},
    ]

    result = propagate_named_events(claims)

    assert result[0]["claim_text"] == "Businesses closed in 1837."
    assert result[1]["claim_text"] == "Workers lost jobs in 1837."


def test_propagate_named_events_leaves_claims_without_a_year_alone():
    claims = [
        {"claim_text": "Businesses closed nationwide."},
        {"claim_text": "This crisis became known as the Panic of 1837."},
    ]

    result = propagate_named_events(claims)

    assert result[0]["claim_text"] == "Businesses closed nationwide."


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


async def test_run_extraction_first_chunk_has_no_preceding_context(kb_db, monkeypatch):
    """The first chunk of an artifact has nothing before it, so it must not
    get a preceding-context block; the second chunk must get the first
    chunk's text as context -- this is the fix for claims like "...it was
    them who pushed for the deal" losing the specific deal named a chunk
    earlier, since chunks are non-overlapping and extraction previously saw
    only one chunk at a time with no way to resolve such a reference."""
    artifact, chunks = await _make_artifact_with_chunks(kb_db, ["chunk zero text", "chunk one text"])
    seen = []

    async def fake_chat(self, messages):
        seen.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)
    config = load_config()
    await run_extraction(kb_db, config, artifact["id"])

    assert "Context immediately before this chunk" not in seen[0]
    assert "Context immediately before this chunk" in seen[1]
    assert "chunk zero text" in seen[1]


async def test_run_extraction_uses_true_preceding_chunk_for_scoped_run(kb_db, monkeypatch):
    """A chunk_ids-scoped run (verify_claim's web-fallback) only sends the
    requested chunk(s) to the LLM for claim extraction, but the preceding-
    context block must still come from the real previous chunk in the full
    artifact, not just whatever happens to precede it within the requested
    subset -- otherwise a scoped run on a later chunk would silently lose
    reference-resolution context a full-artifact run would have had."""
    artifact, chunks = await _make_artifact_with_chunks(
        kb_db, ["chunk zero text", "chunk one text", "chunk two text"],
    )
    seen = []

    async def fake_chat(self, messages):
        seen.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("deep_research.llm.LLMClient.chat", fake_chat)
    config = load_config()
    await run_extraction(kb_db, config, artifact["id"], chunk_ids=[chunks[2]["id"]])

    assert len(seen) == 1
    assert "chunk one text" in seen[0]
    assert "chunk two text" in seen[0]


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
