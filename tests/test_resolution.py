from deep_research.config import Config
from deep_research.kb import resolution as r


# -- _entity_similarity: pure function, no I/O -------------------------------

def test_exact_match_returns_none_handled_elsewhere():
    # Exact matches auto-merge via get_or_create_entity's UNIQUE constraint,
    # not here -- this function must never flag them as a "candidate".
    assert r._entity_similarity("openai", "openai", "organization") is None


def test_short_names_are_never_fuzzy_matched():
    # The spike's "ai" vs "britain" false-positive (unguarded substring match)
    # is exactly what the minimum-length gate exists to prevent.
    assert r._entity_similarity("ai", "britain", "concept") is None
    assert r._entity_similarity("ai", "air", "concept") is None


def test_genuine_substring_match_above_length_gate():
    score, method = r._entity_similarity("data center", "data centers", "product")
    assert method == "substring"
    assert score == 0.85


def test_similar_names_above_trigram_threshold_match():
    # A typo, not a substring relationship -- must go through the trigram
    # branch, not the (checked-first) substring branch.
    result = r._entity_similarity("silicon valley bank", "silicon valey bank", "organization")
    assert result is not None
    score, method = result
    assert method == "trigram"
    assert score >= r.FUZZY_ENTITY_TRIGRAM_THRESHOLD


def test_dissimilar_names_below_threshold_do_not_match():
    assert r._entity_similarity("federal reserve", "silicon valley bank", "organization") is None


def test_dates_never_fuzzy_match_even_when_textually_close():
    # Real false positive: "June 22, 2026" vs "June 29, 2026" scored 0.917 by
    # trigram (same month/format, only the day digits differ) despite being
    # two genuinely different points in time -- dates should only ever match
    # exactly (which auto-merges before this function is even reached).
    assert r._entity_similarity("june 22 2026", "june 29 2026", "date") is None
    assert r._entity_similarity("june 15 2026", "june 22 2026", "date") is None


def test_generic_word_buried_in_specific_name_suppressed_for_non_person():
    # Real false positives: "bank" substring-matches nearly every organization
    # whose name happens to contain the word "bank".
    assert r._entity_similarity("bank", "silicon valley bank", "organization") is None
    assert r._entity_similarity("bank", "central banks", "organization") is None
    assert r._entity_similarity("economy", "british industrial economy", "concept") is None


def test_generic_word_ratio_filter_does_not_apply_to_person():
    # Real, valuable pattern: a bare surname substring-matching someone's full
    # name has a low length ratio too, but this is almost always the same
    # person, not noise -- must not be suppressed like the organization case.
    result = r._entity_similarity("clayton", "christopher clayton", "person")
    assert result == (0.85, "substring")
    result = r._entity_similarity("coppola", "antonio coppola", "person")
    assert result == (0.85, "substring")


def test_generic_word_ratio_filter_allows_close_length_non_person_matches():
    # Singular/plural and short/long forms close in length should still match
    # for non-person types -- only a *large* length gap is suppressed.
    score, method = r._entity_similarity("data center", "data centers", "product")
    assert (score, method) == (0.85, "substring")


def test_trivial_plural_of_bare_word_suppressed_for_non_person():
    # Real false positive: "bank" / "banks" is just grammatical number on a
    # generic noun, not a meaningful entity distinction either way.
    assert r._entity_similarity("bank", "banks", "organization") is None
    assert r._entity_similarity("regulator", "regulators", "organization") is None


def test_trivial_plural_does_not_apply_to_multiword_phrases():
    # "data center" -> "data centers" pluralizes a real, specific multi-word
    # concept, not a bare generic noun -- must still match.
    score, method = r._entity_similarity("data center", "data centers", "product")
    assert (score, method) == (0.85, "substring")


def test_trivial_qualifier_suppressed_regardless_of_which_word_it_is():
    # Real false positives: exactly one extra qualifying word, whichever word
    # it happens to be, almost always makes it a distinctly more specific
    # concept -- not limited to a fixed list of "metric" words.
    assert r._entity_similarity("unemployment", "unemployment rate", "concept") is None
    assert r._entity_similarity("us equity market", "us equity market capitalization", "concept") is None
    assert r._entity_similarity("capital cycle", "capital cycle theory", "concept") is None
    assert r._entity_similarity("regulators", "financial regulators", "organization") is None


def test_trivial_qualifier_requires_exactly_one_extra_word():
    # Two or more extra words is a bigger, less certain gap -- falls through
    # to the ordinary length-ratio handling instead of this stricter check.
    assert r._entity_similarity("bank", "silicon valley bank", "organization") is None  # still suppressed, via the ratio filter
    result = r._entity_similarity("federal reserve bank", "federal reserve bank of new york", "organization")
    assert result is not None  # 3 extra words, ratio is high enough to still be a plausible match


# -- numeric mismatch: disqualifying for entity names, unlike claim text ----
# A name containing a number is essentially that number -- pure trigram
# similarity can't tell "late 1990s"/"late 1920s" apart from a genuine
# spelling variant, so any numeric mismatch suppresses the candidate outright.

def test_numeric_mismatch_suppresses_high_trigram_score():
    # Same score class as a genuine typo (only two digits differ), but a
    # decade apart -- must not be flagged as a candidate at all.
    assert r._entity_similarity("late 1990s", "late 1920s", "event") is None
    assert r._entity_similarity("section 1 6a 19", "section 1 6a 18", "concept") is None
    assert r._entity_similarity("colossus 1", "colossus 2", "product") is None


def test_matching_numbers_still_proceed_to_normal_fuzzy_evaluation():
    # Both sides cite the same number -- the numeric guard must not itself
    # block a genuine match; ordinary trigram/substring rules still apply.
    result = r._entity_similarity("federal reserve bank 12", "federal reserve bank of 12", "organization")
    assert result is not None


def test_number_on_only_one_side_still_disqualifies():
    # A number appearing on just one side is still a mismatch (one names a
    # specific instance, the other doesn't) -- "colossus" vs "colossus 2" are
    # not confidently the same thing either.
    assert r._entity_similarity("colossus", "colossus 2", "product") is None


def test_no_numbers_on_either_side_does_not_engage_the_guard():
    # Neither name has a number at all -- confirms the guard only engages
    # when there's an actual numeric mismatch to disqualify on, not every
    # ordinary fuzzy match.
    result = r._entity_similarity("silicon valey bank", "silicon valley bank", "organization")
    assert result is not None


# -- spacing/hyphen variants: confident enough to auto-merge, not review ----
# Found as the single biggest source of resolution-queue noise in practice --
# "jp morgan"/"jpmorgan", "data centers"/"datacenters" are certainly the same
# real-world thing, not a coincidental fuzzy near-miss, so these skip the
# review queue entirely (see _generate_entity_candidates).

def test_is_spacing_variant_true_for_identical_modulo_whitespace():
    assert r._is_spacing_variant("jp morgan", "jpmorgan") is True
    assert r._is_spacing_variant("data centers", "datacenters") is True
    assert r._is_spacing_variant("saline datacenter", "saline data center") is True


def test_is_spacing_variant_false_for_genuinely_different_names():
    assert r._is_spacing_variant("jp morgan", "goldman sachs") is False
    assert r._is_spacing_variant("data center", "data centers") is False  # a real plural, not a spacing artifact


# -- _generate_entity_candidates: needs a real DB ----------------------------

async def test_generate_entity_candidates_creates_row_for_fuzzy_match(kb_db):
    e1, _ = await kb_db.get_or_create_entity("product", "data center")
    e2, _ = await kb_db.get_or_create_entity("product", "data centers")

    count = await r._generate_entity_candidates(kb_db, e1)

    assert count == 1
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert len(candidates) == 1
    ids = {candidates[0]["left_entity_id"], candidates[0]["right_entity_id"]}
    assert ids == {e1["id"], e2["id"]}


async def test_generate_entity_candidates_skips_dissimilar_entities(kb_db):
    e1, _ = await kb_db.get_or_create_entity("organization", "Federal Reserve")
    await kb_db.get_or_create_entity("organization", "Silicon Valley Bank")

    count = await r._generate_entity_candidates(kb_db, e1)

    assert count == 0


async def test_generate_entity_candidates_auto_merges_spacing_variants(kb_db):
    """The exact "Saline Datacenter" / "Saline data center" pattern this was
    written for -- confident enough to merge immediately rather than sit in
    the review queue as one more near-duplicate to click through."""
    e1, _ = await kb_db.get_or_create_entity("product", "JPMorgan")
    e2, _ = await kb_db.get_or_create_entity("product", "JP Morgan")

    count = await r._generate_entity_candidates(kb_db, e1)

    assert count == 0  # not queued as a candidate at all
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert candidates == []

    winner = await kb_db.get_entity(e1["id"])
    loser = await kb_db.get_entity(e2["id"])
    # one of them absorbed the other -- exactly one still points nowhere,
    # the other points at it.
    assert (winner["merged_into_entity_id"] is None) != (loser["merged_into_entity_id"] is None)
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert candidates == []


# -- LLM vetting: same/different verdicts skip the review queue -------------

def test_parse_entity_classification_handles_plain_json():
    result = r._parse_entity_classification('{"relationship": "same", "confidence": 0.9, "reasoning": "ok"}')
    assert result == {"relationship": "same", "confidence": 0.9, "reasoning": "ok"}


def test_parse_entity_classification_extracts_json_from_surrounding_text():
    content = 'Sure, here you go:\n{"relationship": "different", "confidence": 0.8, "reasoning": "ok"}\nDone.'
    result = r._parse_entity_classification(content)
    assert result["relationship"] == "different"


def test_parse_entity_classification_falls_back_safely_on_garbage():
    result = r._parse_entity_classification("not json at all")
    assert result == {"relationship": "different", "confidence": 0.0, "reasoning": "could not parse model output"}


async def test_generate_entity_candidates_auto_merges_on_confident_llm_same_verdict(kb_db, monkeypatch):
    e1, _ = await kb_db.get_or_create_entity("person", "Allan Greenspan")
    e2, _ = await kb_db.get_or_create_entity("person", "Alan Grenspan")

    async def fake_classify(llm, name_a, name_b, entity_type):
        return {"relationship": "same", "confidence": 0.95, "reasoning": "test"}

    monkeypatch.setattr(r, "_classify_entity_duplicate", fake_classify)

    count = await r._generate_entity_candidates(kb_db, e1, llm=object())

    assert count == 0  # not queued -- merged directly
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert candidates == []
    winner = await kb_db.get_entity(e1["id"])
    loser = await kb_db.get_entity(e2["id"])
    assert (winner["merged_into_entity_id"] is None) != (loser["merged_into_entity_id"] is None)


async def test_generate_entity_candidates_drops_on_confident_llm_different_verdict(kb_db, monkeypatch):
    e1, _ = await kb_db.get_or_create_entity("concept", "market correlation")
    await kb_db.get_or_create_entity("concept", "market correction")

    async def fake_classify(llm, name_a, name_b, entity_type):
        return {"relationship": "different", "confidence": 0.9, "reasoning": "test"}

    monkeypatch.setattr(r, "_classify_entity_duplicate", fake_classify)

    count = await r._generate_entity_candidates(kb_db, e1, llm=object())

    assert count == 0  # dropped silently, not merged and not queued
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert candidates == []


async def test_generate_entity_candidates_queues_for_review_on_low_confidence_llm_verdict(kb_db, monkeypatch):
    e1, _ = await kb_db.get_or_create_entity("organization", "central banks")
    await kb_db.get_or_create_entity("organization", "central bankers")

    async def fake_classify(llm, name_a, name_b, entity_type):
        return {"relationship": "same", "confidence": 0.4, "reasoning": "not sure"}

    monkeypatch.setattr(r, "_classify_entity_duplicate", fake_classify)

    count = await r._generate_entity_candidates(kb_db, e1, llm=object())

    assert count == 1  # falls through to the ordinary review queue
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert len(candidates) == 1


async def test_generate_entity_candidates_queues_for_review_when_llm_call_raises(kb_db, monkeypatch):
    e1, _ = await kb_db.get_or_create_entity("organization", "central banks")
    await kb_db.get_or_create_entity("organization", "central bankers")

    async def raising_classify(llm, name_a, name_b, entity_type):
        raise ConnectionError("simulated transient LLM failure")

    monkeypatch.setattr(r, "_classify_entity_duplicate", raising_classify)

    count = await r._generate_entity_candidates(kb_db, e1, llm=object())

    assert count == 1  # a broken LLM call must never block the safe fallback
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert len(candidates) == 1


async def test_generate_entity_candidates_without_llm_preserves_old_behavior(kb_db):
    # llm=None (the default) must behave exactly like before this feature --
    # existing callers/tests that omit it are unaffected.
    e1, _ = await kb_db.get_or_create_entity("product", "data center")
    await kb_db.get_or_create_entity("product", "data centers")

    count = await r._generate_entity_candidates(kb_db, e1)

    assert count == 1


# -- claim embedding + candidate generation (embed_texts mocked out) ---------
# Real cosine similarity is computed by Postgres/pgvector over whatever
# vectors we hand it -- mocking embed_texts (the Ollama call) still exercises
# the real SQL/threshold logic, just with deterministic fake embeddings
# instead of depending on a live embedding server.

def _fake_config() -> Config:
    config = Config()
    config.kb.claim_duplicate_threshold = 0.85
    return config


async def test_embed_new_claims_and_generate_candidates_for_near_duplicate(kb_db, monkeypatch):
    claim_a, _ = await kb_db.get_or_create_claim("fact", "Unemployment fell to 3.9% in Q2 2025.")
    claim_b, _ = await kb_db.get_or_create_claim("fact", "Unemployment dropped to 3.9 percent in Q2 2025.")

    # embed_new_claims embeds in the same order as the claim_ids list passed to
    # it, so returning fixed vectors by position (not by inspecting the text)
    # matches the call in test order below: [claim_a, claim_b].
    fake_vectors_in_order = [
        [1.0, 0.0, 0.0] + [0.0] * 765,   # claim_a
        [0.99, 0.01, 0.0] + [0.0] * 765,  # claim_b: near-identical -> cosine ~1.0
    ]

    async def fake_embed_texts(texts, base_url, model, instruction_prefix="clustering: "):
        assert len(texts) == len(fake_vectors_in_order)
        return fake_vectors_in_order

    monkeypatch.setattr(r, "embed_texts", fake_embed_texts)

    config = _fake_config()
    await r.embed_new_claims(kb_db, config, [claim_a["id"], claim_b["id"]])

    refreshed_a = await kb_db.get_claim(claim_a["id"])
    assert refreshed_a["embedding"] is not None

    count = await r.generate_claim_resolution_candidates(kb_db, config, [claim_a["id"], claim_b["id"]])
    assert count >= 1
    candidates = await kb_db.list_resolution_candidates(candidate_type="claim_duplicate", status="open")
    ids = {candidates[0]["left_claim_id"], candidates[0]["right_claim_id"]}
    assert ids == {claim_a["id"], claim_b["id"]}


async def test_generate_candidates_skips_claims_with_no_embedding(kb_db, monkeypatch):
    claim_a, _ = await kb_db.get_or_create_claim("fact", "A claim whose embedding call will fail.")

    async def failing_embed_texts(*args, **kwargs):
        raise ConnectionError("simulated Ollama outage")

    monkeypatch.setattr(r, "embed_texts", failing_embed_texts)

    config = _fake_config()
    # embed_new_claims is best-effort -- must not raise even though embedding fails.
    await r.embed_new_claims(kb_db, config, [claim_a["id"]])

    refreshed = await kb_db.get_claim(claim_a["id"])
    assert refreshed["embedding"] is None

    # generate_claim_resolution_candidates must skip (not crash on) a claim with no embedding.
    count = await r.generate_claim_resolution_candidates(kb_db, config, [claim_a["id"]])
    assert count == 0


# -- _extract_numbers: pure function, no I/O ---------------------------------

def test_extract_numbers_basic_integers():
    assert r._extract_numbers("There were 140 bank failures in 2009.") == {"140", "2009"}


def test_extract_numbers_strips_comma_thousands_separators():
    assert r._extract_numbers("Revenue was $1,200 million.") == {"1200"}


def test_extract_numbers_keeps_decimals():
    assert r._extract_numbers("Unemployment fell to 3.9%.") == {"3.9"}


def test_extract_numbers_returns_empty_set_when_no_numbers():
    assert r._extract_numbers("The market showed extreme enthusiasm.") == set()


def test_extract_numbers_disjoint_sets_have_no_overlap():
    a = r._extract_numbers("There were 140 bank failures in 2009.")
    b = r._extract_numbers("There were 157 bank failures in 2010.")
    assert a.isdisjoint(b)


def test_extract_numbers_overlapping_sets_share_a_value():
    a = r._extract_numbers("The average PE ratio historically has been 16.")
    b = r._extract_numbers("The average PE ratio is 16.")
    assert not a.isdisjoint(b)


# -- numeric-mismatch suppression in candidate generation --------------------

async def test_candidates_suppressed_when_claims_cite_different_numbers(kb_db, monkeypatch):
    """The exact real-world false positive this filter was added for: two
    structurally similar, high-cosine-similarity claims that are both true and
    not duplicates because they're about different years/counts."""
    claim_a, _ = await kb_db.get_or_create_claim("fact", "There were 140 bank failures in 2009.")
    claim_b, _ = await kb_db.get_or_create_claim("fact", "There were 157 bank failures in 2010.")

    # Near-identical fake vectors -- would clear the similarity threshold easily.
    fake_vectors_in_order = [
        [1.0, 0.0, 0.0] + [0.0] * 765,
        [0.99, 0.01, 0.0] + [0.0] * 765,
    ]

    async def fake_embed_texts(texts, base_url, model, instruction_prefix="clustering: "):
        return fake_vectors_in_order

    monkeypatch.setattr(r, "embed_texts", fake_embed_texts)

    config = _fake_config()
    await r.embed_new_claims(kb_db, config, [claim_a["id"], claim_b["id"]])
    count = await r.generate_claim_resolution_candidates(kb_db, config, [claim_a["id"], claim_b["id"]])

    assert count == 0
    candidates = await kb_db.list_resolution_candidates(candidate_type="claim_duplicate", status="open")
    ids_involved = {claim_a["id"], claim_b["id"]}
    matching = [c for c in candidates if {c["left_claim_id"], c["right_claim_id"]} == ids_involved]
    assert matching == []


async def test_candidates_not_suppressed_when_only_one_side_has_numbers(kb_db, monkeypatch):
    """A number appearing on only one side isn't a disagreement -- there's
    nothing on the other side to conflict with, so the filter must not
    suppress this pair."""
    claim_a, _ = await kb_db.get_or_create_claim("fact", "Silicon Valley Bank collapsed suddenly.")
    claim_b, _ = await kb_db.get_or_create_claim("fact", "Silicon Valley Bank had $175 billion in deposits.")

    fake_vectors_in_order = [
        [1.0, 0.0, 0.0] + [0.0] * 765,
        [0.99, 0.01, 0.0] + [0.0] * 765,
    ]

    async def fake_embed_texts(texts, base_url, model, instruction_prefix="clustering: "):
        return fake_vectors_in_order

    monkeypatch.setattr(r, "embed_texts", fake_embed_texts)

    config = _fake_config()
    await r.embed_new_claims(kb_db, config, [claim_a["id"], claim_b["id"]])
    count = await r.generate_claim_resolution_candidates(kb_db, config, [claim_a["id"], claim_b["id"]])

    assert count >= 1
