from deep_research.config import Config
from deep_research.kb import resolution as r


# -- _entity_similarity: pure function, no I/O -------------------------------

def test_exact_match_returns_none_handled_elsewhere():
    # Exact matches auto-merge via get_or_create_entity's UNIQUE constraint,
    # not here -- this function must never flag them as a "candidate".
    assert r._entity_similarity("openai", "openai") is None


def test_short_names_are_never_fuzzy_matched():
    # The spike's "ai" vs "britain" false-positive (unguarded substring match)
    # is exactly what the minimum-length gate exists to prevent.
    assert r._entity_similarity("ai", "britain") is None
    assert r._entity_similarity("ai", "air") is None


def test_genuine_substring_match_above_length_gate():
    score, method = r._entity_similarity("data center", "data centers")
    assert method == "substring"
    assert score == 0.85


def test_similar_names_above_trigram_threshold_match():
    # A typo, not a substring relationship -- must go through the trigram
    # branch, not the (checked-first) substring branch.
    result = r._entity_similarity("silicon valley bank", "silicon valey bank")
    assert result is not None
    score, method = result
    assert method == "trigram"
    assert score >= r.FUZZY_ENTITY_TRIGRAM_THRESHOLD


def test_dissimilar_names_below_threshold_do_not_match():
    assert r._entity_similarity("federal reserve", "silicon valley bank") is None


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
    candidates = await kb_db.list_resolution_candidates(candidate_type="entity_duplicate", status="open")
    assert candidates == []


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
