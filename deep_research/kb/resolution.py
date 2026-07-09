"""Resolution + promotion: turns raw extracted_observations into canonical
entities/events/claims/metrics, per the resolution strategy locked in decision
25 of PLAN_KB_ARCHITECTURE.md.

Two separate concerns, deliberately kept apart:
- promotion (exact-match only): entities/events/claims auto-merge exclusively
  on exact normalized-name/text match (enforced by DB UNIQUE constraints in
  get_or_create_entity/get_or_create_event/get_or_create_claim). This always
  happens, synchronously, as part of promoting an observation.
- candidate generation (fuzzy/embedding, never auto-merge): guarded fuzzy
  matching for entities, embedding cosine similarity for claims. Both only
  ever write to resolution_candidates for human review.
"""

import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from deep_research.config import Config
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import embed_texts

MIN_FUZZY_ENTITY_NAME_LENGTH = 4
FUZZY_ENTITY_TRIGRAM_THRESHOLD = 0.82
# Below this shorter/longer normalized-name length ratio, a substring match is
# usually a generic word buried in a much more specific name ("bank" inside
# "Silicon Valley Bank", "economy" inside "British industrial economy") rather
# than the same real-world thing referred to two ways -- not applied to
# `person`, where a short surname matching a full name (`Clayton` inside
# `Christopher Clayton`) is a genuinely common and valuable pattern that this
# ratio would otherwise wrongly suppress.
FUZZY_SUBSTRING_MIN_LENGTH_RATIO = 0.5
# Two different dates are never "the same" fuzzy-adjacent entity the way a
# nickname or acronym might be -- an identical date already auto-merges via
# the exact-match tier, so fuzzy matching on dates only ever produces noise
# ("June 22, 2026" scoring 0.917 against "June 29, 2026" by trigram overlap).
NO_FUZZY_MATCH_ENTITY_TYPES = {"date"}

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")  # decimal point only consumed if followed by a digit


def _extract_numbers(text: str) -> set[str]:
    """Numbers as they literally appear in claim text (years, counts, dollar
    amounts, percentages) — comma thousands separators stripped so '1,200'
    and '1200' compare equal. Used as a cheap sanity check before treating two
    structurally-similar claims as duplicates: 'X bank failures in 2009' and
    'Y bank failures in 2010' score high on embedding similarity despite being
    two distinct (and both true) facts — if two claims cite numbers at all,
    and none of those numbers match, they're essentially never the same
    fact."""
    return {m.replace(",", "") for m in _NUMBER_RE.findall(text)}


@dataclass
class PromotionResult:
    promoted_count: int = 0
    new_claim_count: int = 0
    new_entity_count: int = 0
    entity_candidate_count: int = 0
    claim_candidate_count: int = 0
    new_claim_ids: list[str] = field(default_factory=list)


def _is_trivial_plural(shorter: str, longer: str) -> bool:
    """True if `longer` is just the plural of `shorter`, and `shorter` is a
    single bare word, not a multi-word phrase ("data center" -> "data
    centers" is a real, specific concept pluralizing; "bank" -> "banks" is
    just grammatical number on a generic noun). A single generic word
    pluralizing is not a meaningful entity distinction worth a reviewer's
    time either way."""
    if " " in shorter:
        return False
    return longer in (shorter + "s", shorter + "es")


def _is_trivial_qualifier(shorter: str, longer: str) -> bool:
    """True if `longer`'s words are exactly `shorter`'s words plus one extra
    qualifying word at the start or end ("capital cycle" + "theory",
    "financial" + "regulators", "unemployment" + "rate", "US equity market" +
    "capitalization"). A single qualifying word — whichever word it happens
    to be — almost always makes it a distinctly different, more specific
    concept, not the same entity spelled two ways."""
    shorter_words = shorter.split()
    longer_words = longer.split()
    if len(longer_words) != len(shorter_words) + 1:
        return False
    return longer_words[1:] == shorter_words or longer_words[:-1] == shorter_words


def _entity_similarity(norm_a: str, norm_b: str, entity_type: str) -> tuple[float, str] | None:
    """Guarded fuzzy match for entities: a minimum name length gate before any
    fuzzy method runs at all. The spike found unguarded substring matching
    unusably noisy (e.g. "ai" flagged as a duplicate of "britain" because
    "britain" contains the substring "ai") — this length gate is the fix.
    entity_type gates several further, type-specific sources of noise: see
    NO_FUZZY_MATCH_ENTITY_TYPES, _is_trivial_plural, _is_trivial_qualifier, and
    FUZZY_SUBSTRING_MIN_LENGTH_RATIO above."""
    if norm_a == norm_b:
        return None  # exact match; handled by get_or_create_entity, not here
    if entity_type in NO_FUZZY_MATCH_ENTITY_TYPES:
        return None
    if len(norm_a) < MIN_FUZZY_ENTITY_NAME_LENGTH or len(norm_b) < MIN_FUZZY_ENTITY_NAME_LENGTH:
        return None
    if norm_a in norm_b or norm_b in norm_a:
        shorter, longer = sorted([norm_a, norm_b], key=len)
        if entity_type != "person":
            if _is_trivial_plural(shorter, longer) or _is_trivial_qualifier(shorter, longer):
                return None
            if len(shorter) / len(longer) < FUZZY_SUBSTRING_MIN_LENGTH_RATIO:
                return None
        return 0.85, "substring"
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= FUZZY_ENTITY_TRIGRAM_THRESHOLD:
        return ratio, "trigram"
    return None


async def _generate_entity_candidates(kb_db: KBDatabase, entity_row: dict) -> int:
    if entity_row["entity_type"] in NO_FUZZY_MATCH_ENTITY_TYPES:
        return 0
    others = await kb_db.list_entities(entity_type=entity_row["entity_type"], limit=2000)
    count = 0
    for other in others:
        if other["id"] == entity_row["id"]:
            continue
        sim = _entity_similarity(entity_row["normalized_name"], other["normalized_name"], entity_row["entity_type"])
        if sim is None:
            continue
        score, method = sim
        _, created = await kb_db.add_entity_resolution_candidate(
            entity_row["id"], other["id"], score, method,
            reason=f"fuzzy match: {entity_row['name']!r} / {other['name']!r}",
        )
        if created:
            count += 1
    return count


async def embed_new_claims(kb_db: KBDatabase, config: Config, new_claim_ids: list[str]) -> None:
    """Step 8: embeds and persists each newly-promoted claim once, at creation
    time, so candidate generation (below) and future resolution runs never
    need to re-embed it again. Best-effort — if the embedding backend (Ollama)
    is unreachable, claims are simply left with embedding=NULL and picked up
    later by the `backfill-embeddings` CLI command; embeddings are a retrieval/
    resolution enhancement, not something that should block promotion."""
    if not new_claim_ids:
        return
    try:
        new_claims = [await kb_db.get_claim(cid) for cid in new_claim_ids]
        vectors = await embed_texts(
            [c["canonical_text"] for c in new_claims],
            config.kb.embedding_base_url, config.kb.embedding_model,
        )
        for claim, vector in zip(new_claims, vectors):
            await kb_db.set_claim_embedding(claim["id"], vector)
    except Exception:
        pass


async def generate_claim_resolution_candidates(
    kb_db: KBDatabase, config: Config, new_claim_ids: list[str],
) -> int:
    """Embedding-similarity candidate generation for claims — the tier decision
    25 requires because lexical matching measured zero true positives in the
    spike. Compares each newly-created claim against every other claim via the
    HNSW nearest-neighbor index over persisted embeddings (embed_new_claims
    above), instead of re-embedding the whole KB on every single resolution
    run — the previous approach that became the actual bottleneck as the KB
    grew. Matches above config.kb.claim_duplicate_threshold become
    resolution_candidates (never auto-merged — the spike measured only 50%
    precision even at 0.85). Matches whose numbers flatly disagree ("140 bank
    failures in 2009" vs. "157 bank failures in 2010" — both true, not
    duplicates) are suppressed entirely: structurally similar sentences score
    high on embedding similarity regardless of the actual numbers inside them,
    and this was measured as a real, recurring source of review-queue noise."""
    if not new_claim_ids:
        return 0

    threshold = config.kb.claim_duplicate_threshold
    candidate_count = 0
    for claim_id in new_claim_ids:
        claim = await kb_db.get_claim(claim_id)
        if claim is None or claim.get("embedding") is None:
            continue  # embedding failed best-effort at creation time; backfill will cover it
        embedding = claim["embedding"].to_list()
        claim_numbers = _extract_numbers(claim["canonical_text"])
        neighbors = await kb_db.find_similar_claims(claim_id, embedding, limit=20)
        for other in neighbors:
            if other["similarity"] < threshold:
                break  # find_similar_claims orders nearest-first (similarity descending)
            other_numbers = _extract_numbers(other["canonical_text"])
            if claim_numbers and other_numbers and claim_numbers.isdisjoint(other_numbers):
                continue  # both claims cite numbers, but share none -- not the same fact
            _, created = await kb_db.add_claim_resolution_candidate(
                claim_id, other["id"], other["similarity"], "embedding_cosine",
                reason=f"cosine={other['similarity']:.3f}",
            )
            if created:
                candidate_count += 1
    return candidate_count


def _parse_metric_value(value) -> tuple[float | None, str | None]:
    if isinstance(value, (int, float)):
        return float(value), None
    if value is None:
        return None, None
    try:
        return float(str(value).replace(",", "")), None
    except ValueError:
        return None, str(value)


async def resolve_and_promote(
    kb_db: KBDatabase, config: Config, extraction_run_id: str,
) -> PromotionResult:
    observations = await kb_db.list_observations(extraction_run_id, status="new")
    result = PromotionResult()
    new_claim_ids: list[str] = []

    for obs in observations:
        # raw_payload is jsonb — asyncpg's registered codec already decodes it
        # to a dict, no json.loads needed (unlike the SQLite TEXT column this
        # replaced).
        payload = obs["raw_payload"]

        chunk = await kb_db.get_artifact_chunk(obs["artifact_chunk_id"])
        artifact = await kb_db.get_artifact(chunk["artifact_id"])
        version = await kb_db.get_source_version(artifact["source_version_id"])
        source = await kb_db.get_source(version["source_id"])

        event_id = None
        event_payload = payload.get("event")
        if isinstance(event_payload, dict) and event_payload.get("title"):
            event_row, _ = await kb_db.get_or_create_event(
                title=event_payload["title"], start_at=event_payload.get("date"),
                date_precision=event_payload.get("date_precision"),
            )
            event_id = event_row["id"]

        for ent in payload.get("entities") or []:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            entity_row, created = await kb_db.get_or_create_entity(
                ent.get("type") or "concept", ent["name"],
            )
            if created:
                result.new_entity_count += 1
                result.entity_candidate_count += await _generate_entity_candidates(kb_db, entity_row)

        claim_row, claim_created = await kb_db.get_or_create_claim(
            claim_type=payload.get("claim_type") or "fact",
            canonical_text=obs["raw_text"],
            event_id=event_id,
            confidence=obs["confidence"],
            importance_score=obs["importance_score"],
        )
        if claim_created:
            new_claim_ids.append(claim_row["id"])

        quote = payload.get("supporting_quote") or ""
        await kb_db.add_claim_evidence(
            claim_id=claim_row["id"], artifact_chunk_id=chunk["id"],
            source_id=source["id"], source_version_id=version["id"],
            evidence_type="support", excerpt_text=quote,
            excerpt_hash=hashlib.sha256(quote.encode()).hexdigest() if quote else None,
            extraction_run_id=extraction_run_id, extracted_observation_id=obs["id"],
            char_start=obs["char_start"], char_end=obs["char_end"],
            confidence=obs["confidence"],
        )

        for m in payload.get("metrics") or []:
            if not isinstance(m, dict) or not m.get("name") or m.get("value") is None:
                continue
            value_numeric, value_text = _parse_metric_value(m.get("value"))
            await kb_db.add_metric(
                metric_name=m["name"], claim_id=claim_row["id"],
                value_numeric=value_numeric, value_text=value_text,
                unit=m.get("unit"), currency_code=m.get("currency"),
            )

        await kb_db.mark_observation_promoted(obs["id"], claim_row["id"])
        result.promoted_count += 1

    result.new_claim_count = len(new_claim_ids)
    result.new_claim_ids = new_claim_ids
    await embed_new_claims(kb_db, config, new_claim_ids)
    result.claim_candidate_count = await generate_claim_resolution_candidates(kb_db, config, new_claim_ids)
    return result
