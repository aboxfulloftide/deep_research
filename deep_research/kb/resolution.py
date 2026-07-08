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
import json
from dataclasses import dataclass
from difflib import SequenceMatcher

from deep_research.config import Config
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import cosine, embed_texts

MIN_FUZZY_ENTITY_NAME_LENGTH = 4
FUZZY_ENTITY_TRIGRAM_THRESHOLD = 0.82


@dataclass
class PromotionResult:
    promoted_count: int = 0
    new_claim_count: int = 0
    new_entity_count: int = 0
    entity_candidate_count: int = 0
    claim_candidate_count: int = 0


def _entity_similarity(norm_a: str, norm_b: str) -> tuple[float, str] | None:
    """Guarded fuzzy match for entities: a minimum name length gate before any
    fuzzy method runs at all. The spike found unguarded substring matching
    unusably noisy (e.g. "ai" flagged as a duplicate of "britain" because
    "britain" contains the substring "ai") — this length gate is the fix."""
    if norm_a == norm_b:
        return None  # exact match; handled by get_or_create_entity, not here
    if len(norm_a) < MIN_FUZZY_ENTITY_NAME_LENGTH or len(norm_b) < MIN_FUZZY_ENTITY_NAME_LENGTH:
        return None
    if norm_a in norm_b or norm_b in norm_a:
        return 0.85, "substring"
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= FUZZY_ENTITY_TRIGRAM_THRESHOLD:
        return ratio, "trigram"
    return None


async def _generate_entity_candidates(kb_db: KBDatabase, entity_row: dict) -> int:
    others = await kb_db.list_entities(entity_type=entity_row["entity_type"], limit=2000)
    count = 0
    for other in others:
        if other["id"] == entity_row["id"]:
            continue
        sim = _entity_similarity(entity_row["normalized_name"], other["normalized_name"])
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


async def generate_claim_resolution_candidates(
    kb_db: KBDatabase, config: Config, new_claim_ids: list[str],
) -> int:
    """Embedding-similarity candidate generation for claims — the tier decision
    25 requires because lexical matching measured zero true positives in the
    spike. Compares each newly-created claim against all existing claims;
    matches above config.kb.claim_duplicate_threshold become resolution_candidates
    (never auto-merged — the spike measured only 50% precision even at 0.85)."""
    if not new_claim_ids:
        return 0

    new_claims = [await kb_db.get_claim(cid) for cid in new_claim_ids]
    all_claims = await kb_db.list_claims(limit=5000)
    if len(all_claims) < 2:
        return 0

    base_url = config.kb.embedding_base_url
    model = config.kb.embedding_model
    threshold = config.kb.claim_duplicate_threshold

    new_vectors = await embed_texts([c["canonical_text"] for c in new_claims], base_url, model)
    all_vectors = await embed_texts([c["canonical_text"] for c in all_claims], base_url, model)

    candidate_count = 0
    for i, new_claim in enumerate(new_claims):
        for j, other_claim in enumerate(all_claims):
            if other_claim["id"] == new_claim["id"]:
                continue
            score = cosine(new_vectors[i], all_vectors[j])
            if score >= threshold:
                _, created = await kb_db.add_claim_resolution_candidate(
                    new_claim["id"], other_claim["id"], score, "embedding_cosine",
                    reason=f"cosine={score:.3f}",
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
        payload = json.loads(obs["raw_payload"])

        chunk = await kb_db.get_artifact_chunk(obs["artifact_chunk_id"])
        artifact = await kb_db.get_artifact(chunk["artifact_id"])
        version = await kb_db.get_source_version(artifact["source_version_id"])
        source = await kb_db.get_source(version["source_id"])

        event_id = None
        event_payload = payload.get("event")
        if isinstance(event_payload, dict) and event_payload.get("title"):
            event_row, _ = await kb_db.get_or_create_event(
                title=event_payload["title"], start_at=event_payload.get("date"),
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
    result.claim_candidate_count = await generate_claim_resolution_candidates(kb_db, config, new_claim_ids)
    return result
