"""Topic suggestion generation (build order step 7, decision 27).

Signal is entity overlap: a topic's "entity set" is derived from its
currently-attached claims, and any other claim mentioning one of those
entities becomes a suggestion (never an auto-attach — decision 27 requires
human review, same as resolution_candidates). Embedding-based topic
suggestion was considered but cut for v1 scope; entity overlap is simpler,
cheaper, and more interpretable ("this claim mentions Nvidia, already central
to this topic").

Two entry points, matching decision 27's "retroactive AND forward" requirement:
- generate_topic_suggestions: full backfill, run at topic creation (or
  manually re-run later)
- check_claims_against_topics: forward check, run after each extraction's
  resolve_and_promote so new claims get checked against existing topics
  without re-scanning the whole KB
"""

from dataclasses import dataclass

from deep_research.kb.chunking import normalize_name
from deep_research.kb.db import KBDatabase

ENTITY_OVERLAP_METHOD = "entity_overlap"


@dataclass
class SuggestionResult:
    claims_suggested: int = 0
    sources_suggested: int = 0


async def get_topic_entity_names(kb_db: KBDatabase, topic_id: str) -> set[str]:
    claim_ids = list(await kb_db.get_topic_claim_ids(topic_id, link_status="attached"))
    entities_by_claim = await kb_db.get_claims_entities_bulk(claim_ids)
    names: set[str] = set()
    for entities in entities_by_claim.values():
        for entity in entities:
            names.add(normalize_name(entity["name"]))
    return names


async def _suggest_claims_and_sources(
    kb_db: KBDatabase, topic_id: str, matches: list[tuple[dict, float, list[str]]],
) -> SuggestionResult:
    result = SuggestionResult()
    suggested_source_ids: set[str] = set()
    for claim, score, matched_names in matches:
        reason = f"entity overlap: {', '.join(matched_names)}"
        _, created = await kb_db.suggest_claim_for_topic(topic_id, claim["id"], ENTITY_OVERLAP_METHOD, score)
        if created:
            result.claims_suggested += 1

        for source_id in await kb_db.get_claim_source_ids(claim["id"]):
            if source_id in suggested_source_ids:
                continue
            suggested_source_ids.add(source_id)
            _, s_created = await kb_db.suggest_source_for_topic(
                topic_id, source_id, f"{ENTITY_OVERLAP_METHOD}_via_claim", score,
            )
            if s_created:
                result.sources_suggested += 1
    return result


async def generate_topic_suggestions(kb_db: KBDatabase, topic_id: str) -> SuggestionResult:
    """Backfill: scan every claim in the KB (not just new ones) against the
    topic's current entity set. Safe to re-run — suggest_claim_for_topic/
    suggest_source_for_topic only ever create a *new* link row, never
    overwrite an existing attached/suggested/rejected one."""
    entity_names = await get_topic_entity_names(kb_db, topic_id)
    if not entity_names:
        return SuggestionResult()

    already_linked = await kb_db.get_topic_claim_ids(topic_id, link_status=None)
    matches = await kb_db.find_claims_by_entity_overlap(entity_names, exclude_claim_ids=already_linked)
    return await _suggest_claims_and_sources(kb_db, topic_id, matches)


async def check_claims_against_topics(kb_db: KBDatabase, claim_ids: list[str]) -> dict[str, SuggestionResult]:
    """Forward check: run after resolve_and_promote so newly-created claims
    get checked against every existing topic, without re-scanning the whole
    KB for topics that already have their suggestions up to date."""
    if not claim_ids:
        return {}

    entities_by_claim = await kb_db.get_claims_entities_bulk(claim_ids)
    claim_names: dict[str, set[str]] = {
        cid: {normalize_name(e["name"]) for e in entities}
        for cid, entities in entities_by_claim.items()
    }
    if not any(claim_names.values()):
        return {}

    results: dict[str, SuggestionResult] = {}
    for topic in await kb_db.list_topics(limit=1000):
        if topic["status"] != "active":
            continue
        topic_entity_names = await get_topic_entity_names(kb_db, topic["id"])
        if not topic_entity_names:
            continue
        already_linked = await kb_db.get_topic_claim_ids(topic["id"], link_status=None)

        result = SuggestionResult()
        for claim_id in claim_ids:
            if claim_id in already_linked:
                continue
            overlap = claim_names.get(claim_id, set()) & topic_entity_names
            if not overlap:
                continue
            claim = await kb_db.get_claim(claim_id)
            score = len(overlap) / max(len(claim_names[claim_id]), 1)
            sub_result = await _suggest_claims_and_sources(
                kb_db, topic["id"], [(claim, score, sorted(overlap))],
            )
            result.claims_suggested += sub_result.claims_suggested
            result.sources_suggested += sub_result.sources_suggested

        if result.claims_suggested or result.sources_suggested:
            results[topic["id"]] = result

    return results
