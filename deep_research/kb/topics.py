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

Entity overlap alone is noisy: a claim mentioning "Nvidia" only in passing
is not necessarily about the same topic as one centrally about Nvidia. When
an LLM client is given, a confident-enough "not actually relevant" verdict
skips creating the suggestion row -- unlike entity/claim-duplicate vetting,
there's no symmetric "confident yes -> skip the queue" tier here, because
decision 27 already requires every suggestion to go through human review
regardless of confidence; the LLM's only job here is cutting queue noise
before it's created, biased toward keeping a suggestion whenever unsure
(a missed suggestion is worse than one extra click)."""

import json
import re
from dataclasses import dataclass

from deep_research.config import Config, LLMConfig
from deep_research.kb.chunking import normalize_name
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import detect_model
from deep_research.llm import LLMClient

ENTITY_OVERLAP_METHOD = "entity_overlap"

TOPIC_RELEVANCE_SYSTEM_PROMPT = """/no_think
You are deciding whether a claim actually belongs in a topic, given that it shares at least one named entity with claims already in the topic.

Sharing a named entity is not enough on its own -- a claim can mention an entity only in passing while being about something else entirely.

Classify as exactly one of:
- "relevant": the claim is actually about the topic's subject matter, not just incidentally mentioning a shared entity
- "not_relevant": the claim is about something else -- the shared entity is incidental, not the actual subject of the claim

If genuinely unsure, prefer "relevant" -- a human reviewing the suggestion queue is a cheap check; silently dropping something that did belong is not recoverable the same way.

Return ONLY a JSON object: {"relationship": "relevant"|"not_relevant", "confidence": 0.0-1.0, "reasoning": "one short sentence"}
"""

# Suppression only fires above this bar -- deliberately higher than the
# entity/claim-duplicate merge/drop thresholds (0.75), since a false
# suppression here is a silent, unrecoverable loss (the suggestion is never
# created at all) rather than a queued item a human can still catch.
TOPIC_RELEVANCE_SUPPRESS_THRESHOLD = 0.85


def _parse_relevance_classification(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Safe default: never suppress a suggestion over a response that
    # couldn't be parsed -- fall through to the existing behavior instead.
    return {"relationship": "relevant", "confidence": 0.0, "reasoning": "could not parse model output"}


async def _classify_claim_topic_relevance(
    llm: LLMClient, topic_name: str, topic_description: str | None, claim_text: str, matched_names: list[str],
) -> dict:
    topic_context = topic_name if not topic_description else f"{topic_name} -- {topic_description}"
    messages = [
        {"role": "system", "content": TOPIC_RELEVANCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Topic: {topic_context}\nShared entity/entities: {', '.join(matched_names)}\n"
                f"Claim: {claim_text}\n\nClassify."
            ),
        },
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    return _parse_relevance_classification(content)


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
    kb_db: KBDatabase, topic_id: str, matches: list[tuple[dict, float, list[str]]], llm: LLMClient | None = None,
) -> SuggestionResult:
    result = SuggestionResult()
    suggested_source_ids: set[str] = set()

    topic = await kb_db.get_topic(topic_id) if llm is not None else None

    for claim, score, matched_names in matches:
        if llm is not None and topic is not None:
            try:
                verdict = await _classify_claim_topic_relevance(
                    llm, topic["name"], topic.get("description"), claim["canonical_text"], matched_names,
                )
            except Exception:
                verdict = {"relationship": "relevant", "confidence": 0.0}
            if (
                verdict.get("relationship") == "not_relevant"
                and (verdict.get("confidence") or 0.0) >= TOPIC_RELEVANCE_SUPPRESS_THRESHOLD
            ):
                continue  # confidently just an incidental entity mention -- skip, don't queue

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


async def _build_llm_client(config: Config) -> LLMClient:
    """Mirrors resolve_and_promote's exact client-construction pattern --
    one client per top-level call, closed by the caller when done."""
    extraction_base_url = config.kb.extraction_llm_base_url
    extraction_model = config.kb.extraction_llm_model or await detect_model(extraction_base_url)
    return LLMClient(Config(llm=LLMConfig(base_url=extraction_base_url, model=extraction_model, api_key="not-needed")))


async def generate_topic_suggestions(kb_db: KBDatabase, config: Config, topic_id: str) -> SuggestionResult:
    """Backfill: scan every claim in the KB (not just new ones) against the
    topic's current entity set. Safe to re-run — suggest_claim_for_topic/
    suggest_source_for_topic only ever create a *new* link row, never
    overwrite an existing attached/suggested/rejected one."""
    entity_names = await get_topic_entity_names(kb_db, topic_id)
    if not entity_names:
        return SuggestionResult()

    already_linked = await kb_db.get_topic_claim_ids(topic_id, link_status=None)
    matches = await kb_db.find_claims_by_entity_overlap(entity_names, exclude_claim_ids=already_linked)
    if not matches:
        return SuggestionResult()

    llm = await _build_llm_client(config)
    try:
        return await _suggest_claims_and_sources(kb_db, topic_id, matches, llm)
    finally:
        await llm.close()


async def check_claims_against_topics(
    kb_db: KBDatabase, config: Config, claim_ids: list[str],
) -> dict[str, SuggestionResult]:
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

    llm = await _build_llm_client(config)
    try:
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
                    kb_db, topic["id"], [(claim, score, sorted(overlap))], llm,
                )
                result.claims_suggested += sub_result.claims_suggested
                result.sources_suggested += sub_result.sources_suggested

            if result.claims_suggested or result.sources_suggested:
                results[topic["id"]] = result

        return results
    finally:
        await llm.close()
