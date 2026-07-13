"""LLM-based trust-tier classification for newly-ingested sources.

trust_tier_code was previously never auto-set at all -- every source needed
a human to explicitly pass it at ingest time, which essentially never
happened in practice (measured against the real KB: 95 of 96 sources had no
tier). That leaves recompute_preferred_source's "prefer the most-trustworthy
corroborating source for a claim" ranking (decision 27, see kb/merge.py)
with nothing to actually rank by for almost every claim.

Classifies from the source's URL/canonical_uri and title only, not a full
content read -- enough signal for the same judgment a human skimming a
source list would make ("nytimes.com" -> reputable reporting, "sec.gov" ->
official, "reddit.com" -> user-generated), without an extra snapshot-file
read on the ingest hot path. Falls back to leaving trust_tier unset on any
parse failure or low-confidence verdict -- a wrong tier actively corrupts
the preferred-source ranking, which is worse than no ranking signal at all,
so this only ever writes a tier it's actually confident about.
"""

import json
import re

from deep_research.config import Config, LLMConfig
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import detect_model
from deep_research.llm import LLMClient

TRUST_TIER_CODES = ("official", "reputable_reporting", "secondary_analysis", "user_generated")

TRUST_TIER_SYSTEM_PROMPT = """/no_think
You are classifying a source's trust tier for a knowledge base, based only on its URL and title.

Classify as exactly one of:
- "official": primary/official statements -- government or regulatory filings, court documents, company press releases, official organization statements
- "reputable_reporting": established news organizations and professional journalism
- "secondary_analysis": analysis, commentary, blogs, trade press, or aggregation/summary of primary reporting
- "user_generated": forums, social media, or other unvetted user-generated content

If the URL/title alone don't give you enough to tell confidently, say so with low confidence rather than guessing.

Return ONLY a JSON object: {"tier": "official"|"reputable_reporting"|"secondary_analysis"|"user_generated", "confidence": 0.0-1.0, "reasoning": "one short sentence"}
"""

# Below this bar, or on a parse failure / request exception, the source is
# left untiered rather than risk polluting the preferred-source ranking with
# a wrong guess -- unlike the entity/claim-duplicate vetting (where an
# ambiguous verdict safely falls through to a human review queue), there is
# no queue here, so the only safe fallback is "don't set anything."
TRUST_TIER_LLM_CONFIDENCE_THRESHOLD = 0.6


def _parse_trust_tier_classification(content: str) -> dict:
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
    return {"tier": None, "confidence": 0.0, "reasoning": "could not parse model output"}


async def classify_source_trust_tier(llm: LLMClient, canonical_uri: str, title: str | None) -> dict:
    messages = [
        {"role": "system", "content": TRUST_TIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"URL: {canonical_uri}\nTitle: {title or '(none)'}\n\nClassify."},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    return _parse_trust_tier_classification(content)


async def set_trust_tier_if_missing(kb_db: KBDatabase, config: Config, source_id: str) -> str | None:
    """Best-effort: classifies and persists a trust tier for `source_id` if
    it doesn't already have one. Returns the tier code actually set, or None
    if nothing changed (already tiered, low-confidence verdict, or the LLM
    call itself failed). Never raises -- meant to be called right after an
    ingest call returns, where a classification hiccup must not fail the
    ingest response itself. Builds and closes its own LLMClient, matching
    verify_claim's pattern -- fine for the single-source-per-call case this
    is meant for; a bulk backfill over many sources should instead call
    classify_source_trust_tier directly with one shared client (see
    scripts/backfill_trust_tiers.py-style one-off usage)."""
    source = await kb_db.get_source(source_id)
    if source is None or source.get("trust_tier_id") is not None:
        return None

    extraction_base_url = config.kb.extraction_llm_base_url
    extraction_model = config.kb.extraction_llm_model or await detect_model(extraction_base_url)
    llm = LLMClient(Config(llm=LLMConfig(base_url=extraction_base_url, model=extraction_model, api_key="not-needed")))
    try:
        verdict = await classify_source_trust_tier(llm, source["canonical_uri"], source.get("title"))
    except Exception:
        return None
    finally:
        await llm.close()

    tier = verdict.get("tier")
    confidence = verdict.get("confidence") or 0.0
    if tier not in TRUST_TIER_CODES or confidence < TRUST_TIER_LLM_CONFIDENCE_THRESHOLD:
        return None

    await kb_db.set_source_trust_tier(source_id, tier)
    return tier
