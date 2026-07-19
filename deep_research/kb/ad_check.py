"""LLM-based ad/sponsor-content classification for newly-promoted claims.

Round 4 of the qwen3-14b-vs-30b eval (2026-07-12/13) found both models
promote real claims out of video sponsor segments -- "The Book makes the
perfect gift...", a cluster of Brilliant.org ad-read claims -- alongside
genuine source content, worse for 30B. Verification doesn't distinguish
these from real claims today: an ad claim just burns a full verification
pass and lands on "unverified" (no web evidence either way), indistinguishable
from a genuine claim nothing could be found for.

Classifies from the claim plus a bounded window of its source context, same
one-call-per-claim shape as trust.py's source classifier, and shares its risk
posture: a false "is_ad"
verdict permanently excludes a real claim from ever being auto-verified,
which is worse than leaving an actual ad claim eligible (worst case, it
just harmlessly resolves to "unverified" like any other unverifiable claim).
So this only ever acts on a confident verdict, and low-confidence/failed
classifications leave the claim untouched.
"""

import json
import re

from deep_research.config import Config, LLMConfig
from deep_research.kb.db import KBDatabase
from deep_research.kb.decision_log import record_decision
from deep_research.kb.extraction import detect_model
from deep_research.llm import LLMClient

AD_CHECK_SYSTEM_PROMPT = """/no_think
You are screening a single claim extracted from a knowledge-base source (an article,
video transcript, etc.) to decide whether it is genuine source content, or promotional /
sponsor / advertisement material -- e.g. a video sponsor read, a book/product plug, a
"use code X for Y% off" message, a call to subscribe/follow/buy.

The extracted claim can sound neutral in isolation (for example, "the host
drinks a meal-replacement shake for breakfast"). Use the surrounding source
context to catch first-person testimonials that lead into a product link,
discount code, price, free trial, or other call to action.

Return ONLY a JSON object: {"is_ad": true|false, "confidence": 0.0-1.0, "reasoning": "one short sentence"}
"""

# Below this bar, or on a parse failure / request exception, the claim is left
# alone -- see module docstring for why a false positive here is worse than a
# false negative.
AD_CHECK_CONFIDENCE_THRESHOLD = 0.7


def _parse_ad_check(content: str) -> dict:
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
    return {"is_ad": False, "confidence": 0.0, "reasoning": "could not parse model output"}


async def classify_claim_is_ad(
    llm: LLMClient, claim_text: str, source_context: str | None = None,
) -> dict:
    user_content = f"Claim: {claim_text}"
    if source_context:
        user_content += f"\n\nSurrounding source context:\n{source_context}"
    user_content += "\n\nClassify."
    messages = [
        {"role": "system", "content": AD_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    parsed = _parse_ad_check(content)
    parsed["_parse_success"] = parsed.get("reasoning") != "could not parse model output"
    return parsed


async def check_claims_for_ads(kb_db: KBDatabase, config: Config, claim_ids: list[str]) -> list[str]:
    """Best-effort: classifies each of `claim_ids` and sets
    verification_override='exclude' on any confidently flagged as ad/sponsor
    content, so it never burns a verification pass. Returns the claim_ids
    actually flagged. Never raises -- meant to be called right after
    resolve_and_promote returns, where a classification hiccup must not fail
    the extraction command itself (matches trust.py's set_trust_tier_if_missing
    posture). Skips any claim that already has a verification_override set,
    so this never overwrites a human's or another process's decision."""
    if not claim_ids:
        return []

    extraction_base_url = config.kb.extraction_llm_base_url
    try:
        extraction_model = config.kb.extraction_llm_model or await detect_model(extraction_base_url)
    except Exception:
        return []
    llm = LLMClient(Config(llm=LLMConfig(base_url=extraction_base_url, model=extraction_model, api_key="not-needed")))

    flagged = []
    try:
        for claim_id in claim_ids:
            claim = await kb_db.get_claim(claim_id)
            if claim is None or claim.get("verification_override") is not None:
                continue
            try:
                evidence_by_claim = await kb_db.get_claims_evidence_bulk([claim_id])
                contexts = [
                    row.get("context_excerpt") or row.get("excerpt_text")
                    for row in evidence_by_claim.get(claim_id, [])
                    if row.get("context_excerpt") or row.get("excerpt_text")
                ]
                source_context = "\n\n---\n\n".join(contexts[:2]) or None
                verdict = await classify_claim_is_ad(
                    llm, claim["canonical_text"], source_context=source_context,
                )
            except Exception:
                continue
            confidence = verdict.get("confidence") or 0.0
            will_exclude = bool(verdict.get("is_ad")) and confidence >= AD_CHECK_CONFIDENCE_THRESHOLD
            await record_decision(
                kb_db, "ad_check", "claim", claim_id,
                "excluded_as_ad" if will_exclude else "left_eligible",
                confidence=confidence, reasoning=verdict.get("reasoning"), model=extraction_model,
                parse_success=bool(verdict.get("_parse_success", True)),
                previous_state={"verification_override": claim.get("verification_override")},
                resulting_state={"verification_override": "exclude" if will_exclude else claim.get("verification_override")},
                reversible=will_exclude,
            )
            if will_exclude:
                await kb_db.set_claim_verification_override(claim_id, "exclude")
                flagged.append(claim_id)
    finally:
        await llm.close()

    return flagged
