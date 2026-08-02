"""Judges whether an already-extracted claim is specific enough to ever be
worth verifying, distinct from both extraction-time filters (extraction.py's
has_unresolved_subject, a mechanical regex over pronoun-led subjects) and
verification itself (whether a specific claim turns out true/false).

Built after two purely mechanical heuristics (a proper-noun regex, then an
entity-tag check) both produced real false positives in both directions on
live KB data -- flagging genuinely specific claims ("eBay went public on 24th
September 1998", "The Recession of 1937-1938 occurred from 1937 to 1938")
while missing others, because "is this claim specific enough to realistically
find evidence for" is a judgment call, not a pattern. An LLM classification
pass is slower and costs real calls, but is the only reliable way to make
that call at scale.
"""

import asyncio
import json
import re

from deep_research.config import Config, LLMConfig
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import detect_model
from deep_research.llm import LLMClient

CLAIM_SPECIFICITY_PROMPT = """/no_think
You are judging whether a factual claim is specific enough to realistically verify against independent sources.

A claim FAILS this check if it lacks any named entity, event, place, or
precise detail that distinguishes it from countless similar situations --
e.g. "Businesses closed in 1837" could describe almost any economic downturn
in any country and year. A claim also fails if it is an opinion or
rhetorical statement rather than a factual assertion (e.g. "The world before
1913 was not a libertarian paradise"), or if it is self-referential/meta
about a video or article rather than a fact about the world (e.g. "A video
discusses what happened in 1971").

A claim PASSES if a person could realistically search for it and expect to
find (or fail to find) real, specific corroborating evidence -- a precise
number, date, or named event is enough even without a named person or
organization (e.g. "Military spending in 1800 was roughly $2.5 million"
passes; "The Recession of 1937-1938 occurred from 1937 to 1938" passes,
since that recession is a specific, identifiable historical event even
though no organization is named).

Return ONLY a JSON object: {"specific": true or false, "reason": "..."}
"""

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_specificity_response(content: str) -> tuple[bool, str]:
    """Fails safe: an unparseable or malformed response keeps the claim
    (returns specific=True) rather than deleting it on an ambiguous
    judgment -- a false "keep" just leaves a claim unverified a little
    longer, while a false "delete" destroys data."""
    content = (content or "").strip()
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(content)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(parsed, dict) or not isinstance(parsed.get("specific"), bool):
        return True, "unparseable classifier response; kept by default"
    return parsed["specific"], str(parsed.get("reason") or "")


async def classify_claim_specificity(llm: LLMClient, claim_text: str) -> tuple[bool, str]:
    messages = [
        {"role": "system", "content": CLAIM_SPECIFICITY_PROMPT},
        {"role": "user", "content": f"Claim: {claim_text}"},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    return _parse_specificity_response(content)


async def sweep_claim_specificity(
    kb_db: KBDatabase, config: Config, *, concurrency: int = 3,
    dry_run: bool = False, on_result=None,
) -> dict:
    """Classifies every status='unverified' claim and deletes (via
    delete_claim_cascade) the ones judged too generic to ever realistically
    verify -- run once as a deliberate cleanup pass, not wired into the
    routine extraction/verification pipeline. Scoped to 'unverified' only:
    a claim that already settled (supported/contradicted) represents real
    completed work regardless of how generic its text reads.

    dry_run=True classifies and reports without deleting, for sanity-
    checking the classifier's real judgments on a sample before trusting it
    with the full backlog.

    `on_result(claim, status, reason)` is called (synchronously) after each
    claim, status one of "kept"/"removed"/"failed"."""
    all_claims = await kb_db.list_claims(limit=10000)
    unverified = [c for c in all_claims if c["status"] == "unverified"]

    base_url = config.kb.verification_llm_base_url or config.kb.extraction_llm_base_url
    model = (
        config.kb.verification_llm_model or config.kb.extraction_llm_model
        or await detect_model(base_url)
    )
    llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))

    semaphore = asyncio.Semaphore(max(1, concurrency))
    counts = {"kept": 0, "removed": 0, "failed": 0}

    async def process(claim: dict) -> None:
        async with semaphore:
            try:
                specific, reason = await classify_claim_specificity(llm, claim["canonical_text"])
            except Exception as exc:
                counts["failed"] += 1
                if on_result:
                    on_result(claim, "failed", str(exc))
                return
            if specific:
                counts["kept"] += 1
                status = "kept"
            else:
                counts["removed"] += 1
                status = "removed"
                if not dry_run:
                    await kb_db.delete_claim_cascade(claim["id"])
            if on_result:
                on_result(claim, status, reason)

    try:
        await asyncio.gather(*(process(c) for c in unverified))
    finally:
        await llm.close()

    return {"total": len(unverified), "dry_run": dry_run, **counts}
