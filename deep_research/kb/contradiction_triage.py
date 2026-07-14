"""LLM assistance for reviewing contradictions without deciding them."""

import json
import re

from deep_research.config import Config, LLMConfig
from deep_research.kb.db import KBDatabase
from deep_research.llm import LLMClient

TRIAGE_PROMPT = """/no_think
Help a human review two claims that may contradict. Never decide which claim
is true and never change either claim. Compare evidence quality and return
only JSON: {"recommendation":"review A first"|"review B first"|"insufficient evidence","reasoning":"one short sentence"}.
Prefer primary and official evidence over secondary summaries.
"""


def _parse(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"recommendation": "insufficient evidence", "reasoning": "Model response could not be parsed."}


async def triage_contradiction(kb_db: KBDatabase, config: Config, candidate_id: str) -> dict:
    candidate = await kb_db.get_resolution_candidate(candidate_id)
    if not candidate or candidate["candidate_type"] != "claim_contradiction":
        raise ValueError("candidate must be a claim_contradiction")
    left, right = await kb_db.get_claim(candidate["left_claim_id"]), await kb_db.get_claim(candidate["right_claim_id"])
    if not left or not right:
        raise ValueError("both contradiction claims must exist")
    base_url = config.kb.verification_llm_base_url or config.kb.extraction_llm_base_url
    model = config.kb.verification_llm_model or config.kb.extraction_llm_model or ""
    llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model)))
    try:
        response = await llm.chat([
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": json.dumps({
                "claim_a": left["canonical_text"], "claim_a_evidence": [e.get("excerpt_text") for e in await kb_db.list_claim_evidence(left["id"])],
                "claim_b": right["canonical_text"], "claim_b_evidence": [e.get("excerpt_text") for e in await kb_db.list_claim_evidence(right["id"])],
            })},
        ])
        parsed = _parse(response["choices"][0]["message"].get("content") or "")
    finally:
        await llm.close()
    recommendation = parsed.get("recommendation")
    if recommendation not in {"review A first", "review B first", "insufficient evidence"}:
        recommendation = "insufficient evidence"
    reasoning = str(parsed.get("reasoning") or "No reasoning supplied.")
    result = await kb_db.set_resolution_candidate_triage(candidate_id, recommendation, reasoning, model or None)
    await kb_db.record_decision(
        "contradiction_triage", "resolution_candidate", candidate_id, recommendation,
        related_ids=[left["id"], right["id"]], reasoning=reasoning, model=model or None,
        parse_success="could not be parsed" not in reasoning,
        resulting_state={"triage_recommendation": recommendation}, reversible=False,
    )
    return result
