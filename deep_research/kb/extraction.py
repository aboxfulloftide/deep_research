"""Claim/entity/event/metric extraction pipeline (build order step 4).

Reuses the exact extraction approach validated in the step-0 spike
(spike/FINDINGS.md): a claim-per-chunk JSON schema, "/no_think" to suppress
reasoning tokens, and whitespace-normalized substring matching to link each
claim back to a verbatim quote in its source chunk. Extended here with an
optional metrics[] field per decision 22 ("key metrics/numbers when present"
is part of the default always-extract set) — not spike-validated on its own,
but the same shape and risk profile as the fields that were validated.

Extraction only writes extracted_observations (raw model output). Turning
those into canonical claims/entities/events/metrics is resolve_and_promote()
in deep_research/kb/resolution.py — kept separate so noisy first-pass output
never lands directly in the curated tables.
"""

import hashlib
import json
import re
from dataclasses import dataclass

import httpx

from deep_research.config import Config, LLMConfig
from deep_research.kb.chunking import find_quote
from deep_research.kb.db import KBDatabase
from deep_research.llm import LLMClient

PROMPT_NAME = "claim_extraction"
PROMPT_VERSION = "v2-with-metrics"
EXTRACTION_SCHEMA_VERSION = "v1"

EXTRACTION_SYSTEM_PROMPT = """/no_think
You are a claim extraction engine for a knowledge base. You will be given one chunk of text from a source. Extract atomic factual claims made in this chunk.

For each claim, output an object with these exact fields:
- claim_text: a single atomic factual statement, in your own words, not a blob of multiple facts
- claim_type: one of "fact", "event_fact", "economic", "historical", "product_spec", "quote"
- entities: array of {"name": str, "type": one of "person","organization","product","location","concept"} mentioned in the claim
- event: {"title": str, "date": str or null} if the claim describes a dated/time-bound happening, otherwise null
- metrics: array of {"name": str, "value": number or string, "unit": str or null, "currency": str or null} for any specific structured numeric/economic/spec value the claim states (a dollar amount, a percentage, a count, a spec number); empty array if the claim has no such value
- supporting_quote: a short VERBATIM excerpt (max ~30 words) copied EXACTLY from the chunk text that supports this claim. Do not paraphrase this field.
- confidence: your confidence 0.0-1.0 that this claim is accurately extracted from the text
- importance: your estimate 0.0-1.0 of how important/central this claim is to the source's overall point

Rules:
- Only extract claims actually present in the chunk. Do not invent facts, numbers, or dates not present in the text.
- Keep each claim atomic: one fact per claim.
- If the chunk has no extractable claims, return an empty array.
- Return ONLY a JSON array. No prose, no markdown fences, no explanation.
"""


@dataclass
class ExtractionRunResult:
    status: str  # "extracted" | "partial" | "unchanged" | "empty"
    extraction_run_id: str | None = None
    observation_count: int = 0
    chunk_count: int = 0
    failed_chunk_count: int = 0


async def detect_model(base_url: str) -> str:
    """Ask the llama.cpp server which model it currently has loaded, so
    extraction doesn't hardcode a model path that may change between runs."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base_url}/models")
        resp.raise_for_status()
        data = resp.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models reported by server at {base_url}")
    return models[0]["id"]


def _parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            data = json.loads(match.group(1))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    return []


def _run_signature(model: str) -> str:
    payload = json.dumps(
        {"model": model, "prompt_name": PROMPT_NAME, "prompt_version": PROMPT_VERSION,
         "schema_version": EXTRACTION_SCHEMA_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def run_extraction(
    kb_db: KBDatabase, config: Config, artifact_id: str, force: bool = False,
) -> ExtractionRunResult:
    artifact = await kb_db.get_artifact(artifact_id)
    if artifact is None:
        raise ValueError(f"No such artifact: {artifact_id}")

    chunks = await kb_db.list_chunks(artifact_id)
    if not chunks:
        return ExtractionRunResult(status="empty", chunk_count=0)

    base_url = config.kb.extraction_llm_base_url
    model = config.kb.extraction_llm_model or await detect_model(base_url)
    run_signature = _run_signature(model)

    if not force:
        existing = await kb_db.find_extraction_run_by_signature(artifact_id, run_signature)
        if existing is not None:
            observations = await kb_db.list_observations(existing["id"])
            return ExtractionRunResult(
                status="unchanged", extraction_run_id=existing["id"],
                observation_count=len(observations), chunk_count=len(chunks),
            )

    run = await kb_db.create_extraction_run(
        artifact_id=artifact_id, run_signature=run_signature, model_id=model,
        prompt_name=PROMPT_NAME, prompt_version=PROMPT_VERSION,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION, runtime="llama.cpp",
        chunk_count=len(chunks),
    )

    llm_config = Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed"))
    llm = LLMClient(llm_config)

    observation_count = 0
    failed_chunk_count = 0
    try:
        for chunk in chunks:
            messages = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Chunk text:\n\n{chunk['chunk_text']}"},
            ]
            try:
                resp = await llm.chat(messages)
                content = resp["choices"][0]["message"]["content"] or ""
            except Exception:
                # Decision 24: keep partial results, mark the run incomplete, retry later.
                failed_chunk_count += 1
                continue

            for claim in _parse_json_array(content):
                if not isinstance(claim, dict) or not claim.get("claim_text"):
                    continue
                quote = claim.get("supporting_quote", "")
                match = find_quote(quote, chunk["chunk_text"])
                match_type, q_start, q_end = match or (None, None, None)
                await kb_db.add_observation(
                    extraction_run_id=run["id"], artifact_chunk_id=chunk["id"],
                    raw_text=claim["claim_text"], raw_payload=claim,
                    confidence=claim.get("confidence"), importance_score=claim.get("importance"),
                    char_start=q_start, char_end=q_end, quote_match_type=match_type,
                )
                observation_count += 1
    finally:
        await llm.close()

    status = "partial" if failed_chunk_count else "completed"
    await kb_db.complete_extraction_run(run["id"], observation_count, status=status)

    return ExtractionRunResult(
        status="extracted" if not failed_chunk_count else "partial",
        extraction_run_id=run["id"], observation_count=observation_count,
        chunk_count=len(chunks), failed_chunk_count=failed_chunk_count,
    )
