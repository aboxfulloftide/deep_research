"""Claim verification (build order step 6): check a claim against the KB's own
data first, then the internet if internal coverage is thin — bounded by a
per-claim budget so cost cannot explode.

See "Verification Policy and Budget" in PLAN_KB_ARCHITECTURE.md:
- triggers: manual request, or importance_score above a threshold
- budget: at most `verification_max_web_searches` web searches,
  `verification_max_sources_examined` additional sources examined, one
  comparison pass per examined source
- stop conditions: 2 independent supporting sources -> supported; a
  contradiction -> contradicted, recorded as a conflict (not resolved inside
  the budget); budget exhausted -> stays unverified, but
  verification_attempted_at is set so a report can tell "never checked" from
  "checked, inconclusive"

The one new piece of infrastructure this needs beyond steps 2-5: embedding
similarity alone can't tell "same fact, different wording" apart from "same
topic, different/conflicting numbers" (e.g. "revenue grew 50%" vs "revenue
grew 500%" embed as nearly identical). An LLM comparison pass handles that —
validated directly against known claim pairs (including a same-fact-different-
units case requiring real unit-conversion reasoning, and a synthetic
contradiction) before building this pipeline around it.
"""

import json
import re
from dataclasses import dataclass, field

from deep_research.config import Config, LLMConfig
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import cosine, embed_texts
from deep_research.kb.extraction import detect_model, run_extraction
from deep_research.kb.ingest import ingest_web_page
from deep_research.kb.resolution import resolve_and_promote
from deep_research.kb.storage import SnapshotStore
from deep_research.llm import LLMClient
from deep_research.tools.search import web_search

COMPARISON_SYSTEM_PROMPT = """/no_think
You are comparing two factual claims extracted from different sources to determine their relationship.

Classify the relationship as exactly one of:
- "supports": the second claim independently corroborates the same underlying fact as the first (may use different wording, units, or approximations, but is consistent once you account for that)
- "contradicts": the second claim states something that actually conflicts with the first about the same specific fact (a different number, outcome, or claim that cannot both be true)
- "unrelated": the second claim is not actually about the same specific fact as the first, even if topically similar

Do the arithmetic/unit conversion yourself if needed before deciding — do not assume different-looking numbers contradict without checking whether they are equivalent.

Return ONLY a JSON object: {"relationship": "supports"|"contradicts"|"unrelated", "confidence": 0.0-1.0, "reasoning": "one short sentence"}
"""

# Only candidates at least this similar are worth an LLM comparison pass at all.
CANDIDATE_SIMILARITY_FLOOR = 0.80


@dataclass
class VerificationResult:
    status: str  # "supported" | "contradicted" | "mixed" | "unverified" | "skipped"
    claim_id: str
    supports_found: int = 0
    contradicts_found: int = 0
    sources_examined: int = 0
    web_searches_used: int = 0
    contradiction_candidate_ids: list[str] = field(default_factory=list)


def _parse_json_object(content: str) -> dict:
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
    return {"relationship": "unrelated", "confidence": 0.0, "reasoning": "could not parse model output"}


async def _classify_relationship(llm: LLMClient, claim_text: str, other_text: str) -> dict:
    messages = [
        {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim A: {claim_text}\nClaim B: {other_text}\n\nClassify."},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    return _parse_json_object(content)


async def _rank_candidates_by_similarity(
    config: Config, claim: dict, candidates: list[dict],
) -> list[tuple[dict, float]]:
    """Step 8: claims are embedded once at creation time (resolution.py's
    embed_new_claims) and persisted, so both the target claim and its
    candidates almost always already carry an embedding here -- this only
    falls back to a live embed_texts call for the rare claim that predates
    that feature and hasn't been backfilled yet."""
    if not candidates:
        return []
    base_url = config.kb.embedding_base_url
    model = config.kb.embedding_model

    target_vec = claim["embedding"].to_list() if claim.get("embedding") is not None else None
    missing = [c for c in candidates if c.get("embedding") is None]
    to_embed_texts = ([claim["canonical_text"]] if target_vec is None else []) + [
        c["canonical_text"] for c in missing
    ]
    if to_embed_texts:
        fresh_vectors = await embed_texts(to_embed_texts, base_url, model)
        if target_vec is None:
            target_vec = fresh_vectors[0]
            fresh_vectors = fresh_vectors[1:]
        for c, v in zip(missing, fresh_vectors):
            c["_vec"] = v

    scored = [
        (c, cosine(target_vec, c["_vec"] if "_vec" in c else c["embedding"].to_list()))
        for c in candidates
    ]
    scored = [(c, s) for c, s in scored if s >= CANDIDATE_SIMILARITY_FLOOR]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


class _Budget:
    """Tracks the per-claim verification budget and the stop conditions."""

    def __init__(self, max_sources: int, max_searches: int):
        self.max_sources = max_sources
        self.max_searches = max_searches
        self.sources_examined = 0
        self.web_searches_used = 0
        self.supports = 0
        self.contradicts = 0

    def sources_remaining(self) -> bool:
        return self.sources_examined < self.max_sources

    def searches_remaining(self) -> bool:
        return self.web_searches_used < self.max_searches

    def should_stop(self) -> bool:
        return self.contradicts > 0 or self.supports >= 2 or not self.sources_remaining()

    def final_status(self) -> str:
        if self.contradicts > 0 and self.supports > 0:
            return "mixed"
        if self.contradicts > 0:
            return "contradicted"
        if self.supports >= 2:
            return "supported"
        return "unverified"


async def _examine_candidates(
    kb_db: KBDatabase, config: Config, llm: LLMClient, claim: dict,
    ranked_candidates: list[tuple[dict, float]], budget: _Budget, examined_source_ids: set[str],
    contradiction_ids: list[str],
) -> None:
    """Runs the LLM comparison pass over ranked candidates, one distinct
    source at a time (the budget counts *sources*, not individual claims —
    several candidate claims from the same source count as one)."""
    for other_claim, similarity in ranked_candidates:
        if budget.should_stop():
            return
        other_source_ids = await kb_db.get_claim_source_ids(other_claim["id"])
        new_sources = other_source_ids - examined_source_ids
        if not new_sources:
            continue  # every source backing this candidate was already examined
        examined_source_ids.update(new_sources)
        budget.sources_examined += 1

        try:
            result = await _classify_relationship(llm, claim["canonical_text"], other_claim["canonical_text"])
        except Exception:
            # A transient LLM failure on this one comparison shouldn't abort
            # the whole verification and lose everything found so far —
            # counts against the source-examined budget like any other
            # inconclusive comparison, but the loop moves on to the next
            # candidate instead of raising out of verify_claim entirely.
            continue
        relationship = result.get("relationship")
        if relationship == "supports":
            budget.supports += 1
        elif relationship == "contradicts":
            budget.contradicts += 1
            _, created = await kb_db.add_claim_contradiction_candidate(
                claim["id"], other_claim["id"], result.get("confidence", 0.5),
                "llm_comparison", result.get("reasoning"),
            )
            if created:
                contradiction_ids.append(other_claim["id"])


async def verify_claim(
    kb_db: KBDatabase, config: Config, claim_id: str, force: bool = False,
) -> VerificationResult:
    claim = await kb_db.get_claim(claim_id)
    if claim is None:
        raise ValueError(f"No such claim: {claim_id}")
    if claim["verification_attempted_at"] is not None and not force:
        return VerificationResult(status="skipped", claim_id=claim_id)

    budget = _Budget(config.kb.verification_max_sources_examined, config.kb.verification_max_web_searches)
    own_source_ids = await kb_db.get_claim_source_ids(claim_id)
    examined_source_ids = set(own_source_ids)
    contradiction_ids: list[str] = []

    extraction_base_url = config.kb.extraction_llm_base_url
    extraction_model = config.kb.extraction_llm_model or await detect_model(extraction_base_url)
    llm = LLMClient(Config(llm=LLMConfig(base_url=extraction_base_url, model=extraction_model, api_key="not-needed")))

    try:
        # Phase 1: search the KB's own data first (decision 23, hybrid retrieval).
        internal_candidates = await kb_db.get_claims_independent_of(list(own_source_ids), claim_id)
        ranked = await _rank_candidates_by_similarity(config, claim, internal_candidates)
        await _examine_candidates(kb_db, config, llm, claim, ranked, budget, examined_source_ids, contradiction_ids)

        # Phase 2: fall back to the internet only if internal coverage was thin
        # and budget remains.
        snapshot_store = SnapshotStore(config.kb_snapshot_dir)
        while not budget.should_stop() and budget.searches_remaining():
            budget.web_searches_used += 1
            try:
                results = await web_search(claim["canonical_text"], config)
            except Exception:
                break
            if not results:
                break

            for result in results:
                if budget.should_stop() or not budget.sources_remaining():
                    break
                ingest_result = await ingest_web_page(result.url, config, kb_db, snapshot_store)
                if ingest_result.status == "failed" or ingest_result.source_id in examined_source_ids:
                    continue

                try:
                    source = await kb_db.get_source(ingest_result.source_id)
                    version = await kb_db.get_source_version(ingest_result.version_id)
                    chunk_result = await build_artifact_for_version(
                        kb_db, snapshot_store, source, version, config=config,
                    )
                    if chunk_result.chunk_count == 0:
                        continue

                    artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
                    extraction_result = await run_extraction(kb_db, config, artifacts[0]["id"])
                    if extraction_result.observation_count == 0:
                        continue
                    await resolve_and_promote(kb_db, config, extraction_result.extraction_run_id)

                    new_source_claims = await kb_db.get_claims_independent_of(
                        list(examined_source_ids), claim_id,
                    )
                    new_ranked = await _rank_candidates_by_similarity(config, claim, new_source_claims)
                    await _examine_candidates(
                        kb_db, config, llm, claim, new_ranked, budget, examined_source_ids, contradiction_ids,
                    )
                except Exception:
                    # One bad web-fallback source (unparseable page, extraction
                    # LLM hiccup) shouldn't abort the whole verification and
                    # lose everything found so far -- treat it like any other
                    # unusable source and move on to the next search result.
                    continue
    finally:
        await llm.close()

    status = budget.final_status()
    notes = {
        "supports_found": budget.supports,
        "contradicts_found": budget.contradicts,
        "sources_examined": budget.sources_examined,
        "web_searches_used": budget.web_searches_used,
    }
    await kb_db.update_claim_verification(claim_id, status, notes)

    return VerificationResult(
        status=status, claim_id=claim_id, supports_found=budget.supports,
        contradicts_found=budget.contradicts, sources_examined=budget.sources_examined,
        web_searches_used=budget.web_searches_used, contradiction_candidate_ids=contradiction_ids,
    )
