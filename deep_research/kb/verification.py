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

import asyncio
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from deep_research.config import Config, LLMConfig
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.canonical import is_social_media_domain
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

If additional context is given, it is a human's note about what specifically to look for -- weigh whether the second claim actually addresses that angle, not just whether it's topically similar to the first claim.

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
    supporting_claim_ids: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


@contextmanager
def _timed(timings: dict[str, float], key: str):
    """Accumulates wall-clock seconds spent in a named phase across however
    many times it's entered (e.g. one entry per web-fallback source examined)
    -- coarse enough to say "which phase dominates" (GPU-bound LLM/embedding
    calls vs. network-bound search/scrape vs. DB) without threading a timing
    object through every function signature in the pipeline."""
    start = time.monotonic()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.monotonic() - start)


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


async def _classify_relationship(llm: LLMClient, claim_text: str, other_text: str, context: str | None = None) -> dict:
    user_content = f"Claim A: {claim_text}\nClaim B: {other_text}"
    if context:
        user_content += f"\nAdditional context (what to specifically look for): {context}"
    user_content += "\n\nClassify."
    messages = [
        {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    return _parse_json_object(content)


SEARCH_QUERY_SUGGESTION_PROMPT = """/no_think
You are choosing a web search query to help verify a factual claim.

Suggest ONE web search query likely to surface independent corroborating or
contradicting sources -- for example naming the specific number/entity/date/
organization involved instead of searching the whole sentence verbatim,
dropping filler words, or trying alternate terminology a source might
actually use.

If additional context is given, it is a human's note about a specific angle
to look for -- make sure the query actually targets that, not just the
literal claim text. If prior queries are listed, suggest something
meaningfully different from all of them, not a minor rephrasing of one.

Return ONLY a JSON object: {"query": "..."}
"""


async def _suggest_search_query(
    llm: LLMClient, claim_text: str, tried_queries: list[str], context: str | None = None,
) -> str:
    """Used whenever a plain search of the literal claim text isn't good
    enough on its own: from the second web-search attempt onward (within one
    verify_claim run, or on a retry of a previously-inconclusive claim, see
    UNVERIFIED_RETRY_COOLDOWN_HOURS) -- repeating the exact same query
    almost never turns up anything new, since a search engine returns the
    same top results for the same string -- and from the very first attempt
    whenever the claim carries added verification_context, since the literal
    claim text alone may not even mention the angle the context asks for
    (e.g. claim "industrial buildings use more electricity than residential"
    + context "compare against datacenter usage" -- a plain search of the
    claim text would never surface a datacenter-specific comparison). Falls
    back to the raw claim text (the original, always-safe default) on any
    parse failure or an empty suggestion."""
    parts = [f"Claim: {claim_text}"]
    if context:
        parts.append(f"Additional context (what to specifically look for): {context}")
    if tried_queries:
        tried = "\n".join(f"- {q}" for q in tried_queries)
        parts.append(f"Queries already tried (do not repeat these):\n{tried}")
    parts.append("Suggest a query.")
    messages = [
        {"role": "system", "content": SEARCH_QUERY_SUGGESTION_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    resp = await llm.chat(messages)
    content = resp["choices"][0]["message"]["content"] or ""
    parsed = _parse_json_object(content)
    query = parsed.get("query")
    return query.strip() if isinstance(query, str) and query.strip() else claim_text


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
        try:
            fresh_vectors = await embed_texts(to_embed_texts, base_url, model)
        except Exception:
            # Best-effort, same as every other embed_texts call site (embed_new_claims,
            # build_artifact_for_version): a transient/unreachable embedding backend
            # shouldn't crash the whole verification and lose whatever this run already
            # found -- degrade to "no internal candidates ranked this round" instead,
            # letting verify_claim fall through to the web-fallback phase.
            return []
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


async def _rank_chunks_by_similarity(
    config: Config, claim: dict, chunks: list[dict],
) -> list[tuple[dict, float]]:
    """Ranks a freshly-scraped page's chunks by relevance to the claim being
    verified, so extraction only has to run on the handful that might
    actually matter -- not the whole page. Chunks are embedded as part of
    chunking (build_artifact_for_version), so this is normally free; falls
    back to a live embed_texts call only for chunks an unreachable embedding
    backend left without one. No similarity floor here (unlike
    _rank_candidates_by_similarity) -- the caller takes a fixed top-N
    regardless of absolute score, since even a middling match is worth one
    cheap comparison and there's no risk of polluting the KB with a bad
    resolution_candidates row the way ranking claims-to-merge would."""
    if not chunks:
        return []
    base_url = config.kb.embedding_base_url
    model = config.kb.embedding_model

    target_vec = claim["embedding"].to_list() if claim.get("embedding") is not None else None
    missing = [c for c in chunks if c.get("embedding") is None]
    to_embed_texts = ([claim["canonical_text"]] if target_vec is None else []) + [
        c["chunk_text"] for c in missing
    ]
    if to_embed_texts:
        try:
            fresh_vectors = await embed_texts(to_embed_texts, base_url, model)
        except Exception:
            return []
        if target_vec is None:
            target_vec = fresh_vectors[0]
            fresh_vectors = fresh_vectors[1:]
        for c, v in zip(missing, fresh_vectors):
            c["_vec"] = v

    scored = [
        (c, cosine(target_vec, c["_vec"] if "_vec" in c else c["embedding"].to_list()))
        for c in chunks
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


class _RunSearchBudget:
    """One atomic external-search allowance shared by a whole batch.

    Individual claim checks already have their own small search budget. This
    additional guard prevents concurrency from multiplying those allowances
    into an unbounded nightly/provider bill.
    """

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0
        self._lock = asyncio.Lock()

    async def reserve(self) -> bool:
        if self.limit is None:
            return True
        async with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


class _Budget:
    """Tracks the per-claim verification budget and the stop conditions.

    The source budget is split per phase (internal KB search vs. web
    fallback) rather than shared. It used to be one shared counter, which
    meant a claim with a few weak/irrelevant internal matches could exhaust
    the entire "sources examined" budget in phase 1 and never get to try
    the web at all -- discovered from a real claim ("JP Morgan estimates
    more than $6 trillion...") that came back `unverified` with
    `web_searches_used: 0`: three internal candidates ate the whole budget
    before phase 2 ever ran, even though a plain web search turned up
    multiple corroborating pages immediately.
    """

    def __init__(self, max_sources: int, max_searches: int):
        self.max_sources = max_sources
        self.max_searches = max_searches
        self.internal_sources_examined = 0
        self.external_sources_examined = 0
        self.web_searches_used = 0
        self.supports = 0
        # Evidence quality matters: an official corroboration can settle a
        # claim, whereas weaker sources must accumulate. `supports` remains a
        # count for backwards-compatible reporting; this is the verdict gate.
        self.support_weight = 0.0
        self.contradicts = 0

    @property
    def sources_examined(self) -> int:
        return self.internal_sources_examined + self.external_sources_examined

    def sources_remaining(self, phase: str = "internal") -> bool:
        examined = self.internal_sources_examined if phase == "internal" else self.external_sources_examined
        return examined < self.max_sources

    def record_source_examined(self, phase: str = "internal") -> None:
        if phase == "internal":
            self.internal_sources_examined += 1
        else:
            self.external_sources_examined += 1

    def searches_remaining(self) -> bool:
        return self.web_searches_used < self.max_searches

    def should_stop(self) -> bool:
        return self.contradicts > 0 or self.support_weight >= 1.0 or self.supports >= 2

    def final_status(self) -> str:
        if self.contradicts > 0 and self.supports > 0:
            return "mixed"
        if self.contradicts > 0:
            return "contradicted"
        if self.support_weight >= 1.0 or self.supports >= 2:
            return "supported"
        return "unverified"


async def _examine_candidates(
    kb_db: KBDatabase, config: Config, llm: LLMClient, claim: dict,
    ranked_candidates: list[tuple[dict, float]], budget: _Budget, examined_source_ids: set[str],
    contradiction_ids: list[str], supporting_ids: list[str] | None = None, phase: str = "internal",
) -> None:
    """Runs the LLM comparison pass over ranked candidates, one distinct
    source at a time (the budget counts *sources*, not individual claims —
    several candidate claims from the same source count as one). `phase`
    selects which half of the source budget this call draws from (internal
    KB search vs. web fallback), so one phase running out doesn't block the
    other from being tried at all. claim["verification_context"], if set, is
    forwarded to every comparison so the LLM weighs whether a candidate
    actually addresses the angle a human flagged, not just topical overlap."""
    for other_claim, similarity in ranked_candidates:
        if budget.should_stop() or not budget.sources_remaining(phase):
            return
        other_source_ids = await kb_db.get_claim_source_ids(other_claim["id"])
        new_sources = other_source_ids - examined_source_ids
        if not new_sources:
            continue  # every source backing this candidate was already examined
        examined_source_ids.update(new_sources)
        budget.record_source_examined(phase)

        other_sources = [s for s in [await kb_db.get_source(sid) for sid in other_source_ids] if s is not None]
        if other_sources and all(is_social_media_domain(s["canonical_uri"]) for s in other_sources):
            # Reddit/Instagram/Facebook are unvetted user-generated content --
            # fine to read, but a claim whose *only* evidence is one of these
            # can never settle a verification by itself. Still counts against
            # the source-examined budget above (we did look at it), but never
            # worth an LLM comparison call.
            continue

        try:
            result = await _classify_relationship(
                llm, claim["canonical_text"], other_claim["canonical_text"], claim.get("verification_context"),
            )
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
            # Default unknown sources to 0.5, preserving the prior
            # two-independent-source behavior. Explicitly classified tiers
            # use their trust ranking, so one official source (1.0) suffices.
            tier_weights = {"official": 1.0, "reputable_reporting": 0.75,
                            "secondary_analysis": 0.5, "user_generated": 0.25}
            source_weight = max((tier_weights.get(s.get("trust_tier_code"), 0.5) for s in other_sources), default=0.5)
            budget.support_weight += source_weight
            if supporting_ids is not None:
                supporting_ids.append(other_claim["id"])
        elif relationship == "contradicts":
            budget.contradicts += 1
            _, created = await kb_db.add_claim_contradiction_candidate(
                claim["id"], other_claim["id"], result.get("confidence", 0.5),
                "llm_comparison", result.get("reasoning"),
            )
            if created:
                contradiction_ids.append(other_claim["id"])


async def _resolve_new_verification_claims(
    kb_db: KBDatabase, new_claim_ids: list[str], kept_ids: set[str], topics: list[dict], source_id: str,
) -> None:
    """Extracting a scraped page's top chunks can promote several new
    claims, but only the one an LLM comparison pass actually classified as
    supports/contradicts (kept_ids) is worth keeping -- the rest are
    tangential facts from the same page that were never established as
    relevant to anything, and get deleted outright. The keeper (and the
    source it came from -- attach_source_to_topic sweeps in every claim
    that source still has evidence for, which by this point is just the
    keeper) gets attached to whatever topic(s) the claim being verified
    already belongs to, so both are visible in context on the Sources page
    too, not just floating unexplained. The keeper claim is also excluded
    from auto-verification by default so it doesn't itself spawn another
    generation of checks.

    Without this, verifying claim A pulls in claims B and C from other pages
    just to compare against A; if B and C then also meet the importance
    threshold, the next sweep verifies *them* too, pulling in D, E, F, G...
    -- unbounded compounding growth with no way to ever finish, discovered
    from a real KB where a single night's cron run left 1500+ such claims,
    almost none of them ever established as relevant to anything, all
    floating with no topic and all eligible to keep the chain going."""
    for new_claim_id in new_claim_ids:
        if new_claim_id not in kept_ids:
            await kb_db.delete_claim_cascade(new_claim_id)

    kept_this_source = [c for c in new_claim_ids if c in kept_ids]
    if kept_this_source:
        for new_claim_id in kept_this_source:
            await kb_db.set_claim_verification_override(new_claim_id, "exclude")
        for topic in topics:
            await kb_db.attach_source_to_topic(topic["id"], source_id, link_reason="verification_evidence")


async def verify_claim(
    kb_db: KBDatabase, config: Config, claim_id: str, force: bool = False,
    run_search_budget: _RunSearchBudget | None = None, extraction_model: str | None = None,
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
    supporting_ids: list[str] = []
    timings: dict[str, float] = {}
    verify_start = time.monotonic()
    # Seeded from a prior inconclusive attempt's notes (see
    # UNVERIFIED_RETRY_COOLDOWN_HOURS) so a retry's first search already knows
    # to ask the LLM for something other than the literal claim text, instead
    # of wasting its first search slot repeating a query that already failed
    # to settle this claim once.
    tried_queries: list[str] = list((claim.get("verification_notes") or {}).get("web_search_queries") or [])
    verification_context = claim.get("verification_context")
    # So any claim discovered as supporting/contradicting evidence during the
    # web-fallback below can be attached to the same topic(s) as the claim
    # being verified, instead of floating with no visible context for why it
    # exists (see _quarantine_new_verification_claims).
    own_topics = await kb_db.get_topics_for_claim(claim_id)

    extraction_base_url = config.kb.verification_llm_base_url or config.kb.extraction_llm_base_url
    if extraction_model is None:
        with _timed(timings, "detect_model"):
            extraction_model = (
                config.kb.verification_llm_model or config.kb.extraction_llm_model
                or await detect_model(extraction_base_url)
            )
    llm = LLMClient(Config(llm=LLMConfig(base_url=extraction_base_url, model=extraction_model, api_key="not-needed")))

    try:
        # Phase 1: search the KB's own data first (decision 23, hybrid retrieval).
        with _timed(timings, "db_other"):
            internal_candidates = await kb_db.get_claims_independent_of(list(own_source_ids), claim_id)
        with _timed(timings, "embedding_rank"):
            ranked = await _rank_candidates_by_similarity(config, claim, internal_candidates)
        with _timed(timings, "llm_classify"):
            await _examine_candidates(
                kb_db, config, llm, claim, ranked, budget, examined_source_ids, contradiction_ids,
                supporting_ids, phase="internal",
            )

        # Phase 2: fall back to the internet only if internal coverage was thin
        # and budget remains.
        snapshot_store = SnapshotStore(config.kb_snapshot_dir)
        while not budget.should_stop() and budget.searches_remaining():
            if run_search_budget is not None and not await run_search_budget.reserve():
                break
            budget.web_searches_used += 1
            if tried_queries or verification_context:
                # Either not the first search attempted against this claim
                # (this run or a prior one) -- repeating the same query
                # verbatim just gets the same top results back -- or the
                # claim carries added verification_context, whose whole
                # point is that the literal claim text alone might not even
                # mention the angle a human wants checked. Either way, ask
                # the LLM for a query instead of using the raw claim text.
                with _timed(timings, "llm_classify"):
                    query = await _suggest_search_query(
                        llm, claim["canonical_text"], tried_queries, verification_context,
                    )
            else:
                query = claim["canonical_text"]
            tried_queries.append(query)
            try:
                with _timed(timings, "web_search"):
                    results = await web_search(query, config)
            except Exception:
                break
            if not results:
                break

            for result in results:
                if budget.should_stop() or not budget.sources_remaining("external"):
                    break
                with _timed(timings, "scrape_ingest"):
                    ingest_result = await ingest_web_page(
                        result.url, config, kb_db, snapshot_store, source_purpose="verification_evidence",
                    )
                if ingest_result.status == "failed" or ingest_result.source_id in examined_source_ids:
                    continue

                try:
                    with _timed(timings, "db_other"):
                        source = await kb_db.get_source(ingest_result.source_id)
                        version = await kb_db.get_source_version(ingest_result.version_id)
                    with _timed(timings, "chunk"):
                        chunk_result = await build_artifact_for_version(
                            kb_db, snapshot_store, source, version, config=config,
                        )
                    if chunk_result.chunk_count == 0:
                        continue

                    with _timed(timings, "db_other"):
                        artifacts = await kb_db.get_current_artifacts_for_version(version["id"])
                        page_chunks = await kb_db.list_chunks(artifacts[0]["id"])
                    with _timed(timings, "embedding_rank"):
                        ranked_chunks = await _rank_chunks_by_similarity(config, claim, page_chunks)
                    top_chunk_ids = [
                        c["id"] for c, _ in ranked_chunks[: config.kb.verification_max_chunks_per_page]
                    ]
                    if not top_chunk_ids:
                        continue
                    with _timed(timings, "llm_extract"):
                        extraction_result = await run_extraction(
                            kb_db, config, artifacts[0]["id"], chunk_ids=top_chunk_ids,
                        )
                    if extraction_result.observation_count == 0:
                        continue
                    with _timed(timings, "resolve_promote"):
                        promotion = await resolve_and_promote(kb_db, config, extraction_result.extraction_run_id)

                    with _timed(timings, "db_other"):
                        new_source_claims = await kb_db.get_claims_independent_of(
                            list(examined_source_ids), claim_id,
                        )
                    with _timed(timings, "embedding_rank"):
                        new_ranked = await _rank_candidates_by_similarity(config, claim, new_source_claims)
                    supports_before, contradicts_before = len(supporting_ids), len(contradiction_ids)
                    with _timed(timings, "llm_classify"):
                        await _examine_candidates(
                            kb_db, config, llm, claim, new_ranked, budget, examined_source_ids, contradiction_ids,
                            supporting_ids, phase="external",
                        )
                    if promotion.new_claim_ids:
                        kept_ids = set(supporting_ids[supports_before:]) | set(contradiction_ids[contradicts_before:])
                        with _timed(timings, "resolve_promote"):
                            await _resolve_new_verification_claims(
                                kb_db, promotion.new_claim_ids, kept_ids, own_topics, ingest_result.source_id,
                            )
                except Exception:
                    # One bad web-fallback source (unparseable page, extraction
                    # LLM hiccup) shouldn't abort the whole verification and
                    # lose everything found so far -- treat it like any other
                    # unusable source and move on to the next search result.
                    continue
                finally:
                    # A page scraped/chunked/extracted purely to check this
                    # claim that ended up contributing no surviving claim at
                    # all (empty page, nothing extractable, or the one claim
                    # it did produce wasn't the prover/disprover) is just as
                    # much dead weight as the discarded claims themselves --
                    # runs regardless of which exit path above was taken
                    # (early continue, exception, or falling all the way
                    # through), since all of them can leave zero evidence.
                    with _timed(timings, "db_other"):
                        if not await kb_db.source_has_claim_evidence(ingest_result.source_id):
                            await kb_db.delete_source_cascade(ingest_result.source_id)
    finally:
        await llm.close()

    timings["total"] = time.monotonic() - verify_start

    status = budget.final_status()
    notes = {
        "supports_found": budget.supports,
        "contradicts_found": budget.contradicts,
        "sources_examined": budget.sources_examined,
        "web_searches_used": budget.web_searches_used,
        "supporting_claim_ids": supporting_ids,
        "contradicting_claim_ids": contradiction_ids,
        "timings": timings,
        "web_search_queries": tried_queries,
    }
    await kb_db.record_claim_supports(claim_id, supporting_ids)
    await kb_db.update_claim_verification(claim_id, status, notes)

    return VerificationResult(
        status=status, claim_id=claim_id, supports_found=budget.supports,
        contradicts_found=budget.contradicts, sources_examined=budget.sources_examined,
        web_searches_used=budget.web_searches_used, contradiction_candidate_ids=contradiction_ids,
        supporting_claim_ids=supporting_ids, timings=timings,
    )


STALE_RUN_THRESHOLD_HOURS = 9  # cron's own timeout is 8h; a "running" row older
# than that survived past its own timeout, which only happens if the process
# that owned it was killed (crash, reboot, manual kill) without a chance to
# mark itself complete -- treat it as abandoned rather than blocking forever.

UNVERIFIED_RETRY_COOLDOWN_HOURS = 72  # A first pass that ends "unverified"
# (budget exhausted, no clear signal) is inconclusive, not a verdict -- unlike
# supported/contradicted/mixed, it should get another look eventually rather
# than being silently abandoned forever the moment verification_attempted_at
# is set. The cooldown exists so the nightly sweep doesn't re-spend a GPU
# search+extract pass on the same still-thin-coverage claim every single
# night; _suggest_search_query gives the retry an actual shot at finding
# something the first pass's query didn't.


def _unverified_retry_due(claim: dict) -> bool:
    if claim.get("status") != "unverified":
        return False
    attempted_at = claim.get("verification_attempted_at")
    if attempted_at is None:
        return False
    return datetime.now(timezone.utc) - attempted_at >= timedelta(hours=UNVERIFIED_RETRY_COOLDOWN_HOURS)


def is_claim_eligible_for_verification(claim: dict, threshold: float, force: bool = False) -> bool:
    """Single source of truth for "should this claim be auto-verified,"
    used by the nightly sweep, verify-source, and the paste-a-conversation
    flow alike -- not every extracted statement is worth checking, and a
    human can override the automatic importance-vs-threshold call either
    way via claims.verification_override (see claim_check_status).

    A claim already attempted is normally done for good -- except a claim
    that came back "unverified" (inconclusive, not a settled verdict), which
    becomes eligible again once _unverified_retry_due's cooldown has passed,
    exactly as if it were never attempted. A deprecated claim (the losing
    side of a claim merge, see kb.merge) is never eligible at all -- it's
    just a pointer to the real, canonical claim now, not a live fact worth
    spending a verification pass on."""
    if claim.get("status") == "deprecated":
        return False
    if not force and claim.get("verification_attempted_at") is not None and not _unverified_retry_due(claim):
        return False
    override = claim.get("verification_override")
    if override == "exclude":
        return False
    if override == "include":
        return True
    return (claim.get("importance_score") or 0) >= threshold


def claim_check_status(claim: dict, threshold: float) -> str:
    """Explains *why* a not-yet-verified claim will or won't be
    auto-checked, for display (e.g. the paste-a-conversation breakdown) --
    distinct from is_claim_eligible_for_verification's plain bool, since the
    UI needs to say e.g. "excluded (manually)" vs "skipped (low
    importance)" rather than just yes/no. Must mirror
    is_claim_eligible_for_verification's rules exactly, or the UI shows a
    "will check" promise the nightly sweep won't actually keep."""
    if claim.get("status") == "deprecated":
        return "deprecated"
    override = claim.get("verification_override")
    if override == "exclude":
        return "manual_exclude"
    if claim.get("verification_attempted_at") is not None and not _unverified_retry_due(claim):
        return "checked" if claim.get("status") != "unverified" else "checked_pending_retry"
    if override == "include":
        return "manual_include"
    if (claim.get("importance_score") or 0) >= threshold:
        return "auto_check"
    return "auto_skip"


async def verify_claims_concurrently(
    kb_db: KBDatabase, config: Config, claims: list[dict], *, force: bool = False,
    concurrency: int | None = None, on_start=None, on_result=None,
    run_search_budget: _RunSearchBudget | None = None,
) -> list[tuple[dict, str, "VerificationResult | Exception"]]:
    """Verifies many claims at once, up to `concurrency` in flight
    simultaneously (default: config.kb.verification_concurrency, which should
    match llama-server's --parallel -- otherwise a batch leaves a GPU slot
    idle the whole time, exactly the gap the timing/concurrency investigation
    found). Used by both the KB-wide sweep and verify-source, so "verify
    several claims at once" always exploits available concurrency rather than
    looping one at a time.

    `on_start(claim)` / `on_result(claim, status, result_or_exception)` are
    awaited (if given) right before a claim starts / right after it finishes
    -- both may run concurrently with each other across different claims, so
    callers doing shared-state bookkeeping (e.g. tracking what's currently in
    flight) need to guard it themselves (see run_verification_sweep).

    One claim's exception never aborts the batch -- it's reported as a
    'failed' status via on_result/the returned tuple, same as verify_claim's
    own per-source exception handling.
    """
    concurrency = concurrency or config.kb.verification_concurrency
    # A 100-claim sweep used to make 100 identical /models probes. Resolve
    # once before spawning tasks, then keep the actual verification calls
    # concurrent as configured.
    shared_model = None
    if claims:
        verification_base_url = config.kb.verification_llm_base_url or config.kb.extraction_llm_base_url
        shared_model = (
            config.kb.verification_llm_model or config.kb.extraction_llm_model
            or await detect_model(verification_base_url)
        )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    outcomes: list[tuple[dict, str, "VerificationResult | Exception"]] = []

    async def run_one(claim: dict) -> None:
        async with semaphore:
            if on_start:
                await on_start(claim)
            try:
                result = await verify_claim(
                    kb_db, config, claim["id"], force=force, run_search_budget=run_search_budget,
                    extraction_model=shared_model,
                )
                status = result.status
            except Exception as exc:
                result = exc
                status = "failed"
            outcomes.append((claim, status, result))
            if on_result:
                await on_result(claim, status, result)

    await asyncio.gather(*(run_one(c) for c in claims))
    return outcomes


async def run_verification_sweep(
    kb_db: KBDatabase, config: Config, *, trigger: str, threshold: float | None = None,
    limit: int | None = None, force: bool = False, on_result=None, concurrency: int | None = None,
) -> dict:
    """KB-wide verification sweep: every claim at/above the importance
    threshold gets checked against independent sources, same eligibility
    rule as verify-source but across the whole KB. Shared by the CLI's
    verify-unverified command (nightly cron + manual), and the web's
    manual-trigger route, so run tracking (verification_runs) only lives
    in one place. `on_result(claim, status, result_or_exception)` is called
    (synchronously) after each claim, in case the caller wants to report
    progress (e.g. console output) as it happens.

    Refuses to start a second sweep while one is already in progress --
    verify_claim makes real LLM calls against a single shared GPU, so
    stacking sweeps would only contend with itself.
    """
    current = await kb_db.get_current_verification_run()
    if current is not None:
        age = datetime.now(timezone.utc) - current["started_at"]
        if age < timedelta(hours=STALE_RUN_THRESHOLD_HOURS):
            raise RuntimeError(
                f"A verification run (trigger={current['trigger']}) is already in progress "
                f"since {current['started_at'].isoformat()}"
            )
        await kb_db.complete_verification_run(
            current["id"], error_message="Abandoned: run exceeded the stale-run threshold, "
            "likely orphaned by a crashed or killed process",
        )

    all_claims = await kb_db.list_claims(limit=10000)
    # Work on attached, high-importance claims first. If the run is bounded
    # by time or its shared search quota, the claims users deliberately put
    # in topics receive the most valuable coverage.
    all_claims.sort(
        key=lambda claim: (bool(claim.get("topics")), claim.get("importance_score") or 0), reverse=True,
    )
    eff_threshold = threshold if threshold is not None else config.kb.verification_importance_threshold
    eligible = [c for c in all_claims if is_claim_eligible_for_verification(c, eff_threshold, force=force)]
    if limit is not None:
        eligible = eligible[:limit]

    counts = {"supported": 0, "contradicted": 0, "mixed": 0, "unverified": 0, "skipped": 0, "failed": 0}
    run_search_budget = _RunSearchBudget(config.kb.verification_run_max_web_searches)
    run = await kb_db.create_verification_run(trigger, claims_total=len(eligible))
    in_flight: dict[str, str] = {}
    in_flight_lock = asyncio.Lock()

    async def handle_start(claim: dict) -> None:
        async with in_flight_lock:
            in_flight[claim["id"]] = claim["canonical_text"]
            await kb_db.set_verification_run_in_flight(run["id"], list(in_flight.values()))

    async def handle_result(claim: dict, status: str, result) -> None:
        counts[status] += 1
        async with in_flight_lock:
            in_flight.pop(claim["id"], None)
            await kb_db.record_verification_run_result(run["id"], f"{status}_count")
            await kb_db.set_verification_run_in_flight(run["id"], list(in_flight.values()))
        if on_result:
            on_result(claim, status, result)

    try:
        await verify_claims_concurrently(
            kb_db, config, eligible, force=force, concurrency=concurrency,
            on_start=handle_start, on_result=handle_result, run_search_budget=run_search_budget,
        )
        await kb_db.complete_verification_run(run["id"])
    except Exception as exc:
        await kb_db.complete_verification_run(run["id"], error_message=str(exc))
        raise

    return {
        "run_id": run["id"], "eligible_count": len(eligible), "threshold": eff_threshold,
        "counts": counts, "web_searches_used": run_search_budget.used,
        "web_search_limit": run_search_budget.limit,
    }
