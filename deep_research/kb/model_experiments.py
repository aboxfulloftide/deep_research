"""Safe, queued experiments for comparing local llama.cpp configurations."""

import asyncio
import json
import time
import uuid
from pathlib import Path

from deep_research.config import Config, LLMConfig
from deep_research.evals import registry
from deep_research.evals.server import start_server, stop_server
from deep_research.kb.extraction import detect_context_size, detect_model
from deep_research.llm import LLMClient
from deep_research.tools.extra_research import (
    analysis_context,
    analyze_sources_separately,
    build_claim_ledger,
    claim_ledger_context,
    collect_sources,
    derive_gap_closing_query,
    derive_follow_up_queries,
    derive_starting_queries,
    has_authoritative_source,
    source_context,
)
from deep_research.tools.llama_server import is_healthy

PARALLEL_REQUEST_SLOTS = 3


def _serialize_source(source) -> dict:
    """Keep a research source JSON-safe when it is attached to queued jobs."""
    return {
        "title": source.title, "url": source.url, "content": source.content,
        "full_content": source.full_content, "level": source.level, "query": source.query,
        "source_kind": source.source_kind, "quality_score": source.quality_score,
    }


def _deserialize_sources(rows: list[dict]):
    from deep_research.tools.extra_research import ResearchSource

    return [ResearchSource(
        title=str(row["title"]), url=str(row["url"]), content=str(row["content"]),
        full_content=str(row.get("full_content") or row["content"]), level=int(row["level"]),
        query=str(row["query"]), source_kind=str(row.get("source_kind") or "secondary"),
        quality_score=int(row.get("quality_score") or 0),
    ) for row in rows]


async def build_frozen_evidence_bundle(kb_db, config: Config, job: dict) -> dict:
    """Gather one auditable source set to be reused by every comparison run.

    Search/query planning is deliberately performed once, using the currently
    loaded server.  Child experiments then only evaluate the supplied sources;
    their results are comparable even if the web changes while they wait.
    """
    payload = job.get("payload") or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("model comparison needs a prompt")
    base_url = config.llm.llama_cpp_base_url
    model = await detect_model(base_url)
    llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))
    try:
        await kb_db.update_processing_job_progress(
            job["id"], "gather_sources", {"bundle_model": model}, lease_seconds=900,
        )
        seen_urls: set[str] = set()
        sources = []
        queries = [prompt, *await derive_starting_queries(llm, prompt)]
        for level in range(1, 5):
            sources.extend(await collect_sources(
                queries, config, level, seen_urls, sources_per_query=1 if level == 4 else None,
            ))
            if level < 3:
                queries = await derive_follow_up_queries(llm, prompt, sources, level)
            elif level == 3:
                queries = await derive_gap_closing_query(llm, prompt, sources)
        if not sources or not has_authoritative_source(sources):
            raise RuntimeError("Could not collect an authoritative model card or paper for the comparison bundle")
    finally:
        await llm.close()

    bundle_id = str(uuid.uuid4())
    return {
        "id": bundle_id,
        "created_with_model": model,
        "source_count": len(sources),
        "source_urls": [source.url for source in sources],
        "sources": [_serialize_source(source) for source in sources],
    }


async def _start_with_retry(profile: dict, attempts: int = 3) -> tuple[bool, object]:
    """Transient systemd units can still be retiring when a swap restores a model."""
    log_path = None
    for attempt in range(attempts):
        ready, log_path = await start_server(profile)
        if ready:
            return True, log_path
        if attempt + 1 < attempts:
            await asyncio.sleep(2 * (attempt + 1))
    return False, log_path


async def available_profiles(config: Config) -> dict:
    """Return the current server plus registered alternate llama profiles."""
    current_model = None
    current_context = None
    try:
        current_model = await detect_model(config.llm.llama_cpp_base_url)
        current_context = await detect_context_size(config.llm.llama_cpp_base_url)
    except Exception:
        pass
    profiles = await registry.list_models(config)
    return {
        "current": {
            "slug": "current", "display_name": current_model or "Current llama.cpp model",
            "context_size": current_context,
            "active": True,
        },
        "profiles": [
            {
                "slug": profile["slug"],
                "display_name": profile["display_name"],
                "context_size": json.loads(profile["server_args_json"]).get("context"),
            }
            for profile in profiles
        ],
    }


async def _active_profile(config: Config, active_model: str) -> dict | None:
    """Find the registry profile backing the currently loaded primary model.

    A registered profile carries the exact launch arguments needed to restore
    the manually managed 8080 server after an alternate-model experiment.
    """
    active_path = Path(active_model).expanduser()
    try:
        active_path = active_path.resolve()
    except OSError:
        pass
    for candidate in await registry.list_models(config):
        candidate_path = Path(candidate["model_path"]).expanduser()
        try:
            candidate_path = candidate_path.resolve()
        except OSError:
            pass
        if candidate_path == active_path:
            primary = dict(candidate)
            primary["port"] = 8080
            return primary
    return None


async def run_model_experiment(kb_db, config: Config, job: dict) -> dict:
    """Compare one model/config while preserving the primary llama server.

    Alternate profiles run on their registry evaluation port and are stopped
    afterward. If an alternate cannot coexist in VRAM, the idle primary is
    temporarily swapped out and restored before normal queue work can resume.
    """
    payload = job.get("payload") or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("model experiment needs a prompt")
    profile_slug = payload.get("profile_slug") or "current"
    reasoning = bool(payload.get("reasoning", True))
    requested_context = payload.get("context_size")
    started_profile = None
    primary_to_restore = None

    if profile_slug == "current":
        base_url = config.llm.llama_cpp_base_url
        model = await detect_model(base_url)
        context_size = await detect_context_size(base_url)
        display_name = model
    else:
        profile = await registry.get_model(config, profile_slug)
        if not profile:
            raise ValueError(f"Unknown registered model profile {profile_slug!r}")
        profile = dict(profile)
        server_args = json.loads(profile["server_args_json"])
        if requested_context:
            # llama.cpp splits -c across --parallel slots.  Experiment/UI
            # context is a per-request value, so preserve it by allocating
            # the corresponding total server context.  Without this, a
            # nominal 11,008-token run with parallel=3 only accepts ~3,669
            # tokens and rejects a normal evidence bundle with HTTP 400.
            server_args["context"] = int(requested_context) * PARALLEL_REQUEST_SLOTS
        profile["server_args_json"] = json.dumps(server_args)
        # The web worker can be restarted while an experiment runs. Put the
        # temporary server in its own user unit so a web-service restart does
        # not kill it mid-swap.
        profile["systemd_detach"] = True
        primary_model = await detect_model(config.llm.llama_cpp_base_url)
        # Two copies of the same compact model can coexist on this machine,
        # which lets us compare context/reasoning settings without touching
        # the interactive server. A larger/different profile cannot coexist
        # in VRAM, so the worker safely swaps only after its queue/GPU gates
        # have found normal work idle, then restores the primary in `finally`.
        if Path(profile["model_path"]).expanduser() != Path(primary_model).expanduser():
            primary_to_restore = await _active_profile(config, primary_model)
            if primary_to_restore is None:
                raise RuntimeError(
                    "Cannot safely swap models because the active 8080 model is not registered for restoration",
                )
            primary_to_restore["systemd_detach"] = True
            await kb_db.update_processing_job_progress(
                job["id"], "swapping_model",
                {"from_model": primary_model, "to_profile": profile_slug}, lease_seconds=900,
            )
            if not await stop_server(primary_to_restore):
                raise RuntimeError("Could not stop the idle primary llama.cpp server for the experiment")
        if not await is_healthy(profile["port"]):
            ready, log_path = await start_server(profile)
            if not ready:
                if primary_to_restore is not None:
                    await _start_with_retry(primary_to_restore)
                raise RuntimeError(f"Experiment server did not become ready; see {log_path}")
            started_profile = profile
        try:
            base_url = f"http://127.0.0.1:{profile['port']}/v1"
            model = await detect_model(base_url)
            detected_context = await detect_context_size(base_url)
            context_size = int(requested_context) if requested_context else detected_context
        except Exception:
            if started_profile is not None:
                await stop_server(started_profile)
            if primary_to_restore is not None:
                ready, log_path = await _start_with_retry(primary_to_restore)
                if not ready:
                    raise RuntimeError(f"Could not restore the primary llama.cpp server; see {log_path}")
            raise
        display_name = profile["display_name"]

    try:
        llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))
        started_at = time.monotonic()
        try:
            frozen_bundle = payload.get("evidence_bundle")
            collection_attempts = []
            if frozen_bundle:
                sources = _deserialize_sources(frozen_bundle.get("sources") or [])
                await kb_db.update_processing_job_progress(
                    job["id"], "load_frozen_evidence",
                    {"bundle_id": frozen_bundle.get("id"), "source_count": len(sources), "profile": profile_slug},
                    lease_seconds=900,
                )
            else:
                await kb_db.update_processing_job_progress(
                    job["id"], "gather_sources",
                    {"profile": profile_slug, "model": model, "context_size": context_size, "reasoning": reasoning},
                    lease_seconds=900,
                )
                seen_urls: set[str] = set()
                sources = []
                queries = [prompt, *await derive_starting_queries(llm, prompt)]
                for level in range(1, 5):
                    level_sources = await collect_sources(
                        queries, config, level, seen_urls,
                        sources_per_query=1 if level == 4 else None,
                    )
                    sources.extend(level_sources)
                    collection_attempts.append({
                        "level": level, "queries": list(queries), "source_count": len(level_sources),
                        "source_urls": [source.url for source in level_sources],
                    })
                    if level < 3:
                        queries = await derive_follow_up_queries(llm, prompt, sources, level)
                    elif level == 3:
                        queries = await derive_gap_closing_query(llm, prompt, sources)

            if payload.get("collection_only"):
                # Keep collection benchmarking separate from extraction and
                # synthesis.  The saved raw bundle is the artifact to review
                # and later freeze for a fair analysis comparison.
                return {
                    "mode": "source_collection",
                    "profile": profile_slug,
                    "display_name": display_name,
                    "model": model,
                    "context_size": context_size,
                    "reasoning": reasoning,
                    "source_count": len(sources),
                    "source_urls": [source.url for source in sources],
                    "sources": [_serialize_source(source) for source in sources],
                    "collection_attempts": collection_attempts,
                    "has_authoritative_source": has_authoritative_source(sources),
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                    "answer": "Source collection complete; review this bundle before analysis.",
                }

            if not sources or not has_authoritative_source(sources):
                raise RuntimeError("Could not collect an authoritative model card or paper for the experiment")

            await kb_db.update_processing_job_progress(
                job["id"], "evaluate", {"source_count": len(sources), "context_size": context_size}, lease_seconds=900,
            )
            briefs = analysis_context(await analyze_sources_separately(llm, prompt, sources))
            claims = await build_claim_ledger(llm, prompt, sources)
            if not claims:
                raise RuntimeError("Could not extract source-quoted claims for the experiment")
            ledger = claim_ledger_context(claims)
            system = (
                "You are evaluating a local research assistant. Write a concise decision memo using ONLY the supplied "
                "claim ledger. Do not add facts from general knowledge or from the source analyses. Every factual or "
                "numerical sentence must contain an exact Markdown link from the ledger. Separate official specifications, "
                "estimates, and unknowns. Call a finding official only when its ledger tier is primary or paper; otherwise "
                "label it technical-reference evidence. Never use [citation: N] or a source not in the ledger."
            )
            if not reasoning:
                system = "/no_think\n" + system
            response = await llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {prompt}\n\nClaim ledger:\n{ledger}\n\nSource analyses (context only):\n{briefs}"},
            ])
            draft = response["choices"][0]["message"].get("content", "No answer produced.")
            fact_check_system = (
                "You are a strict final fact checker. Check the draft against the original question and the supplied "
                "claim ledger. Remove unsupported, overstated, or uncited claims instead of guessing. Return only the "
                "corrected evidence-grounded answer with Markdown source links from the ledger. Never output [citation: N] "
                "or label secondary/technical-reference evidence as official."
            )
            if not reasoning:
                fact_check_system = "/no_think\n" + fact_check_system
            response = await llm.chat([
                {"role": "system", "content": fact_check_system},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {prompt}\n\nDraft answer:\n{draft}\n\n"
                        f"Claim ledger:\n{ledger}\n\nSource excerpts:\n{source_context(sources, per_source_chars=900)}"
                    ),
                },
            ])
            answer = response["choices"][0]["message"].get("content", "").strip() or draft
        finally:
            await llm.close()
        return {
            "profile": profile_slug,
            "display_name": display_name,
            "model": model,
            "context_size": context_size,
            "reasoning": reasoning,
            "evidence_bundle_id": frozen_bundle.get("id") if frozen_bundle else None,
            "source_count": len(sources),
            "elapsed_seconds": round(time.monotonic() - started_at, 1),
            "answer": answer,
        }
    finally:
        if started_profile is not None:
            await stop_server(started_profile)
        if primary_to_restore is not None:
            await kb_db.update_processing_job_progress(
                job["id"], "restoring_model", {"profile": primary_to_restore["slug"]}, lease_seconds=900,
            )
            ready, log_path = await _start_with_retry(primary_to_restore)
            if not ready:
                raise RuntimeError(f"Could not restore the primary llama.cpp server; see {log_path}")
