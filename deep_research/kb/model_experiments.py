"""Safe, queued experiments for comparing local llama.cpp configurations."""

import json
import time

from deep_research.config import Config, LLMConfig
from deep_research.evals import registry
from deep_research.evals.server import start_server, stop_server
from deep_research.kb.extraction import detect_context_size, detect_model
from deep_research.llm import LLMClient
from deep_research.tools.extra_research import collect_sources, source_context
from deep_research.tools.llama_server import is_healthy


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


async def run_model_experiment(kb_db, config: Config, job: dict) -> dict:
    """Compare one model/config without changing the primary llama server.

    Alternate profiles run on their registry evaluation port and are stopped
    afterward. The 8080 interactive/KB server is never stopped or reloaded.
    """
    payload = job.get("payload") or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("model experiment needs a prompt")
    profile_slug = payload.get("profile_slug") or "current"
    reasoning = bool(payload.get("reasoning", True))
    requested_context = payload.get("context_size")
    started_profile = None

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
            server_args["context"] = int(requested_context)
        profile["server_args_json"] = json.dumps(server_args)
        # Never replace the primary server. This temporary server only starts
        # after the worker's normal-work and GPU-idle gates have passed.
        if not await is_healthy(profile["port"]):
            ready, log_path = await start_server(profile)
            if not ready:
                raise RuntimeError(f"Experiment server did not become ready; see {log_path}")
            started_profile = profile
        try:
            base_url = f"http://127.0.0.1:{profile['port']}/v1"
            model = await detect_model(base_url)
            context_size = await detect_context_size(base_url)
        except Exception:
            if started_profile is not None:
                await stop_server(started_profile)
            raise
        display_name = profile["display_name"]

    try:
        await kb_db.update_processing_job_progress(
            job["id"], "gather_sources",
            {"profile": profile_slug, "model": model, "context_size": context_size, "reasoning": reasoning},
            lease_seconds=900,
        )
        sources = await collect_sources([prompt], config, 1, set())
        if not sources:
            raise RuntimeError("Could not collect source context for model experiment")

        await kb_db.update_processing_job_progress(
            job["id"], "evaluate", {"source_count": len(sources), "context_size": context_size}, lease_seconds=900,
        )
        system = (
            "You are evaluating a local research assistant. Give a precise, evidence-grounded answer. "
            "State uncertainty and cite supplied sources as Markdown links."
        )
        if not reasoning:
            system = "/no_think\n" + system
        llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))
        started_at = time.monotonic()
        try:
            response = await llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {prompt}\n\nEvidence:\n{source_context(sources)}"},
            ])
        finally:
            await llm.close()
        answer = response["choices"][0]["message"].get("content", "No answer produced.")
        return {
            "profile": profile_slug,
            "display_name": display_name,
            "model": model,
            "context_size": context_size,
            "reasoning": reasoning,
            "source_count": len(sources),
            "elapsed_seconds": round(time.monotonic() - started_at, 1),
            "answer": answer,
        }
    finally:
        if started_profile is not None:
            await stop_server(started_profile)
