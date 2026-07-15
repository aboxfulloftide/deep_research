"""Safe, persistent primary llama.cpp profile switching for interactive chat."""

from pathlib import Path

from deep_research.config import Config
from deep_research.evals import registry
from deep_research.evals.server import start_server, stop_server
from deep_research.kb.gpu_idle import gpu_is_idle
from deep_research.kb.jobs import GPU_WORKER_ADVISORY_LOCK
from deep_research.kb.extraction import detect_model
from deep_research.tools.llama_server import is_healthy


class ModelSwitchUnavailable(RuntimeError):
    """The primary model cannot safely be changed at this moment."""


def _same_model(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return Path(left).expanduser() == Path(right).expanduser()


async def _primary_profile(config: Config, model_path: str) -> dict | None:
    for profile in await registry.list_models(config):
        if _same_model(profile["model_path"], model_path):
            profile = dict(profile)
            profile["port"] = 8080
            return profile
    return None


async def _normal_work_is_pending(kb_db) -> bool:
    for status in ("queued", "running"):
        jobs = await kb_db.list_processing_jobs(status=status, limit=1000)
        if any(job["job_type"] != "model_experiment" for job in jobs):
            return True
    return False


async def switch_primary_profile(kb_db, config: Config, profile_slug: str) -> str:
    """Load a registered profile at port 8080 when all normal work is idle.

    The worker advisory lock makes the idle check and server swap atomic with
    respect to the KB worker. Unlike a model experiment, this deliberately
    leaves the selected profile running for subsequent interactive chat.
    """
    target = await registry.get_model(config, profile_slug)
    if target is None:
        raise ValueError(f"Unknown model profile {profile_slug!r}")
    target = dict(target)
    target["port"] = 8080
    target["systemd_detach"] = True

    async with kb_db.pool.acquire() as lock_conn:
        acquired = await lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", GPU_WORKER_ADVISORY_LOCK)
        if not acquired:
            raise ModelSwitchUnavailable("Research processing is active; wait for it to finish before switching models")
        try:
            if await _normal_work_is_pending(kb_db):
                raise ModelSwitchUnavailable("Processing work is queued or running; wait until the system is idle")
            if not await gpu_is_idle():
                raise ModelSwitchUnavailable("The GPU is busy; wait until the active generation finishes")

            current = await detect_model(config.llm.llama_cpp_base_url)
            if _same_model(current, target["model_path"]):
                return current

            primary = await _primary_profile(config, current)
            if primary is None:
                raise ModelSwitchUnavailable("The currently loaded model is not registered, so it cannot be safely restored")
            primary["systemd_detach"] = True
            if not await stop_server(primary):
                raise RuntimeError("Could not stop the currently loaded llama.cpp model")

            ready, log_path = await start_server(target)
            if not ready:
                restored, restore_log = await start_server(primary)
                if not restored:
                    raise RuntimeError(
                        f"The selected model failed to start ({log_path}) and the prior model could not be restored ({restore_log})",
                    )
                raise RuntimeError(f"The selected model failed to start; restored the prior model. See {log_path}")
            return await detect_model(config.llm.llama_cpp_base_url)
        finally:
            await lock_conn.execute("SELECT pg_advisory_unlock($1)", GPU_WORKER_ADVISORY_LOCK)
