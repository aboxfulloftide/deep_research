"""llama-server lifecycle management for a registered eval model -- a Python
port of the bash start/wait_ready/stop_server block hand-copied into every
verify_round*.sh script tonight, so the next round doesn't need a fresh one.

Only one model's server is expected to run on a given port at a time (the
same assumption tonight's manual rounds made -- start one, use it, stop it,
start the next).
"""

import asyncio
import json
import re
from pathlib import Path

import httpx
from deep_research.tools.llama_server import build_launch_command as _build_launch_command, is_healthy, wait_ready

HEALTH_POLL_INTERVAL_SECONDS = 2
HEALTH_POLL_MAX_ATTEMPTS = 30
STOP_POLL_INTERVAL_SECONDS = 2
STOP_POLL_MAX_ATTEMPTS = 15


def logs_dir() -> Path:
    path = Path.cwd() / "evals" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_launch_command(model: dict) -> list[str]:
    return _build_launch_command(model["model_path"], model["port"], json.loads(model["server_args_json"]))


async def start_server(model: dict) -> tuple[bool, Path]:
    """Launches the model's llama-server detached (survives after this
    process exits, same as tonight's `nohup ... & disown`) and waits for
    /health. Returns (ready, log_path) -- log_path is where stdout/stderr
    landed, useful to tail if ready is False."""
    log_path = logs_dir() / f"{model['slug']}-server.log"
    cmd = build_launch_command(model)

    if model.get("systemd_detach"):
        # A boot launcher is itself a systemd unit. Starting a child directly
        # from it would leave that child in the launcher's cgroup, which is
        # cleaned up as soon as a Type=oneshot launcher exits. Put the actual
        # server in a transient sibling unit instead. It intentionally has no
        # restart policy: model-experiment code stops it by model path while
        # swapping profiles and restores the primary itself afterward.
        unit_name = model.get("systemd_unit", f"deep-research-llama-runtime-{model['port']}")
        proc = await asyncio.create_subprocess_exec(
            "systemd-run", "--user", "--quiet", "--collect", "--no-block",
            f"--unit={unit_name}", *cmd,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        if await proc.wait() != 0:
            return False, log_path
    else:
        with open(log_path, "ab") as log_file:
            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_file, stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )

    ready = await wait_ready(model["port"])
    return ready, log_path


async def stop_server(model: dict) -> bool:
    """Stop exactly one registered llama-server process.

    Profiles can use the same GGUF on an evaluation port while the primary
    server keeps that GGUF loaded on 8080. Matching only the model path would
    kill both; include the port so experiment cleanup cannot stop the primary.
    """
    model_path = model["model_path"]
    port = model["port"]
    pattern = f"llama-server.*-m {re.escape(model_path)}.*--port {port}"
    proc = await asyncio.create_subprocess_exec(
        "pkill", "-f", pattern,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    for _ in range(STOP_POLL_MAX_ATTEMPTS):
        check = await asyncio.create_subprocess_exec(
            "pgrep", "-f", pattern,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await check.wait()
        if rc != 0:  # pgrep found nothing -- process is gone
            return True
        await asyncio.sleep(STOP_POLL_INTERVAL_SECONDS)
    return False
