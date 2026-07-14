"""llama-server lifecycle management for a registered eval model -- a Python
port of the bash start/wait_ready/stop_server block hand-copied into every
verify_round*.sh script tonight, so the next round doesn't need a fresh one.

Only one model's server is expected to run on a given port at a time (the
same assumption tonight's manual rounds made -- start one, use it, stop it,
start the next).
"""

import asyncio
import json
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
    """Matches tonight's `pkill -f "llama-server.*<model path>"` + poll-until-
    gone approach -- shells out rather than adding a psutil dependency for
    process matching that already works fine as a one-liner."""
    model_path = model["model_path"]
    proc = await asyncio.create_subprocess_exec(
        "pkill", "-f", f"llama-server.*{model_path}",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    for _ in range(STOP_POLL_MAX_ATTEMPTS):
        check = await asyncio.create_subprocess_exec(
            "pgrep", "-f", f"llama-server.*{model_path}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await check.wait()
        if rc != 0:  # pgrep found nothing -- process is gone
            return True
        await asyncio.sleep(STOP_POLL_INTERVAL_SECONDS)
    return False
