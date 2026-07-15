"""Registry-neutral lifecycle helpers for a local llama-server process."""

import asyncio
import shutil
from pathlib import Path

import httpx


def build_launch_command(model_path: str, port: int, args: dict | None = None) -> list[str]:
    args = args or {}
    server_bin = args.get("llama_server_bin", "llama-server")
    if not shutil.which(server_bin):
        local_build = Path.home() / "llama" / "llama.cpp" / "build" / "bin" / "llama-server"
        if local_build.exists():
            server_bin = str(local_build)
    # Three slots are the machine-wide policy, shared by the boot primary,
    # registered evaluation profiles, and temporary model swaps. Do not let a
    # stale profile silently lower it for a different model.
    cmd = [server_bin, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", str(args.get("gpu_layers", 99)), "-c", str(args.get("context", 32768)),
           "-b", str(args.get("batch", 4096)), "-ub", str(args.get("ubatch", 512)), "--parallel", "3"]
    if args.get("flash_attn", True): cmd += ["-fa", "on"]
    # --jinja renders prompts with the model's own embedded chat template and
    # is what unlocks native OpenAI-style tool calling (without it,
    # llama-server 400s the `tools` param and LLMClient silently downgrades
    # the research agent to its no-native-tools path). Default on since
    # 2026-07-15 -- eval rounds 1-4 / cross-verify baselines predate this,
    # so their numbers aren't strictly comparable to runs made with it. Opt
    # out per model ({"jinja": false}) for a GGUF whose embedded template is
    # broken or uses Jinja features llama.cpp's minja engine doesn't
    # support; {"chat_template_file": path} overrides the template while
    # keeping --jinja on.
    if args.get("jinja", True): cmd += ["--jinja"]
    if args.get("chat_template_file"): cmd += ["--chat-template-file", args["chat_template_file"]]
    for key, flag in (("tensor_split", "-ts"), ("devices", "-dev"), ("split_mode", "-sm")):
        if args.get(key): cmd += [flag, args[key]]
    return cmd


async def is_healthy(port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code == 200 and response.json().get("status") == "ok"
    except httpx.HTTPError:
        return False


async def wait_ready(port: int, attempts: int = 30, interval_seconds: float = 2) -> bool:
    for _ in range(attempts):
        if await is_healthy(port): return True
        await asyncio.sleep(interval_seconds)
    return False


async def start_server(model_path: str, port: int, args: dict, log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log_file:
        await asyncio.create_subprocess_exec(*build_launch_command(model_path, port, args), stdout=log_file,
            stderr=asyncio.subprocess.STDOUT, stdin=asyncio.subprocess.DEVNULL, start_new_session=True)
    return await wait_ready(port)
