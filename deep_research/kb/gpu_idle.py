"""Conservative, workload-agnostic GPU idle detection for speculative jobs."""

import asyncio

import httpx


OWN_INFERENCE_PROCESS_NAMES = ("llama-server", "ollama")
# Desktop applications can appear in nvidia-smi's compute-app list on hybrid
# Wayland/NVIDIA systems despite not running CUDA inference.
DESKTOP_PROCESS_NAMES = ("gnome-shell", "ptyxis", "xorg", "wayland")


async def _llama_slots_are_idle() -> bool:
    """Distinguish normal desktop GPU rendering from actual model inference."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://127.0.0.1:8080/slots")
            response.raise_for_status()
            slots = response.json()
        return all(not slot.get("is_processing", False) for slot in slots)
    except Exception:
        return False


async def gpu_is_idle(samples: int = 3, interval_seconds: float = 2.0) -> bool:
    """Return true only after sustained absence of compute/model work.

    Any unknown compute process counts as busy. A locally idle llama-server or
    Ollama process is allowed so this project can begin speculative work with
    its own inference server already loaded. If nvidia-smi is unavailable,
    fail closed: speculative work must not compete blindly.
    """
    for index in range(samples):
        try:
            apps = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            app_stdout, _ = await apps.communicate()
            if apps.returncode != 0:
                return False
            names = [name.strip().lower() for name in app_stdout.decode().splitlines() if name.strip()]
            if any(
                not any(own in name for own in OWN_INFERENCE_PROCESS_NAMES)
                and not any(desktop in name for desktop in DESKTOP_PROCESS_NAMES)
                for name in names
            ):
                return False
        except (FileNotFoundError, ValueError):
            return False
        if not await _llama_slots_are_idle():
            return False
        if index + 1 < samples:
            await asyncio.sleep(interval_seconds)
    return True
