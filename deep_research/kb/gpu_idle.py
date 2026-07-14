"""Conservative, workload-agnostic GPU idle detection for speculative jobs."""

import asyncio


OWN_INFERENCE_PROCESS_NAMES = ("llama-server", "ollama")


async def gpu_is_idle(samples: int = 3, interval_seconds: float = 2.0, max_utilization: int = 10) -> bool:
    """Return true only after sustained low GPU use.

    Any unknown compute process counts as busy. A locally idle llama-server or
    Ollama process is allowed so this project can begin speculative work with
    its own inference server already loaded. If nvidia-smi is unavailable,
    fail closed: speculative work must not compete blindly.
    """
    for index in range(samples):
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0 or any(int(line.strip()) > max_utilization for line in stdout.decode().splitlines() if line.strip()):
                return False
            apps = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            app_stdout, _ = await apps.communicate()
            if apps.returncode != 0:
                return False
            names = [name.strip().lower() for name in app_stdout.decode().splitlines() if name.strip()]
            if any(not any(own in name for own in OWN_INFERENCE_PROCESS_NAMES) for name in names):
                return False
        except (FileNotFoundError, ValueError):
            return False
        if index + 1 < samples:
            await asyncio.sleep(interval_seconds)
    return True
