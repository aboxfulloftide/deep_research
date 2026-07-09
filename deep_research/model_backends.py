"""Lets the interactive research agent (CLI + web) point at either Ollama or
llama.cpp instead of being hardcoded to one backend. LLMClient itself doesn't
know about "backends" at all -- it just reads config.llm.base_url/model/
api_key, so switching means resolving a named preset into those three fields
before constructing an LLMClient, and listing models in whichever shape the
active backend's API returns them (Ollama's native /api/tags vs. llama.cpp's
OpenAI-compatible /models).
"""

import httpx

from deep_research.config import Config

BACKENDS = ("ollama", "llama_cpp")


def resolve_backend(config: Config, backend: str) -> tuple[str, str]:
    """Returns (base_url, api_key) for the named backend's configured preset."""
    if backend == "ollama":
        return config.llm.ollama_base_url, config.llm.ollama_api_key
    if backend == "llama_cpp":
        return config.llm.llama_cpp_base_url, config.llm.llama_cpp_api_key
    raise ValueError(f"Unknown backend {backend!r}, expected one of {BACKENDS}")


def apply_backend(config: Config, backend: str) -> Config:
    """Returns a config with llm.base_url/api_key/backend set from the named
    backend's preset -- model is left untouched, since it's chosen separately
    (explicit --model/request field, or a selection prompt/dropdown)."""
    base_url, api_key = resolve_backend(config, backend)
    config.llm.backend = backend
    config.llm.base_url = base_url
    config.llm.api_key = api_key
    return config


async def list_models(backend: str, base_url: str) -> list[str]:
    """Fetches available model names from whichever backend is active."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        if backend == "ollama":
            ollama_url = base_url.replace("/v1", "")
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        elif backend == "llama_cpp":
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        raise ValueError(f"Unknown backend {backend!r}, expected one of {BACKENDS}")
