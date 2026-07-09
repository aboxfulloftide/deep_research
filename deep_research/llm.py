import json
import re
from typing import AsyncIterator

import httpx

from deep_research.config import Config
from deep_research.retry import with_retries

# Pattern to strip reasoning/thinking tags from models like qwen3, deepseek
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
# Also catch unclosed think tags (model sometimes forgets to close)
_THINK_UNCLOSED_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


def _strip_thinking(text: str | None) -> str | None:
    """Remove <think>...</think> blocks from reasoning models."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


class LLMClient:
    def __init__(self, config: Config):
        self.base_url = config.llm.base_url.rstrip("/")
        self.model = config.llm.model
        self.api_key = config.llm.api_key
        self.supports_tools: bool | None = None  # Auto-detected on first call
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send a chat completion request. Returns the full response dict."""
        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Try with tools if supported (or unknown)
        use_tools = tools and self.supports_tools is not False
        if use_tools:
            payload["tools"] = tools

        async def _post() -> httpx.Response:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp

        try:
            resp = await with_retries(_post)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and use_tools:
                # Model doesn't support tool calling — retry without tools
                self.supports_tools = False
                payload.pop("tools", None)
                resp = await with_retries(_post)
            else:
                raise

        if use_tools and self.supports_tools is None:
            self.supports_tools = True

        data = resp.json()

        # Strip thinking tags from response content
        if data.get("choices"):
            msg = data["choices"][0].get("message", {})
            msg["content"] = _strip_thinking(msg.get("content"))

        return data

    async def chat_stream(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content chunks."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    async def close(self):
        await self._client.aclose()
