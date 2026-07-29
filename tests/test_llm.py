import asyncio

import httpx

from deep_research.config import Config
from deep_research.llm import LLMClient


def _fake_response(content: str = "ok") -> httpx.Response:
    request = httpx.Request("POST", "http://fake/chat/completions")
    return httpx.Response(
        200,
        request=request,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


async def test_chat_respects_max_concurrent_requests_per_endpoint(monkeypatch):
    """The exact live regression this guards against: an outer layer
    verifying N claims at once, each of which may itself resolve M new
    claims concurrently, must never be able to multiply past the LLM
    server's real concurrent-request capacity (llama.cpp's --parallel N) --
    measured live at 15-44 minutes per claim once a second, independently-
    sized layer of concurrency was added without a shared ceiling."""
    config = Config()
    config.llm.base_url = "http://fake-endpoint-a/v1"
    config.llm.max_concurrent_requests = 2

    in_flight = 0
    max_in_flight = 0

    async def fake_post(url, json=None):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return _fake_response()

    clients = [LLMClient(config) for _ in range(6)]
    for client in clients:
        monkeypatch.setattr(client._client, "post", fake_post)

    await asyncio.gather(*(client.chat([{"role": "user", "content": "hi"}]) for client in clients))

    assert max_in_flight == 2
    for client in clients:
        await client.close()


async def test_chat_endpoints_have_independent_semaphores(monkeypatch):
    """A different LLM endpoint (a different backend, or extraction vs.
    verification pointing at different servers) must not share the other
    endpoint's concurrency budget -- throttling one server's calls should
    never block requests to an unrelated one."""
    config_a = Config()
    config_a.llm.base_url = "http://fake-endpoint-b/v1"
    config_a.llm.max_concurrent_requests = 1
    config_b = Config()
    config_b.llm.base_url = "http://fake-endpoint-c/v1"
    config_b.llm.max_concurrent_requests = 1

    in_flight_by_endpoint = {"b": 0, "c": 0}
    max_in_flight_by_endpoint = {"b": 0, "c": 0}

    def make_fake_post(key):
        async def fake_post(url, json=None):
            in_flight_by_endpoint[key] += 1
            max_in_flight_by_endpoint[key] = max(max_in_flight_by_endpoint[key], in_flight_by_endpoint[key])
            await asyncio.sleep(0.05)
            in_flight_by_endpoint[key] -= 1
            return _fake_response()
        return fake_post

    client_a = LLMClient(config_a)
    client_b = LLMClient(config_b)
    monkeypatch.setattr(client_a._client, "post", make_fake_post("b"))
    monkeypatch.setattr(client_b._client, "post", make_fake_post("c"))

    # Two concurrent calls to each of two different, single-slot endpoints --
    # if they shared one semaphore, only one of the four would run at a time.
    await asyncio.gather(
        client_a.chat([{"role": "user", "content": "1"}]),
        client_a.chat([{"role": "user", "content": "2"}]),
        client_b.chat([{"role": "user", "content": "3"}]),
        client_b.chat([{"role": "user", "content": "4"}]),
    )

    assert max_in_flight_by_endpoint["b"] == 1
    assert max_in_flight_by_endpoint["c"] == 1
    await client_a.close()
    await client_b.close()
