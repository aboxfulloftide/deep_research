import pytest

from deep_research.config import Config
from deep_research.kb import model_experiments as experiments


class _FakeDB:
    def __init__(self):
        self.progress = []

    async def update_processing_job_progress(self, job_id, stage, progress=None, **kwargs):
        self.progress.append((stage, progress))


class _FakeLLM:
    def __init__(self, config):
        self.config = config

    async def chat(self, messages):
        if "extract evidence for a research claim ledger" in messages[0]["content"].lower():
            return {"choices": [{"message": {"content": '[{"statement":"An evidence-grounded fact.","quote":"Evidence text","confidence":0.9}]'}}]}
        return {"choices": [{"message": {"content": "An evidence-grounded answer."}}]}

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_current_model_experiment_uses_active_server_without_starting_profile(monkeypatch):
    async def fake_model(base_url):
        return "current.gguf"

    async def fake_context(base_url):
        return 32768

    from deep_research.tools.extra_research import ResearchSource

    async def sources(queries, config, level, seen_urls, **kwargs):
        return [ResearchSource("Source", "https://huggingface.co/example", "Evidence text", level, queries[0], quality_score=5, source_kind="primary")]

    monkeypatch.setattr(experiments, "detect_model", fake_model)
    monkeypatch.setattr(experiments, "detect_context_size", fake_context)
    monkeypatch.setattr(experiments, "collect_sources", sources)
    monkeypatch.setattr(experiments, "LLMClient", _FakeLLM)

    db = _FakeDB()
    result = await experiments.run_model_experiment(
        db, Config(), {"id": "job-1", "payload": {"prompt": "Test prompt", "profile_slug": "current", "reasoning": False}},
    )

    assert result["model"] == "current.gguf"
    assert result["context_size"] == 32768
    assert result["reasoning"] is False
    assert result["answer"] == "An evidence-grounded answer."
    assert [stage for stage, _ in db.progress] == ["gather_sources", "evaluate"]


@pytest.mark.asyncio
async def test_larger_profile_safely_swaps_and_restores_primary_model(monkeypatch):
    primary = {
        "slug": "qwen3-14b", "display_name": "Qwen3 14B", "model_path": "/models/primary.gguf", "port": 18080,
        "server_args_json": "{}",
    }
    alternate = {
        "slug": "qwen3-30b", "display_name": "Qwen3 30B", "model_path": "/models/alternate.gguf", "port": 18080,
        "server_args_json": "{}",
    }
    calls = []

    async def fake_model(base_url):
        return "/models/primary.gguf" if "8080" in base_url else "/models/alternate.gguf"

    async def fake_context(base_url):
        return 16384

    async def fake_sources(queries, config, level, seen_urls, **kwargs):
        from deep_research.tools.extra_research import ResearchSource
        return [ResearchSource("Source", "https://huggingface.co/example", "Evidence text", level, queries[0], quality_score=5, source_kind="primary")]

    async def fake_get_model(config, slug):
        return alternate if slug == "qwen3-30b" else None

    async def fake_list_models(config):
        return [primary, alternate]

    async def fake_healthy(port):
        return False

    async def fake_start(profile):
        calls.append(("start", profile["slug"], profile["port"]))
        return True, "/tmp/model.log"

    async def fake_stop(profile):
        calls.append(("stop", profile["slug"], profile["port"]))
        return True

    monkeypatch.setattr(experiments, "detect_model", fake_model)
    monkeypatch.setattr(experiments, "detect_context_size", fake_context)
    monkeypatch.setattr(experiments, "collect_sources", fake_sources)
    monkeypatch.setattr(experiments, "LLMClient", _FakeLLM)
    monkeypatch.setattr(experiments.registry, "get_model", fake_get_model)
    monkeypatch.setattr(experiments.registry, "list_models", fake_list_models)
    monkeypatch.setattr(experiments, "is_healthy", fake_healthy)
    monkeypatch.setattr(experiments, "start_server", fake_start)
    monkeypatch.setattr(experiments, "stop_server", fake_stop)

    db = _FakeDB()
    result = await experiments.run_model_experiment(
        db, Config(), {"id": "job-1", "payload": {"prompt": "Test prompt", "profile_slug": "qwen3-30b"}},
    )

    assert result["profile"] == "qwen3-30b"
    assert calls == [
        ("stop", "qwen3-14b", 8080),
        ("start", "qwen3-30b", 18080),
        ("stop", "qwen3-30b", 18080),
        ("start", "qwen3-14b", 8080),
    ]
    assert [stage for stage, _ in db.progress] == ["swapping_model", "gather_sources", "evaluate", "restoring_model"]
