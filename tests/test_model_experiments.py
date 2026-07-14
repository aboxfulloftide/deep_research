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

    async def sources(queries, config, level, seen_urls):
        return [ResearchSource("Source", "https://example.test", "Evidence text", level, queries[0])]

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
