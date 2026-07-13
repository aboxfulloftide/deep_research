import pytest

from deep_research.config import Config, DBConfig
from deep_research.evals import registry as r


@pytest.fixture
def config(tmp_path):
    """A Config whose db_path resolves under tmp_path, so registry_db_path()
    lands on a throwaway SQLite file instead of the real
    ~/.local/share/deep_research/eval_registry.db."""
    return Config(db=DBConfig(path=str(tmp_path / "research.db")))


async def test_register_and_get_model(config):
    await r.register_model(
        config, "qwen3-14b", model_path="/models/qwen3-14b.gguf", port=18080,
        server_args_json='{"gpu_layers": 99}', postgres_dsn="postgresql://x/deep_research_eval_qwen3_14b",
        snapshot_dir="/snap/qwen3-14b", config_path="/evals/configs/qwen3-14b.yaml",
        display_name="Qwen3-14B",
    )

    model = await r.get_model(config, "qwen3-14b")

    assert model is not None
    assert model["display_name"] == "Qwen3-14B"
    assert model["port"] == 18080
    assert model["postgres_dsn"] == "postgresql://x/deep_research_eval_qwen3_14b"


async def test_get_model_returns_none_for_unknown_slug(config):
    assert await r.get_model(config, "does-not-exist") is None


async def test_register_model_is_idempotent_and_updates_fields(config):
    await r.register_model(
        config, "qwen3-14b", model_path="/models/v1.gguf", port=18080,
        server_args_json="{}", postgres_dsn="postgresql://x/db1",
        snapshot_dir="/snap1", config_path="/cfg1.yaml",
    )
    await r.register_model(
        config, "qwen3-14b", model_path="/models/v2.gguf", port=18081,
        server_args_json="{}", postgres_dsn="postgresql://x/db2",
        snapshot_dir="/snap2", config_path="/cfg2.yaml", display_name="Updated",
    )

    models = await r.list_models(config)

    assert len(models) == 1  # re-registering the same slug updates, not duplicates
    assert models[0]["model_path"] == "/models/v2.gguf"
    assert models[0]["port"] == 18081
    assert models[0]["display_name"] == "Updated"


async def test_list_models_empty(config):
    assert await r.list_models(config) == []


async def test_list_models_returns_all_registered(config):
    for slug in ("model-a", "model-b"):
        await r.register_model(
            config, slug, model_path=f"/models/{slug}.gguf", port=18080,
            server_args_json="{}", postgres_dsn=f"postgresql://x/{slug}",
            snapshot_dir=f"/snap/{slug}", config_path=f"/cfg/{slug}.yaml",
        )

    models = await r.list_models(config)

    assert {m["slug"] for m in models} == {"model-a", "model-b"}


async def test_add_and_get_source(config):
    await r.add_source(config, "130years", url="https://youtube.com/watch?v=abc", title="130 Years")

    source = await r.get_source(config, "130years")

    assert source is not None
    assert source["url"] == "https://youtube.com/watch?v=abc"
    assert source["title"] == "130 Years"


async def test_get_source_returns_none_for_unknown_slug(config):
    assert await r.get_source(config, "does-not-exist") is None


async def test_add_source_is_idempotent_and_updates_fields(config):
    await r.add_source(config, "130years", url="https://youtube.com/watch?v=old")
    await r.add_source(config, "130years", url="https://youtube.com/watch?v=new", title="Updated")

    sources = await r.list_sources(config)

    assert len(sources) == 1
    assert sources[0]["url"] == "https://youtube.com/watch?v=new"
    assert sources[0]["title"] == "Updated"


async def test_list_sources_empty(config):
    assert await r.list_sources(config) == []
