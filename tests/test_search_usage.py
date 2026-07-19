from deep_research.config import Config, DBConfig
from deep_research.tools.search_usage import (
    get_usage_summary,
    log_search_call,
    provider_monthly_quota_exhausted,
    providers_allowed_by_circuit_breaker,
)


def _config(tmp_path):
    return Config(db=DBConfig(path=str(tmp_path / "research.db")))


async def test_provider_circuit_breaker_opens_after_first_error(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "startpage", "scrape", "ok")
    await log_search_call(config, "google cse", "scrape", "error", error_message="too many requests")

    allowed = await providers_allowed_by_circuit_breaker(
        config, ("google cse", "startpage"), max_attempts=20, cooldown_hours=48,
    )

    assert allowed == {"startpage"}


async def test_provider_circuit_breaker_enforces_rolling_attempt_cap(tmp_path):
    config = _config(tmp_path)
    for _ in range(2):
        await log_search_call(config, "startpage", "scrape", "ok")

    allowed = await providers_allowed_by_circuit_breaker(
        config, ("startpage",), max_attempts=2, cooldown_hours=48,
    )

    assert allowed == set()


async def test_provider_circuit_breaker_can_apply_error_cooldown_without_attempt_cap(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "duckduckgo", "scrape", "error", error_message="CAPTCHA")

    allowed = await providers_allowed_by_circuit_breaker(
        config, ("duckduckgo",), max_attempts=None, cooldown_hours=1,
    )

    assert allowed == set()


async def test_monthly_quota_circuit_opens_only_for_logged_429(tmp_path):
    config = _config(tmp_path)
    await log_search_call(
        config, "brave", "api", "error",
        error_message="500 Internal Server Error",
    )
    assert await provider_monthly_quota_exhausted(config, "brave") is False

    await log_search_call(
        config, "brave", "api", "error",
        error_message="429 Too Many Requests",
    )
    assert await provider_monthly_quota_exhausted(config, "brave") is True
    assert await provider_monthly_quota_exhausted(config, "brave_fallback") is False


async def test_usage_summary_hides_retired_wikimedia_scrape_engines(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "wikipedia", "scrape", "error")
    await log_search_call(config, "wikidata", "scrape", "error")
    await log_search_call(config, "wikipedia_api", "api", "ok", result_count=2)
    await log_search_call(config, "wikidata_api", "api", "ok", result_count=1)

    summary = await get_usage_summary(config)

    assert "wikipedia" not in summary["providers"]
    assert "wikidata" not in summary["providers"]
    assert "wikipedia_api" in summary["providers"]
    assert "wikidata_api" in summary["providers"]
    assert {call["provider"] for call in summary["recent_calls"]} == {
        "wikipedia_api", "wikidata_api",
    }
