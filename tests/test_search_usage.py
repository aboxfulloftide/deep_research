from datetime import datetime, timezone

import aiosqlite

from deep_research.config import Config, DBConfig
from deep_research.tools.search_usage import (
    get_usage_summary,
    log_search_call,
    provider_monthly_quota_exhausted,
    providers_allowed_by_circuit_breaker,
    usage_db_path,
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
        error_message="500 Internal Server Error", error_category="server_error",
    )
    assert await provider_monthly_quota_exhausted(config, "brave") is False

    await log_search_call(
        config, "brave", "api", "error",
        error_message="429 Too Many Requests", error_category="rate_limited",
    )
    assert await provider_monthly_quota_exhausted(config, "brave") is True
    assert await provider_monthly_quota_exhausted(config, "brave_fallback") is False


async def test_monthly_quota_circuit_requires_the_typed_category_not_just_429_text(tmp_path):
    """Guards against reintroducing the old string-matching heuristic --
    a "429" substring in error_message alone must not open the circuit."""
    config = _config(tmp_path)
    await log_search_call(
        config, "brave", "api", "error",
        error_message="429 Too Many Requests",
    )
    assert await provider_monthly_quota_exhausted(config, "brave") is False


async def test_log_search_call_persists_run_plan_facet_attempt_and_capability(tmp_path):
    config = _config(tmp_path)
    await log_search_call(
        config, "serper", "api", "ok", result_count=2,
        run_id="run-1", plan_id="plan-1", facet_id="facet-1",
        attempt_id="attempt-1", capability="scholarly",
    )

    async with aiosqlite.connect(usage_db_path(config)) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT run_id, plan_id, facet_id, attempt_id, capability FROM search_calls",
        )

    assert dict(rows[0]) == {
        "run_id": "run-1", "plan_id": "plan-1", "facet_id": "facet-1",
        "attempt_id": "attempt-1", "capability": "scholarly",
    }


async def test_log_search_call_defaults_context_ids_to_null(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "brave", "api", "ok")

    async with aiosqlite.connect(usage_db_path(config)) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            "SELECT run_id, plan_id, facet_id, attempt_id, capability FROM search_calls",
        )

    assert all(value is None for value in dict(rows[0]).values())


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


async def test_usage_summary_uses_viewers_local_day_and_month(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "serper", "api", "ok", result_count=3)
    async with aiosqlite.connect(usage_db_path(config)) as db:
        await db.execute(
            "UPDATE search_calls SET created_at = ?",
            ("2026-07-24T23:23:00+00:00",),
        )
        await db.commit()

    # At this instant it is already July 25 UTC, but still July 24 in Detroit.
    now_utc = datetime(2026, 7, 25, 1, 38, tzinfo=timezone.utc)
    local_summary = await get_usage_summary(
        config, timezone_name="America/Detroit", now_utc=now_utc,
    )
    utc_summary = await get_usage_summary(
        config, timezone_name="UTC", now_utc=now_utc,
    )

    assert local_summary["providers"]["serper"]["calls_today"] == 1
    assert local_summary["providers"]["serper"]["calls_month"] == 1
    assert local_summary["timezone"] == "America/Detroit"
    assert utc_summary["providers"]["serper"]["calls_today"] == 0


async def test_usage_summary_keeps_local_month_until_local_midnight(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "bing", "scrape", "ok", result_count=8)
    async with aiosqlite.connect(usage_db_path(config)) as db:
        await db.execute(
            "UPDATE search_calls SET created_at = ?",
            ("2026-07-31T23:30:00+00:00",),
        )
        await db.commit()

    # UTC has reached August, while Detroit is still in the evening of July 31.
    summary = await get_usage_summary(
        config,
        timezone_name="America/Detroit",
        now_utc=datetime(2026, 8, 1, 1, 30, tzinfo=timezone.utc),
    )

    assert summary["providers"]["bing"]["calls_today"] == 1
    assert summary["providers"]["bing"]["calls_month"] == 1
