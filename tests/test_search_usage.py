from datetime import datetime, timedelta, timezone

import aiosqlite

from deep_research.config import Config, DBConfig
from deep_research.tools.search_usage import (
    get_usage_summary,
    log_search_call,
    provider_monthly_quota_exhausted,
    providers_allowed_by_circuit_breaker,
    serper_behind_pace,
    serper_pace_status,
    serper_quota_remaining,
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


# -- serper_quota_remaining: manual balance snapshot, decayed by real calls -
# Serper's paid credit balance never resets monthly and isn't exposed via
# their search API, so it can't be queried live -- only estimated from a
# manually-checked snapshot (serper.dev/dashboard) minus every call logged
# since.

async def test_serper_quota_remaining_returns_none_without_a_snapshot(tmp_path):
    config = _config(tmp_path)
    assert await serper_quota_remaining(config) is None


async def test_serper_quota_remaining_decays_by_calls_since_snapshot(tmp_path):
    config = _config(tmp_path)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()

    await log_search_call(config, "serper", "api", "ok")
    await log_search_call(config, "serper", "api", "error", error_message="boom")

    assert await serper_quota_remaining(config) == 98


async def test_serper_quota_remaining_ignores_calls_before_the_snapshot(tmp_path):
    config = _config(tmp_path)
    # A call logged before the snapshot was taken must not count against it --
    # the snapshot already reflects the balance as of that point in time.
    await log_search_call(config, "serper", "api", "ok")
    config.serper.quota_remaining_snapshot = 50
    config.serper.quota_remaining_snapshot_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()

    assert await serper_quota_remaining(config) == 50


async def test_serper_quota_remaining_never_goes_negative(tmp_path):
    config = _config(tmp_path)
    config.serper.quota_remaining_snapshot = 1
    config.serper.quota_remaining_snapshot_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()

    for _ in range(3):
        await log_search_call(config, "serper", "api", "ok")

    assert await serper_quota_remaining(config) == 0


# -- serper_pace_status/serper_behind_pace: use-it-or-lose-it credits -------
# Serper's balance expires on a fixed date and is forfeited unused (unlike
# the monthly-resetting providers), so usage should be pushed higher as that
# date approaches rather than conserved.

async def test_serper_pace_status_returns_none_without_expiry_configured(tmp_path):
    config = _config(tmp_path)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = datetime.now(timezone.utc).isoformat()

    assert await serper_pace_status(config) is None


async def test_serper_pace_status_behind_pace_when_no_calls_made_halfway_to_expiry(tmp_path):
    config = _config(tmp_path)
    now = datetime.now(timezone.utc)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = (now - timedelta(days=50)).isoformat()
    config.serper.quota_expires_at = (now + timedelta(days=50)).isoformat()

    status = await serper_pace_status(config)

    assert status["remaining"] == 100
    assert status["expected_remaining"] == 50
    assert status["behind_pace"] is True
    assert status["days_left"] in (49, 50)


async def test_serper_pace_status_on_pace_when_spend_matches_schedule(tmp_path):
    config = _config(tmp_path)
    now = datetime.now(timezone.utc)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = (now - timedelta(days=50)).isoformat()
    config.serper.quota_expires_at = (now + timedelta(days=50)).isoformat()
    for _ in range(50):
        await log_search_call(config, "serper", "api", "ok")

    status = await serper_pace_status(config)

    assert status["remaining"] == 50
    assert status["behind_pace"] is False


async def test_serper_behind_pace_false_without_configuration(tmp_path):
    config = _config(tmp_path)
    assert await serper_behind_pace(config) is False


async def test_serper_behind_pace_true_when_underspending(tmp_path):
    config = _config(tmp_path)
    now = datetime.now(timezone.utc)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = (now - timedelta(days=50)).isoformat()
    config.serper.quota_expires_at = (now + timedelta(days=50)).isoformat()

    assert await serper_behind_pace(config) is True


async def test_usage_summary_includes_serper_pacing_fields(tmp_path):
    config = _config(tmp_path)
    now = datetime.now(timezone.utc)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = (now - timedelta(days=50)).isoformat()
    config.serper.quota_expires_at = (now + timedelta(days=50)).isoformat()

    summary = await get_usage_summary(config)
    serper = summary["providers"]["serper"]

    assert serper["quota_hard_limit"] is True
    assert serper["behind_pace"] is True
    assert serper["expected_remaining"] == 50
    assert serper["days_left"] in (49, 50)


async def test_usage_summary_includes_serper_quota_remaining(tmp_path):
    config = _config(tmp_path)
    config.serper.quota_remaining_snapshot = 100
    config.serper.quota_remaining_snapshot_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    await log_search_call(config, "serper", "api", "ok")

    summary = await get_usage_summary(config)
    assert summary["providers"]["serper"]["quota_remaining"] == 99


async def test_usage_summary_includes_tavily_quota_remaining_when_configured(tmp_path):
    """Unlike Serper's running paid balance, Tavily's free tier (1,000/month)
    actually resets on the calendar-month boundary already used for
    calls_month -- no manual snapshot needed, just a configured limit."""
    config = _config(tmp_path)
    config.tavily.monthly_quota = 1000
    await log_search_call(config, "tavily", "api", "ok")
    await log_search_call(config, "tavily", "api", "empty")

    summary = await get_usage_summary(config)
    assert summary["providers"]["tavily"]["quota_remaining"] == 998


async def test_usage_summary_omits_tavily_quota_remaining_when_not_configured(tmp_path):
    config = _config(tmp_path)
    await log_search_call(config, "tavily", "api", "ok")

    summary = await get_usage_summary(config)
    assert "quota_remaining" not in summary["providers"]["tavily"]


async def test_usage_summary_tracks_brave_primary_and_fallback_quotas_independently(tmp_path):
    """Primary and fallback are two separate Brave subscriptions with their
    own monthly allowances (see search.py's _brave_search_layered) -- each
    must be tracked against its own quota, not share one."""
    config = _config(tmp_path)
    config.brave.monthly_quota = 2000
    config.brave.fallback_monthly_quota = 2500
    await log_search_call(config, "brave", "api", "ok")
    for _ in range(5):
        await log_search_call(config, "brave_fallback", "api", "ok")

    summary = await get_usage_summary(config)
    assert summary["providers"]["brave"]["quota_remaining"] == 1999
    assert summary["providers"]["brave_fallback"]["quota_remaining"] == 2495


async def test_usage_summary_marks_brave_fallback_as_soft_limit_and_others_hard(tmp_path):
    """Brave's paid backup key keeps working past its budgeted monthly
    figure (it just costs more), while Brave primary/Tavily/Serper actually
    stop returning results once exhausted -- the distinction should be
    visible, not just a uniform "N remaining" for all four."""
    config = _config(tmp_path)
    config.brave.monthly_quota = 2000
    config.brave.fallback_monthly_quota = 2500
    config.tavily.monthly_quota = 1000
    await log_search_call(config, "brave", "api", "ok")
    await log_search_call(config, "brave_fallback", "api", "ok")
    await log_search_call(config, "tavily", "api", "ok")

    summary = await get_usage_summary(config)
    assert summary["providers"]["brave"]["quota_hard_limit"] is True
    assert summary["providers"]["tavily"]["quota_hard_limit"] is True
    assert summary["providers"]["brave_fallback"]["quota_hard_limit"] is False


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
