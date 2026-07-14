from datetime import datetime, timedelta, timezone

import pytest

from deep_research.kb import decision_log
from deep_research.kb.jobs import enqueue_manual_job, enqueue_model_experiment


async def test_enqueue_processing_job_is_idempotent_and_leases_once(kb_db):
    job, created = await kb_db.enqueue_processing_job(
        "source_pipeline", "source_request",
        idempotency_key="source_pipeline:https-example-article:v1",
        payload={"url": "https://example.com/article"}, priority=10,
    )
    duplicate, duplicate_created = await kb_db.enqueue_processing_job(
        "source_pipeline", "source_request",
        idempotency_key="source_pipeline:https-example-article:v1",
        payload={"url": "https://example.com/article"}, priority=10,
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == job["id"]

    claimed = await kb_db.claim_next_processing_job("worker-a", lease_seconds=60)
    assert claimed["id"] == job["id"]
    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert await kb_db.claim_next_processing_job("worker-b") is None


async def test_processing_job_cancel_retry_and_expired_lease_recovery(kb_db):
    job, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source_request",
        idempotency_key="source_pipeline:recoverable:v1",
    )
    claimed = await kb_db.claim_next_processing_job("worker-a", lease_seconds=60)
    assert claimed["id"] == job["id"]

    progress = await kb_db.update_processing_job_progress(
        job["id"], "extract", {"chunks_total": 4, "chunks_done": 1}, lease_seconds=60,
    )
    assert progress["stage"] == "extract"
    assert progress["progress"] == {"chunks_total": 4, "chunks_done": 1}

    requested = await kb_db.request_processing_job_cancel(job["id"])
    assert requested["status"] == "running"
    assert requested["cancel_requested"] is True
    cancelled = await kb_db.finish_processing_job(job["id"], "cancelled", stage="extract")
    assert cancelled["status"] == "cancelled"

    retried = await kb_db.retry_processing_job(job["id"])
    assert retried["status"] == "queued"
    assert retried["cancel_requested"] is False

    reclaimed = await kb_db.claim_next_processing_job("worker-b", lease_seconds=60)
    assert reclaimed["id"] == job["id"]
    assert reclaimed["attempt_count"] == 2

    async with kb_db.pool.acquire() as conn:
        await conn.execute(
            "UPDATE processing_jobs SET lease_expires_at = $1 WHERE id = $2",
            datetime.now(timezone.utc) - timedelta(seconds=1), job["id"],
        )
    recovered = await kb_db.requeue_expired_processing_jobs()
    assert [row["id"] for row in recovered] == [job["id"]]
    assert recovered[0]["status"] == "queued"


async def test_queued_processing_job_cancels_without_a_worker(kb_db):
    job, _ = await kb_db.enqueue_processing_job(
        "report_refresh", "topic", idempotency_key="report:topic-1:v1", subject_id="topic-1",
    )
    cancelled = await kb_db.request_processing_job_cancel(job["id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["completed_at"] is not None
    assert await kb_db.claim_next_processing_job("worker-a") is None


async def test_processing_queue_pause_is_persistent_and_reversible(kb_db):
    initial = await kb_db.get_processing_queue_control()
    assert initial["paused"] is False

    paused = await kb_db.set_processing_queue_paused(True)
    assert paused["paused"] is True
    assert paused["paused_at"] is not None
    assert (await kb_db.get_processing_queue_control())["paused"] is True

    resumed = await kb_db.set_processing_queue_paused(False)
    assert resumed["paused"] is False
    assert resumed["paused_at"] is None


async def test_explicit_user_actions_get_separate_durable_jobs(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/manual-jobs", canonical_key="manual-jobs",
    )
    first = await enqueue_manual_job(kb_db, "source_verify", "source", source["id"], source_id=source["id"])
    second = await enqueue_manual_job(kb_db, "source_verify", "source", source["id"], source_id=source["id"])

    assert first["id"] != second["id"]
    assert {job["id"] for job in await kb_db.list_processing_jobs(source_id=source["id"])} == {first["id"], second["id"]}


async def test_model_experiments_are_low_priority_speculative_jobs(kb_db):
    experiment = await enqueue_model_experiment(kb_db, {"prompt": "Compare models"})
    normal, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id="source-1", idempotency_key="normal-before-experiment", priority=100,
    )

    claimed = await kb_db.claim_next_processing_job("worker")

    assert experiment["is_speculative"] is True
    assert experiment["priority"] == -1000
    assert experiment["stage"] == "waiting_for_idle"
    assert claimed["id"] == normal["id"]


async def test_queued_jobs_can_be_moved_to_front_or_back(kb_db):
    first, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id="source-1", idempotency_key="move-first", priority=10,
    )
    second, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id="source-2", idempotency_key="move-second", priority=10,
    )

    promoted = await kb_db.move_processing_job_in_queue(second["id"], "next")
    assert promoted["priority"] > first["priority"]
    assert (await kb_db.claim_next_processing_job("worker"))["id"] == second["id"]

    await kb_db.release_processing_job(second["id"])
    demoted = await kb_db.move_processing_job_in_queue(second["id"], "back")
    assert demoted["priority"] < first["priority"]


async def test_model_experiment_can_explicitly_run_after_current_work(kb_db):
    experiment = await enqueue_model_experiment(kb_db, {"prompt": "Compare models"})
    normal, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id="source-1", idempotency_key="normal-queued", priority=100,
    )

    promoted = await kb_db.prioritize_model_experiment(experiment["id"])

    assert promoted["payload"]["run_after_current"] is True
    assert promoted["priority"] > normal["priority"]


async def test_source_list_exposes_durable_lifecycle_status(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/lifecycle", canonical_key="lifecycle",
    )
    job, _ = await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id=source["id"], source_id=source["id"],
        idempotency_key="source_pipeline:lifecycle:v1",
    )

    listed = next(row for row in await kb_db.list_sources(limit=10) if row["id"] == source["id"])
    assert listed["lifecycle"] == "queued"
    assert listed["processing_status"] == "queued"

    await kb_db.claim_next_processing_job("worker")
    await kb_db.finish_processing_job(job["id"], "failed", error_message="snapshot unavailable")
    listed = next(row for row in await kb_db.list_sources(limit=10) if row["id"] == source["id"])
    assert listed["lifecycle"] == "failed"
    assert listed["processing_error"] == "snapshot unavailable"


async def test_archived_sources_are_hidden_by_default_but_restorable(kb_db):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/archive", canonical_key="archive",
    )
    assert await kb_db.set_source_active(source["id"], False)

    assert not any(row["id"] == source["id"] for row in await kb_db.list_sources(limit=20))
    archived = await kb_db.list_sources(limit=20, include_inactive=True)
    assert next(row for row in archived if row["id"] == source["id"])["is_active"] is False

    await kb_db.set_source_active(source["id"], True)
    assert any(row["id"] == source["id"] for row in await kb_db.list_sources(limit=20))


async def test_decision_journal_preserves_explanation_parse_state_and_undo_chain(kb_db):
    original = await decision_log.record_decision(
        kb_db, "trust_tier", "source", "source-1", "tier: reputable_reporting",
        confidence=0.91, reasoning="Known major newsroom", model="qwen3-14b",
        parse_success=True, previous_state={"trust_tier_code": None},
        resulting_state={"trust_tier_code": "reputable_reporting"}, reversible=True,
    )
    undo = await decision_log.record_undo(
        kb_db, original["id"], "trust_tier_reset", "source", "source-1", "tier reset to automatic",
        previous_state={"trust_tier_code": "reputable_reporting"},
        resulting_state={"trust_tier_code": None}, reversible=False,
    )

    assert original["parse_success"] is True
    assert original["previous_state"] == {"trust_tier_code": None}
    assert undo["undo_of_decision_id"] == original["id"]

    decisions = await kb_db.list_decisions(subject_type="source", subject_id="source-1")
    assert [row["id"] for row in decisions] == [undo["id"], original["id"]]


async def test_decision_journal_rejects_undo_of_nonreversible_action(kb_db):
    original = await kb_db.record_decision(
        "playlist_video_ingested", "source", "source-2", "ingested", reversible=False,
    )

    with pytest.raises(ValueError, match="not marked reversible"):
        await decision_log.record_undo(
            kb_db, original["id"], "playlist_video_uningested", "source", "source-2", "undone",
        )
