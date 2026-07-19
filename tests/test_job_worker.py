from types import SimpleNamespace

from deep_research.config import Config
from deep_research.kb import jobs
from deep_research.kb.jobs import ProcessingJobWorker, enqueue_manual_job
from deep_research.kb.storage import SnapshotStore


async def test_worker_claims_and_records_an_unhandled_job_failure(kb_db, tmp_path):
    """A worker failure must be durable and visible instead of disappearing
    with a background task exception."""
    job, _ = await kb_db.enqueue_processing_job(
        "future_job_type", "topic", subject_id="topic-1", idempotency_key="future:topic-1",
    )
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))

    assert await worker.run_once() is True

    completed = await kb_db.get_processing_job(job["id"])
    assert completed["status"] == "failed"
    assert "No worker handler" in completed["error_message"]
    assert completed["attempt_count"] == 1


async def test_worker_leaves_empty_queue_untouched(kb_db, tmp_path):
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))

    assert await worker.run_once() is False


async def test_worker_does_not_start_new_work_while_queue_is_paused(kb_db, tmp_path):
    job, _ = await kb_db.enqueue_processing_job(
        "future_job_type", "topic", subject_id="topic-1", idempotency_key="pause:topic-1",
    )
    await kb_db.set_processing_queue_paused(True)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))

    assert await worker.run_once() is False
    assert (await kb_db.get_processing_job(job["id"]))["status"] == "queued"


async def test_source_pipeline_treats_reused_extraction_as_completed(kb_db, tmp_path, monkeypatch):
    source, _ = await kb_db.get_or_create_source(
        source_type_code="web", canonical_uri="https://example.com/reused-extraction",
        canonical_key="reused-extraction",
    )
    version, _ = await kb_db.add_source_version(
        source["id"], content_hash="reused-hash", snapshot_path="/tmp/reused",
        http_status=200, mime_type="text/plain",
    )
    artifact, _ = await kb_db.upsert_artifact(
        artifact_id="reused-artifact", source_version_id=version["id"],
        artifact_type="clean_text", storage_path="/tmp/reused.txt",
        content_hash="reused-hash", chunk_params_hash="reused-params",
    )
    chunk = await kb_db.add_chunk(artifact["id"], 0, "A reused claim.", "reused-chunk")
    claim, _ = await kb_db.get_or_create_claim("fact", "A reused claim.")
    await kb_db.add_claim_evidence(
        claim_id=claim["id"], artifact_chunk_id=chunk["id"], source_id=source["id"],
        source_version_id=version["id"], excerpt_text="A reused claim.",
    )
    job = await enqueue_manual_job(
        kb_db, "source_pipeline", "source", source["id"], source_id=source["id"],
        payload={"version_id": version["id"], "defer_verification": True},
    )
    claimed = await kb_db.claim_next_processing_job("test-worker")

    async def no_trust_change(*args, **kwargs):
        return None

    async def reused_artifact(*args, **kwargs):
        return SimpleNamespace(chunk_count=1)

    async def reused_extraction(*args, **kwargs):
        return SimpleNamespace(
            status="unchanged", observation_count=1, extraction_run_id="existing-run",
        )

    async def should_not_promote(*args, **kwargs):
        raise AssertionError("an unchanged extraction must not be promoted twice")

    monkeypatch.setattr(jobs, "set_trust_tier_if_missing", no_trust_change)
    monkeypatch.setattr(jobs, "build_artifact_for_version", reused_artifact)
    monkeypatch.setattr(jobs, "run_extraction", reused_extraction)
    monkeypatch.setattr(jobs, "resolve_and_promote", should_not_promote)

    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))
    await worker._run_source_pipeline(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["stage"] == "complete"
    assert completed["error_message"] is None
    assert completed["progress"]["claim_count"] == 1
    assert completed["progress"]["observation_count"] == 1


async def test_topic_verification_refreshes_the_topic_report(kb_db, tmp_path, monkeypatch):
    topic = await kb_db.create_topic("Verification refresh")
    job = await enqueue_manual_job(kb_db, "topic_verify", "topic", topic["id"], topic_id=topic["id"])
    claimed = await kb_db.claim_next_processing_job("test-worker")
    assert claimed["id"] == job["id"]

    async def fake_verify(*args, **kwargs):
        return [({}, "supported", object())]

    generated = {"topic_id": None}

    async def fake_report(_db, _config, topic_id):
        generated["topic_id"] = topic_id
        return type("Report", (), {"report_id": "report-1"})()

    monkeypatch.setattr(jobs, "verify_claims_concurrently", fake_verify)
    monkeypatch.setattr(jobs, "generate_topic_report", fake_report)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))

    await worker._run_topic_verify(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["progress"]["report_id"] == "report-1"
    assert generated["topic_id"] == topic["id"]


async def test_ad_sweep_checks_only_claims_without_manual_override(kb_db, tmp_path, monkeypatch):
    open_claim, _ = await kb_db.get_or_create_claim("fact", "Potential sponsor claim")
    excluded_claim, _ = await kb_db.get_or_create_claim("fact", "Already reviewed claim")
    await kb_db.set_claim_verification_override(excluded_claim["id"], "exclude")
    job = await enqueue_manual_job(kb_db, "ad_sweep", "knowledge_base", "default", payload={"limit": 10})
    claimed = await kb_db.claim_next_processing_job("test-worker")

    seen = []

    async def fake_ad_check(_db, _config, claim_ids):
        seen.extend(claim_ids)
        return claim_ids

    monkeypatch.setattr(jobs, "check_claims_for_ads", fake_ad_check)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))
    await worker._run_ad_sweep(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert seen == [open_claim["id"]]
    assert completed["progress"] == {"claims_checked": 1, "claims_excluded": 1}


async def test_cron_verification_sweep_queues_counter_evidence(kb_db, tmp_path, monkeypatch):
    from deep_research.kb import verification

    job = await enqueue_manual_job(
        kb_db, "verification_sweep", "knowledge_base", "default",
        payload={"trigger": "cron", "verification_threshold": 0.7},
    )
    claimed = await kb_db.claim_next_processing_job("test-worker")
    seen = {"trigger": None, "limit": None}

    async def fake_verification_sweep(_db, _config, **kwargs):
        seen["trigger"] = kwargs["trigger"]
        return {"run_id": "verification-run-1"}

    async def fake_counter_queue(_db, *, limit):
        seen["limit"] = limit
        return {"eligible": 12, "queued": 10, "already_queued": 0, "remaining": 2}

    monkeypatch.setattr(verification, "run_verification_sweep", fake_verification_sweep)
    monkeypatch.setattr(jobs, "enqueue_supported_counter_evidence", fake_counter_queue)
    config = Config()
    config.kb.nightly_counter_evidence_limit = 10
    worker = ProcessingJobWorker(kb_db, config, SnapshotStore(tmp_path))

    await worker._run_verification_sweep(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert seen == {"trigger": "cron", "limit": 10}
    assert completed["status"] == "completed"
    assert completed["progress"] == {
        "verification_run_id": "verification-run-1",
        "counter_evidence": {"eligible": 12, "queued": 10, "already_queued": 0, "remaining": 2},
    }


async def test_manual_verification_sweep_does_not_queue_counter_evidence(kb_db, tmp_path, monkeypatch):
    from deep_research.kb import verification

    job = await enqueue_manual_job(
        kb_db, "verification_sweep", "knowledge_base", "default", payload={"trigger": "manual"},
    )
    claimed = await kb_db.claim_next_processing_job("test-worker")

    async def fake_verification_sweep(_db, _config, **kwargs):
        return {"run_id": "verification-run-2"}

    async def fail_counter_queue(*args, **kwargs):
        raise AssertionError("manual sweeps must not enqueue nightly counter-view work")

    monkeypatch.setattr(verification, "run_verification_sweep", fake_verification_sweep)
    monkeypatch.setattr(jobs, "enqueue_supported_counter_evidence", fail_counter_queue)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))

    await worker._run_verification_sweep(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert completed["progress"] == {
        "verification_run_id": "verification-run-2", "counter_evidence": None,
    }


async def test_claim_resolution_sweep_records_worker_progress(kb_db, tmp_path, monkeypatch):
    from deep_research.kb import resolution

    job = await enqueue_manual_job(
        kb_db, "claim_resolution_sweep", "knowledge_base", "default", payload={"limit": 25},
    )
    claimed = await kb_db.claim_next_processing_job("test-worker")
    seen = {"limit": None}

    async def fake_sweep(_db, _config, *, limit):
        seen["limit"] = limit
        return {"checked": 4, "merged": 2, "rejected": 1, "uncertain": 1, "failed": 0, "remaining": 1}

    monkeypatch.setattr(resolution, "resolve_open_claim_duplicates", fake_sweep)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))
    await worker._run_claim_resolution_sweep(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert seen["limit"] == 25
    assert completed["status"] == "completed"
    assert completed["progress"]["merged"] == 2


async def test_entity_resolution_sweep_records_worker_progress(kb_db, tmp_path, monkeypatch):
    from deep_research.kb import resolution

    job = await enqueue_manual_job(
        kb_db, "entity_resolution_sweep", "knowledge_base", "default", payload={"limit": 300},
    )
    claimed = await kb_db.claim_next_processing_job("test-worker")
    seen = {"limit": None}

    async def fake_sweep(_db, _config, *, limit):
        seen["limit"] = limit
        return {"checked": 20, "merged": 5, "rejected": 14, "uncertain": 1, "failed": 0, "remaining": 1}

    monkeypatch.setattr(resolution, "resolve_open_entity_duplicates", fake_sweep)
    worker = ProcessingJobWorker(kb_db, Config(), SnapshotStore(tmp_path))
    await worker._run_entity_resolution_sweep(claimed)

    completed = await kb_db.get_processing_job(job["id"])
    assert seen["limit"] == 300
    assert completed["status"] == "completed"
    assert completed["progress"]["merged"] == 5
