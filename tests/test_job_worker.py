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
