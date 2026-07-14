"""Durable worker for hands-off knowledge-base processing.

All long-running source work is represented by ``processing_jobs``. The
worker takes a PostgreSQL advisory lock before claiming work, so separate web
or CLI processes cannot simultaneously drive the one local GPU/search budget.
"""

import asyncio
import uuid

from deep_research.config import Config
from deep_research.kb.ad_check import check_claims_for_ads
from deep_research.kb.artifacts import build_artifact_for_version
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import run_extraction
from deep_research.kb.gpu_idle import gpu_is_idle
from deep_research.kb.reports import generate_topic_report
from deep_research.kb.resolution import resolve_and_promote
from deep_research.kb.storage import SnapshotStore
from deep_research.kb.topics import check_claims_against_topics
from deep_research.kb.trust import set_trust_tier_if_missing
from deep_research.kb.verification import is_claim_eligible_for_verification, verify_claims_concurrently


# Stable project-wide key, not a security boundary. PostgreSQL advisory locks
# are held per connection, which lets unrelated DB work continue normally.
GPU_WORKER_ADVISORY_LOCK = 734_918


async def enqueue_source_pipeline(
    kb_db: KBDatabase,
    source_id: str,
    version_id: str,
    *,
    topic_id: str | None = None,
    threshold: float | None = None,
    chunk_size: int | None = None,
    force_extract: bool = False,
    defer_verification: bool = False,
    priority: int = 100,
) -> tuple[dict, bool]:
    """Queue one idempotent source pipeline for an ingested source version."""
    scope = topic_id or "independent"
    return await kb_db.enqueue_processing_job(
        "source_pipeline", "source", subject_id=source_id, source_id=source_id,
        topic_id=topic_id, idempotency_key=f"source_pipeline:{source_id}:{version_id}:{scope}",
        payload={
            "version_id": version_id, "verification_threshold": threshold,
            "chunk_size": chunk_size, "force_extract": force_extract,
            "defer_verification": defer_verification,
        }, priority=priority,
        stage="queued",
    )


async def enqueue_manual_job(
    kb_db: KBDatabase, job_type: str, subject_type: str, subject_id: str,
    *, source_id: str | None = None, topic_id: str | None = None, payload: dict | None = None,
) -> dict:
    """Queue an explicit user action. A fresh key deliberately preserves
    repeat user requests; the worker still serializes their GPU/search work."""
    job, _ = await kb_db.enqueue_processing_job(
        job_type, subject_type, subject_id=subject_id, source_id=source_id, topic_id=topic_id,
        idempotency_key=f"{job_type}:{subject_id}:{uuid.uuid4()}", payload=payload or {}, priority=200,
    )
    return job


async def enqueue_playlist_poll(kb_db: KBDatabase, playlist_id: str) -> tuple[dict, bool]:
    """Queue low-priority discovery work; the worker's GPU-idle gate applies."""
    return await kb_db.enqueue_processing_job(
        "playlist_poll", "playlist", playlist_id,
        idempotency_key=f"playlist_poll:{playlist_id}", priority=-100, is_speculative=True,
    )


class ProcessingJobWorker:
    """Single logical worker for durable KB work.

    Multiple application processes may instantiate this class. PostgreSQL's
    advisory lock means only one actually runs a GPU/search-driving job at a
    time; queued work remains visible and recoverable in the database.
    """

    def __init__(self, kb_db: KBDatabase, config: Config, snapshot_store: SnapshotStore):
        self.kb_db = kb_db
        self.config = config
        self.snapshot_store = snapshot_store
        self.worker_id = f"web-{uuid.uuid4()}"
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        await self.kb_db.requeue_expired_processing_jobs()
        self._task = asyncio.create_task(self.run_forever(), name="deep-research-kb-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run_forever(self, idle_seconds: float = 1.0) -> None:
        while not self._stop_event.is_set():
            ran = await self.run_once()
            if not ran:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=idle_seconds)
                except asyncio.TimeoutError:
                    pass

    async def run_once(self) -> bool:
        """Run at most one job while holding the global GPU/search lock."""
        async with self.kb_db.pool.acquire() as lock_conn:
            acquired = await lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", GPU_WORKER_ADVISORY_LOCK)
            if not acquired:
                return False
            try:
                job = await self.kb_db.claim_next_processing_job(self.worker_id)
                if job is None:
                    return False
                if job["is_speculative"] and not await gpu_is_idle():
                    await self.kb_db.release_processing_job(job["id"])
                    return False
                try:
                    await self._run_job(job)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    current = await self.kb_db.get_processing_job(job["id"])
                    if current and current["status"] == "running":
                        await self.kb_db.finish_processing_job(job["id"], "failed", error_message=str(exc))
                return True
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock($1)", GPU_WORKER_ADVISORY_LOCK)

    async def _cancelled(self, job_id: str) -> bool:
        job = await self.kb_db.get_processing_job(job_id)
        return job is None or bool(job["cancel_requested"])

    async def _finish_cancelled(self, job_id: str, stage: str) -> None:
        await self.kb_db.finish_processing_job(job_id, "cancelled", stage=stage)

    async def _run_job(self, job: dict) -> None:
        if job["job_type"] == "source_pipeline":
            await self._run_source_pipeline(job)
            return
        if job["job_type"] == "source_verify":
            await self._run_source_verify(job)
            return
        if job["job_type"] == "topic_verify":
            await self._run_topic_verify(job)
            return
        if job["job_type"] == "claim_verify":
            await self._run_claim_verify(job)
            return
        if job["job_type"] == "verification_sweep":
            await self._run_verification_sweep(job)
            return
        if job["job_type"] == "ad_sweep":
            await self._run_ad_sweep(job)
            return
        if job["job_type"] == "contradiction_triage":
            await self._run_contradiction_triage(job)
            return
        if job["job_type"] == "playlist_poll":
            await self._run_playlist_poll(job)
            return
        if job["job_type"] == "counter_evidence":
            await self._run_counter_evidence(job)
            return
        if job["job_type"] == "topic_discovery":
            await self._run_topic_discovery(job)
            return
        else:
            await self.kb_db.finish_processing_job(
                job["id"], "failed", error_message=f"No worker handler for job type {job['job_type']!r}",
            )

    async def _run_source_pipeline(self, job: dict) -> None:
        """One shared ingest-aftercare path for web, conversation, and later CLI/playlist jobs."""
        job_id = job["id"]
        source_id = job.get("source_id")
        produced_output = False
        try:
            if not source_id:
                raise ValueError("source_pipeline job has no source_id")
            source = await self.kb_db.get_source(source_id)
            if source is None:
                raise ValueError(f"source {source_id!r} no longer exists")
            version = await self.kb_db.get_latest_version(source_id)
            if version is None:
                raise ValueError(f"source {source_id!r} has no latest version")
            requested_version_id = (job.get("payload") or {}).get("version_id")
            if requested_version_id and requested_version_id != version["id"]:
                raise ValueError("source changed after this job was queued; enqueue its latest version instead")

            await self.kb_db.update_processing_job_progress(job_id, "trust", {"source_id": source_id}, lease_seconds=900)
            if await self._cancelled(job_id):
                await self._finish_cancelled(job_id, "trust")
                return
            await set_trust_tier_if_missing(self.kb_db, self.config, source_id)

            await self.kb_db.update_processing_job_progress(job_id, "chunk", lease_seconds=900)
            chunk_result = await build_artifact_for_version(
                self.kb_db, self.snapshot_store, source, version, config=self.config,
                chunk_size=(job.get("payload") or {}).get("chunk_size") or 1200,
            )
            if chunk_result.chunk_count == 0:
                await self.kb_db.finish_processing_job(
                    job_id, "completed", stage="complete", progress={"chunk_count": 0, "claim_count": 0},
                )
                return
            if await self._cancelled(job_id):
                await self._finish_cancelled(job_id, "chunk")
                return

            artifacts = await self.kb_db.get_current_artifacts_for_version(version["id"])
            if not artifacts:
                raise RuntimeError("chunking completed without a current artifact")
            await self.kb_db.update_processing_job_progress(
                job_id, "extract", {"chunk_count": chunk_result.chunk_count}, lease_seconds=900,
            )
            extraction = await run_extraction(
                self.kb_db, self.config, artifacts[0]["id"],
                force=bool((job.get("payload") or {}).get("force_extract")),
            )
            produced_output = extraction.observation_count > 0
            new_claim_ids: list[str] = []
            if extraction.status in ("extracted", "partial"):
                promotion = await resolve_and_promote(self.kb_db, self.config, extraction.extraction_run_id)
                new_claim_ids = promotion.new_claim_ids
                produced_output = produced_output or bool(new_claim_ids)
                await self.kb_db.update_processing_job_progress(
                    job_id, "ad_check", {"new_claim_count": len(new_claim_ids)}, lease_seconds=900,
                )
                await check_claims_for_ads(self.kb_db, self.config, new_claim_ids)
                await check_claims_against_topics(self.kb_db, self.config, new_claim_ids)
            else:
                raise RuntimeError(f"extraction ended with status {extraction.status!r}")
            if await self._cancelled(job_id):
                await self._finish_cancelled(job_id, "extract")
                return

            # Explicit membership is authoritative: reattach after promotion so
            # claims created after the initial source link enter the topic too.
            if job.get("topic_id"):
                await self.kb_db.update_processing_job_progress(job_id, "attach", lease_seconds=900)
                await self.kb_db.attach_source_to_topic(
                    job["topic_id"], source_id, link_reason="pipeline_source_attached",
                )

            source_claims = await self.kb_db.list_claims_for_source(source_id, limit=5000)
            threshold = (job.get("payload") or {}).get("verification_threshold")
            threshold = threshold if threshold is not None else self.config.kb.verification_importance_threshold
            eligible = [claim for claim in source_claims if is_claim_eligible_for_verification(claim, threshold)]
            if not (job.get("payload") or {}).get("defer_verification"):
                await self.kb_db.update_processing_job_progress(job_id, "verify", lease_seconds=900)
                await verify_claims_concurrently(self.kb_db, self.config, eligible)
            else:
                eligible = []

            if job.get("topic_id"):
                await self.kb_db.update_processing_job_progress(job_id, "report", lease_seconds=900)
                await generate_topic_report(self.kb_db, self.config, job["topic_id"])

            status = "partial" if extraction.status == "partial" else "completed"
            await self.kb_db.finish_processing_job(
                job_id, status, stage="complete",
                progress={
                    "chunk_count": chunk_result.chunk_count,
                    "observation_count": extraction.observation_count,
                    "claim_count": len(source_claims),
                    "verified_count": len(eligible),
                },
            )
        except asyncio.CancelledError:
            # The server is shutting down. Leave the lease to expire so the
            # next worker can requeue it rather than inventing a false result.
            raise
        except Exception as exc:
            current = await self.kb_db.get_processing_job(job_id)
            if current is None or current["status"] in ("partial", "failed", "completed", "cancelled"):
                return
            if current["cancel_requested"]:
                await self._finish_cancelled(job_id, current["stage"])
                return
            await self.kb_db.finish_processing_job(
                job_id, "partial" if produced_output else "failed", error_message=str(exc),
            )

    async def _run_source_verify(self, job: dict) -> None:
        source_id = job.get("source_id")
        if not source_id:
            raise ValueError("source_verify job has no source_id")
        payload = job.get("payload") or {}
        await self.kb_db.update_processing_job_progress(job["id"], "verify", lease_seconds=900)
        source_claims = await self.kb_db.list_claims_for_source(source_id, limit=5000)
        threshold = payload.get("verification_threshold")
        threshold = threshold if threshold is not None else self.config.kb.verification_importance_threshold
        eligible = [c for c in source_claims if is_claim_eligible_for_verification(c, threshold, force=bool(payload.get("force")))]
        outcomes = await verify_claims_concurrently(self.kb_db, self.config, eligible, force=bool(payload.get("force")))
        failed = sum(isinstance(result, Exception) for _, _, result in outcomes)
        await self.kb_db.finish_processing_job(
            job["id"], "partial" if failed else "completed", stage="complete",
            progress={"verified_count": len(outcomes), "failed_count": failed},
        )

    async def _run_topic_verify(self, job: dict) -> None:
        topic_id = job.get("topic_id")
        if not topic_id:
            raise ValueError("topic_verify job has no topic_id")
        payload = job.get("payload") or {}
        await self.kb_db.update_processing_job_progress(job["id"], "verify", lease_seconds=900)
        claims = await self.kb_db.list_topic_claims(topic_id, link_status="attached")
        threshold = payload.get("verification_threshold")
        threshold = threshold if threshold is not None else self.config.kb.verification_importance_threshold
        eligible = [c for c in claims if is_claim_eligible_for_verification(c, threshold, force=bool(payload.get("force")))]
        outcomes = await verify_claims_concurrently(self.kb_db, self.config, eligible, force=bool(payload.get("force")))
        failed = sum(isinstance(result, Exception) for _, _, result in outcomes)
        report_id = None
        if outcomes:
            # A verification result materially changes how assertions should
            # read in the overview, so refresh the cached topic report while
            # the same serialized worker still owns the model.
            await self.kb_db.update_processing_job_progress(job["id"], "report", lease_seconds=900)
            report = await generate_topic_report(self.kb_db, self.config, topic_id)
            report_id = report.report_id
        await self.kb_db.finish_processing_job(
            job["id"], "partial" if failed else "completed", stage="complete",
            progress={"verified_count": len(outcomes), "failed_count": failed, "report_id": report_id},
        )

    async def _run_claim_verify(self, job: dict) -> None:
        from deep_research.kb.verification import verify_claim

        claim_id = job.get("subject_id")
        if not claim_id:
            raise ValueError("claim_verify job has no claim id")
        await self.kb_db.update_processing_job_progress(job["id"], "verify", lease_seconds=900)
        result = await verify_claim(self.kb_db, self.config, claim_id, force=bool((job.get("payload") or {}).get("force")))
        await self.kb_db.finish_processing_job(
            job["id"], "completed", stage="complete", progress={"status": result.status},
        )

    async def _run_verification_sweep(self, job: dict) -> None:
        from deep_research.kb.verification import run_verification_sweep

        payload = job.get("payload") or {}
        await self.kb_db.update_processing_job_progress(job["id"], "verify", lease_seconds=900)
        result = await run_verification_sweep(
            self.kb_db, self.config, trigger="job_worker", threshold=payload.get("verification_threshold"),
            limit=payload.get("limit"), force=bool(payload.get("force")),
        )
        await self.kb_db.finish_processing_job(
            job["id"], "completed", stage="complete", progress={"verification_run_id": result["run_id"]},
        )

    async def _run_ad_sweep(self, job: dict) -> None:
        payload = job.get("payload") or {}
        await self.kb_db.update_processing_job_progress(job["id"], "ad_check", lease_seconds=900)
        claims = await self.kb_db.list_claims(limit=payload.get("limit") or 10000)
        candidate_ids = [claim["id"] for claim in claims if claim.get("verification_override") is None]
        flagged = await check_claims_for_ads(self.kb_db, self.config, candidate_ids)
        await self.kb_db.finish_processing_job(
            job["id"], "completed", stage="complete",
            progress={"claims_checked": len(candidate_ids), "claims_excluded": len(flagged)},
        )

    async def _run_contradiction_triage(self, job: dict) -> None:
        from deep_research.kb.contradiction_triage import triage_contradiction

        await self.kb_db.update_processing_job_progress(job["id"], "triage", lease_seconds=900)
        result = await triage_contradiction(self.kb_db, self.config, job["subject_id"])
        await self.kb_db.finish_processing_job(
            job["id"], "completed", stage="complete",
            progress={"recommendation": result["triage_recommendation"]},
        )

    async def _run_playlist_poll(self, job: dict) -> None:
        from deep_research.kb.playlists import poll_playlist

        await self.kb_db.update_processing_job_progress(job["id"], "discover", lease_seconds=900)
        result = await poll_playlist(self.kb_db, self.config, self.snapshot_store, job["subject_id"])
        await self.kb_db.finish_processing_job(job["id"], "completed", stage="complete", progress=result)

    async def _run_counter_evidence(self, job: dict) -> None:
        from deep_research.kb.counter_evidence import find_strongest_counter_claim
        await self.kb_db.update_processing_job_progress(job["id"], "counter_evidence", lease_seconds=900)
        result = await find_strongest_counter_claim(
            self.kb_db, self.config, job["subject_id"], force=bool((job.get("payload") or {}).get("force")),
        )
        await self.kb_db.finish_processing_job(job["id"], "completed", stage="complete", progress=result)

    async def _run_topic_discovery(self, job: dict) -> None:
        from deep_research.kb.topic_discovery import discover_topic_proposals
        await self.kb_db.update_processing_job_progress(job["id"], "discover", lease_seconds=900)
        result = await discover_topic_proposals(self.kb_db)
        await self.kb_db.finish_processing_job(job["id"], "completed", stage="complete", progress=result)
