"""Cross-model comparison reporting -- replaces the one-off asyncpg scripts
hand-written through Bash for every comparison tonight (resolution rate by
model, claim counts, status breakdowns). Queries each registered model's own
Postgres DB live, so there's nothing to keep in sync -- see registry.py's
docstring for why there's no separate stats table.
"""

from dataclasses import dataclass, field

import asyncpg


@dataclass
class ModelSourceStats:
    model_slug: str
    display_name: str
    found: bool
    total_claims: int = 0
    eligible_claims: int = 0
    status_counts: dict = field(default_factory=dict)

    @property
    def resolved_count(self) -> int:
        return sum(self.status_counts.get(s, 0) for s in ("supported", "contradicted", "mixed"))

    @property
    def resolution_rate(self) -> float | None:
        if self.eligible_claims == 0:
            return None
        return self.resolved_count / self.eligible_claims


async def _find_source_id(conn: asyncpg.Connection, canonical_uri: str) -> str | None:
    return await conn.fetchval("SELECT id FROM sources WHERE canonical_uri = $1", canonical_uri)


async def compute_stats_for_source(
    postgres_dsn: str, display_name: str, model_slug: str, canonical_uri: str, importance_threshold: float = 0.8,
) -> ModelSourceStats:
    conn = await asyncpg.connect(postgres_dsn)
    try:
        source_id = await _find_source_id(conn, canonical_uri)
        if source_id is None:
            return ModelSourceStats(model_slug=model_slug, display_name=display_name, found=False)

        total = await conn.fetchval(
            "SELECT count(DISTINCT c.id) FROM claims c JOIN claim_evidence ce ON ce.claim_id = c.id "
            "WHERE ce.source_id = $1",
            source_id,
        )
        rows = await conn.fetch(
            "SELECT c.status, count(DISTINCT c.id) AS n FROM claims c "
            "JOIN claim_evidence ce ON ce.claim_id = c.id "
            "WHERE ce.source_id = $1 AND c.importance_score >= $2 GROUP BY c.status",
            source_id, importance_threshold,
        )
        status_counts = {r["status"]: r["n"] for r in rows}
        eligible = sum(status_counts.values())

        return ModelSourceStats(
            model_slug=model_slug, display_name=display_name, found=True,
            total_claims=total or 0, eligible_claims=eligible, status_counts=status_counts,
        )
    finally:
        await conn.close()
