"""Idle-time, advisory topic discovery based on unassigned claim clusters."""

from deep_research.kb.db import KBDatabase


async def discover_topic_proposals(kb_db: KBDatabase, minimum_claims: int = 3) -> dict:
    clusters = await kb_db.list_unassigned_entity_claim_clusters(minimum_claims)
    created = 0
    for cluster in clusters:
        proposal, was_created = await kb_db.create_topic_discovery_proposal(
            cluster["entity_id"], list(cluster["claim_ids"]), cluster["entity_name"],
            f"{len(cluster['claim_ids'])} unassigned claims share the entity {cluster['entity_name']!r}.",
        )
        if was_created:
            created += 1
            await kb_db.record_decision(
                "topic_discovery_proposal", "topic_discovery_proposal", proposal["id"], "topic proposal created",
                related_ids=list(cluster["claim_ids"]), reasoning=proposal["reasoning"], reversible=True,
            )
    return {"clusters_scanned": len(clusters), "proposals_created": created}
