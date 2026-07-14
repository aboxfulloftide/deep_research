"""Shared writer for the append-only automation action journal.

Automation modules should call this small boundary rather than inventing
ad-hoc audit rows. The database owns persistence and filtering; this module
keeps the product-level contract visible at each decision call site.
"""

from deep_research.kb.db import KBDatabase


async def record_decision(
    kb_db: KBDatabase,
    decision_type: str,
    subject_type: str,
    subject_id: str,
    decision: str,
    **kwargs,
) -> dict:
    """Append an automated action and its explanation to the journal."""
    return await kb_db.record_decision(
        decision_type, subject_type, subject_id, decision, **kwargs,
    )


async def record_undo(
    kb_db: KBDatabase,
    original_decision_id: str,
    decision_type: str,
    subject_type: str,
    subject_id: str,
    decision: str,
    **kwargs,
) -> dict:
    """Append a reversal row after the caller has safely applied the undo."""
    return await kb_db.record_decision(
        decision_type, subject_type, subject_id, decision,
        undo_of_decision_id=original_decision_id, **kwargs,
    )
