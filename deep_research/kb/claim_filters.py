"""High-confidence claim rules shared by verification and review generation."""

import re


_DATED_NOBODY_WANTS_TO_WORK_RE = re.compile(
    r"^\s*in\s+(?:18|19|20)\d{2},\s+(?:no\s+one|nobody)\s+"
    r"(?:wanted|wants)\s+to\s+work(?:\s+anymore)?[.!]?\s*$",
    re.IGNORECASE,
)


def is_review_excluded_claim(claim_text: str) -> bool:
    """Claims deliberately retained in the KB but excluded from pair review.

    The dated "nobody wants to work" refrain is a series of examples used to
    illustrate one source's broader argument. Comparing each year-specific
    fragment against generic labor claims creates noise, while merging them
    would erase the chronology the source is documenting.
    """
    return bool(_DATED_NOBODY_WANTS_TO_WORK_RE.fullmatch(claim_text or ""))
