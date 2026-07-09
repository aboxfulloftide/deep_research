"""Timeline view for a topic (build order step 7, decision 27).

Timeline entries are strict: only claims with an `event_id` and a start date
qualify — no loose inclusion of claims with a date-like phrase in their text
but no formal event, since that risks pulling in claims that aren't actually
timeline-worthy. Dates are parsed best-effort at query time (not stored
differently); `events.start_at` stays exactly what extraction produced.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from deep_research.kb.db import KBDatabase

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_ISO_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_MONTH_YEAR_RE = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_ANY_YEAR_RE = re.compile(r"(19|20)\d{2}")


def parse_date_for_sorting(raw: str | None) -> date | None:
    """Best-effort parse of the free-text dates extraction actually produces
    ("2023-06-12", "2017", "March 2024", "2008-2013", "1999 and a part of
    2000") into a sortable date. Only used for ordering — the original
    raw text is what gets displayed, so this never fabricates false precision."""
    if not raw:
        return None
    text = raw.strip()

    m = _ISO_DATE_RE.match(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _ISO_MONTH_RE.match(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass

    m = _MONTH_YEAR_RE.match(text)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            return date(int(m.group(2)), month, 1)

    m = _YEAR_ONLY_RE.match(text)
    if m:
        return date(int(m.group(1)), 1, 1)

    # Fallback: first 4-digit year found anywhere — handles ranges like
    # "2008-2013" (takes the start) or loose phrasing like "1999 and a part
    # of 2000" (takes the first year mentioned).
    m = _ANY_YEAR_RE.search(text)
    if m:
        return date(int(m.group(0)), 1, 1)

    return None


@dataclass
class TimelineEntry:
    event: dict
    claims: list[dict] = field(default_factory=list)
    sort_date: date | None = None


async def get_topic_timeline(kb_db: KBDatabase, topic_id: str) -> list[TimelineEntry]:
    claims = await kb_db.list_topic_claims(topic_id, link_status="attached")

    events_by_id: dict[str, dict] = {}
    dates_by_id: dict[str, date] = {}
    claims_by_event: dict[str, list[dict]] = {}

    for claim in claims:
        event_id = claim.get("event_id")
        if not event_id:
            continue
        if event_id not in events_by_id:
            event = await kb_db.get_event(event_id)
            if event is None or not event.get("start_at"):
                continue
            sort_date = parse_date_for_sorting(event["start_at"])
            if sort_date is None:
                continue
            events_by_id[event_id] = event
            dates_by_id[event_id] = sort_date
            claims_by_event[event_id] = []
        if event_id in events_by_id:
            claims_by_event[event_id].append(claim)

    entries = [
        TimelineEntry(event=events_by_id[eid], claims=claims_by_event[eid], sort_date=dates_by_id[eid])
        for eid in events_by_id
    ]
    entries.sort(key=lambda e: e.sort_date)
    return entries
