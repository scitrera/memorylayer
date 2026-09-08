"""Rule-based query intent classification for retrieval routing (no LLM).

Classifies a free-text query into intent labels (entity / temporal / event /
general) with cheap, deterministic heuristics (~microseconds, no network), and
extracts an explicit absolute date window when one is clearly present.

Routing (see ``MemoryService._route_by_intent``) uses these signals to *softly*
tune existing retrieval knobs — bumping recency, amplifying alias/backlink
boosts, enabling graph expansion. The only result-dropping action, a temporal
window filter, fires solely on a confidently parsed explicit date range; fuzzy
temporal cues ("recently") only nudge recency. A misclassification therefore
falls back to general behaviour and never drops results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

ENTITY = "entity"
TEMPORAL = "temporal"
EVENT = "event"
GENERAL = "general"

# Tight temporal vocabulary — deliberately excludes noisy words like "time",
# "last", "when", "date" that frequently appear in non-temporal queries.
_TEMPORAL_TOKENS = frozenset(
    {
        "yesterday",
        "today",
        "tomorrow",
        "recently",
        "lately",
        "ago",
        "since",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "quarter",
        "quarterly",
        "decade",
        "yearly",
        "monthly",
        "weekly",
    }
)
_TEMPORAL_PHRASES = (
    "last week",
    "last month",
    "last year",
    "last quarter",
    "this week",
    "this month",
    "this year",
    "past week",
    "past month",
    "past year",
    "over time",
)

_EVENT_TOKENS = frozenset(
    {
        "meeting",
        "meetings",
        "standup",
        "sync",
        "retro",
        "retrospective",
        "decided",
        "decision",
        "decisions",
        "launched",
        "launch",
        "shipped",
        "released",
        "release",
        "happened",
        "occurred",
        "incident",
        "outage",
    }
)
_EVENT_PHRASES = ("what happened", "when did", "when we", "the time we")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")
_YEAR_RE = re.compile(r"\b(19|20)\d\d\b")
# Two or more consecutive Capitalized words -> likely a proper-noun entity.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)+\b")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,})[\"']")

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_NAME_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)


@dataclass
class QueryIntent:
    """Classified intent for a query plus any explicitly parsed date window."""

    labels: set[str] = field(default_factory=set)
    event_after: datetime | None = None
    event_before: datetime | None = None

    def has(self, label: str) -> bool:
        return label in self.labels


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC) if month == 12 else datetime(year, month + 1, 1, tzinfo=UTC)
    return start, end


def _parse_explicit_window(lower: str) -> tuple[datetime | None, datetime | None]:
    """Parse a confident, explicit absolute date range from the query.

    Recognizes ISO dates (YYYY-MM-DD), year-month (YYYY-MM), month-name + year
    ("march 2021"), and bare year(s). Returns an inclusive [after, before] window
    (before is the last instant of the period). Returns (None, None) when no
    explicit absolute date is present — fuzzy/relative cues do not produce a
    window (they only nudge recency during routing).
    """
    m = _ISO_DATE_RE.search(lower)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            start = datetime(y, mo, d, tzinfo=UTC)
        except ValueError:
            return None, None
        return start, start + timedelta(days=1) - timedelta(microseconds=1)

    m = _MONTH_NAME_YEAR_RE.search(lower)
    if m:
        start, end = _month_bounds(int(m.group(2)), _MONTHS[m.group(1).lower()])
        return start, end - timedelta(microseconds=1)

    m = _YEAR_MONTH_RE.search(lower)
    if m and 1 <= int(m.group(2)) <= 12:
        start, end = _month_bounds(int(m.group(1)), int(m.group(2)))
        return start, end - timedelta(microseconds=1)

    years = [int(y) for y in re.findall(r"\b(?:19|20)\d\d\b", lower)]
    if years:
        start = datetime(min(years), 1, 1, tzinfo=UTC)
        end = datetime(max(years) + 1, 1, 1, tzinfo=UTC)
        return start, end - timedelta(microseconds=1)

    return None, None


def classify_query_intent(query: str) -> QueryIntent:
    """Classify a query into intent labels and extract any explicit date window."""
    text = query.strip()
    lower = text.lower()
    tokens = set(_TOKEN_RE.findall(lower))
    labels: set[str] = set()

    after, before = _parse_explicit_window(lower)
    if after is not None or before is not None or (tokens & _TEMPORAL_TOKENS) or any(p in lower for p in _TEMPORAL_PHRASES):
        labels.add(TEMPORAL)

    if (tokens & _EVENT_TOKENS) or any(p in lower for p in _EVENT_PHRASES):
        labels.add(EVENT)

    if _QUOTED_RE.search(text) or _PROPER_NOUN_RE.search(text):
        labels.add(ENTITY)

    if not labels:
        labels.add(GENERAL)

    return QueryIntent(labels=labels, event_after=after, event_before=before)
