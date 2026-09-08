"""Datetime utilities for consistent timestamp handling."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Get current UTC datetime as ISO format string.

    Returns:
        ISO 8601 formatted datetime string
    """
    return datetime.now(UTC).isoformat()


def to_utc_iso(dt: datetime | None) -> str | None:
    """Serialize a datetime to a canonical UTC ISO string.

    Matches the format produced by ``utc_now_iso`` (used for created_at/updated_at)
    so stored timestamps sort correctly when compared lexically. Naive datetimes
    are assumed UTC; aware datetimes are converted to UTC.

    Args:
        dt: datetime to serialize, or None

    Returns:
        Canonical UTC ISO 8601 string, or None if input is None
    """
    if dt is None:
        return None
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.isoformat()


def parse_datetime_utc(dt_str: str | None) -> datetime | None:
    """Parse datetime string and ensure it's timezone-aware (UTC).

    Args:
        dt_str: ISO format datetime string, or None

    Returns:
        Timezone-aware datetime in UTC, or None if input is None
    """
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
