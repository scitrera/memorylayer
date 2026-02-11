"""Datetime utilities for consistent timestamp handling."""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Get current UTC datetime as ISO format string.

    Returns:
        ISO 8601 formatted datetime string
    """
    return datetime.now(timezone.utc).isoformat()


def parse_datetime_utc(dt_str: Optional[str]) -> Optional[datetime]:
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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
