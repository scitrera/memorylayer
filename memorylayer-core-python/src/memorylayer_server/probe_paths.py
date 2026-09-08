"""Canonical internal probe/metrics paths.

Single source of truth for "this request is infrastructure, not user traffic".
Shared by the rate-limit middleware (which must never throttle k8s probes) and the
uvicorn access-log rollup (which must not drown the log in them). Keeping one list
means a new probe endpoint cannot be exempt from one and not the other.
"""
from __future__ import annotations

# Prefix-matched. ``/v1/health`` is listed separately because the enterprise
# dependency probe lives under ``/v1`` and so does not match ``/health``.
PROBE_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/healthz",
    "/livez",
    "/metrics",
    "/v1/health",
)


def is_probe_path(path: str) -> bool:
    """Return ``True`` if ``path`` is an internal probe/metrics endpoint.

    Matches the path itself or any child segment, so ``/health`` also covers
    ``/health/ready`` and ``/health/live`` — but NOT ``/healthcheck-something``,
    which is a different route that happens to share a prefix.
    """
    return any(path == p or path.startswith(p + "/") for p in PROBE_PATH_PREFIXES)
