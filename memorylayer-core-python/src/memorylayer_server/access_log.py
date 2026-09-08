"""Rollup of uvicorn access-log lines for k8s probe endpoints.

WHY. Probes dominate the access log completely: measured on a live tenant, a
600-line tail held 340 ``GET /livez`` and 256 ``GET /health/ready`` and ZERO
application requests. That is not merely noisy — it made a production incident
undiagnosable, because the callers actually consuming a per-user rate limit were
invisible behind the probe traffic.

WHAT. Individual probe lines are suppressed and replaced by ONE periodic summary
naming the paths and counts. Non-probe requests are untouched: real traffic still
logs line-per-request. Setting the interval to 0 drops probe lines silently.

HOW IT IS INSTALLED. uvicorn calls ``dictConfig`` during startup, which REPLACES
handlers and filters on its own loggers — so a filter attached before
``uvicorn.run`` is discarded. It has to arrive through ``log_config``; see
:func:`build_uvicorn_log_config`.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from memorylayer_server.probe_paths import is_probe_path

# Emitted under its own logger so the summary cannot be swallowed by the very
# filter that produces it.
_ROLLUP_LOGGER = "memorylayer.access.probes"

DEFAULT_PROBE_ROLLUP_SECONDS = 300


class ProbeAccessLogFilter(logging.Filter):
    """Suppress per-request probe access lines, emitting a periodic rollup instead.

    Attached to the ``uvicorn.access`` logger. uvicorn formats those records as
    ``'%s - "%s %s HTTP/%s" %d'`` with args
    ``(client_addr, method, full_path, http_version, status_code)``, so the path
    is ``record.args[2]``. Anything not matching that shape is passed through
    unfiltered rather than guessed at — a logging filter must never be the reason
    a line disappears.
    """

    def __init__(self, interval_seconds: int = DEFAULT_PROBE_ROLLUP_SECONDS) -> None:
        super().__init__()
        self._interval = max(0, int(interval_seconds))
        self._counts: Dict[str, int] = {}
        self._window_started = time.monotonic()
        self._logger = logging.getLogger(_ROLLUP_LOGGER)

    @staticmethod
    def _path_of(record: logging.LogRecord) -> Optional[str]:
        args: Any = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return None
        path = args[2]
        if not isinstance(path, str):
            return None
        # Strip any query string so ``/livez?x=1`` rolls up with ``/livez``.
        return path.split("?", 1)[0]

    def filter(self, record: logging.LogRecord) -> bool:
        path = self._path_of(record)
        if path is None or not is_probe_path(path):
            return True  # real traffic, or an unrecognised record shape

        self._counts[path] = self._counts.get(path, 0) + 1

        # interval 0 => suppress entirely, never summarise
        if self._interval:
            elapsed = time.monotonic() - self._window_started
            if elapsed >= self._interval:
                self._flush(elapsed)

        return False

    def _flush(self, elapsed: float) -> None:
        total = sum(self._counts.values())
        breakdown = ", ".join(
            f"{p}={n}" for p, n in sorted(self._counts.items(), key=lambda kv: -kv[1])
        )
        self._counts.clear()
        self._window_started = time.monotonic()
        self._logger.info(
            "health probes: %d requests in %.0fs (%s)", total, elapsed, breakdown
        )


def build_uvicorn_log_config(interval_seconds: int = DEFAULT_PROBE_ROLLUP_SECONDS) -> dict:
    """Return uvicorn's logging config with the probe rollup filter attached.

    Deep-copied from uvicorn's own ``LOGGING_CONFIG`` so formatters and handlers
    stay exactly as upstream defines them; only a filter is added. Pass the result
    as ``uvicorn.run(log_config=...)``.
    """
    import copy

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["probe_rollup"] = {
        "()": f"{__name__}.ProbeAccessLogFilter",
        "interval_seconds": interval_seconds,
    }
    access_logger = config.get("loggers", {}).get("uvicorn.access")
    if access_logger is not None:
        # Logger-level filter: uvicorn.access records are emitted through this
        # logger, so this catches them all regardless of handler wiring.
        access_logger.setdefault("filters", []).append("probe_rollup")
    # The rollup logger must propagate to root so the summary actually surfaces.
    config.setdefault("loggers", {})[_ROLLUP_LOGGER] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    return config
