"""
Unit tests for the rate-limit middleware exempt-path list.

These guard the invariant that internal infrastructure probes (k8s
liveness/readiness) and metrics scrapes are NEVER throttled. Regression
context: ``/livez`` (the connection-gated liveness probe added for Aether
reconnect-robustness) lives at the root rather than under ``/health`` and was
not in the exempt list, so the limiter returned 429 to the kubelet liveness
probe and the pod crashlooped.
"""

import time

import pytest
from fastapi import FastAPI
from scitrera_app_framework import Variables
from starlette.testclient import TestClient

from memorylayer_server.middleware import rate_limit as rl
from memorylayer_server.middleware.rate_limit import (
    RateLimitMiddleware,
    _is_rate_limit_exempt,
)
from memorylayer_server.services.rate_limit.base import RateLimitResult, RateLimitService

# Paths that MUST bypass the limiter (liveness/readiness/metrics/deps).
EXEMPT_PATHS = (
    "/health",
    "/health/ready",
    "/health/live",
    "/healthz",
    "/livez",
    "/metrics",
    "/v1/health/dependencies",
)


# ---------------------------------------------------------------------------
# _is_rate_limit_exempt() helper
# ---------------------------------------------------------------------------


class TestIsRateLimitExempt:
    """Prefix-matching exemption logic."""

    @pytest.mark.parametrize("path", EXEMPT_PATHS)
    def test_health_and_probe_paths_are_exempt(self, path: str):
        assert _is_rate_limit_exempt(path) is True

    def test_livez_is_exempt(self):
        # Explicit regression assertion for the crashloop fix.
        assert _is_rate_limit_exempt("/livez") is True

    @pytest.mark.parametrize(
        "path",
        ["/v1/memories", "/v1/recall", "/", "/healthcheck-but-not-health", "/livezz"],
    )
    def test_normal_api_paths_are_not_exempt(self, path: str):
        assert _is_rate_limit_exempt(path) is False


# ---------------------------------------------------------------------------
# End-to-end: middleware over a FastAPI app with a deny-all limiter
# ---------------------------------------------------------------------------


class _DenyAllRateLimitService(RateLimitService):
    """Rate limit service whose check always trips (allowed=False)."""

    async def check_rate_limit(self, key, limit=0, window_seconds=0):  # noqa: D102
        return RateLimitResult(allowed=False, limit=10, remaining=0, reset_at=time.time() + 60)

    async def get_usage(self, key):  # noqa: D102
        return (10, 10)


def _build_client(monkeypatch) -> TestClient:
    """FastAPI app wired with RateLimitMiddleware + a deny-all limiter.

    The middleware resolves its service via the module-level ``get_extension``;
    we monkeypatch that to return the deny-all service so any non-exempt path
    receives a 429.
    """
    monkeypatch.setattr(
        rl, "get_extension", lambda *_a, **_k: _DenyAllRateLimitService(), raising=True
    )

    app = FastAPI()

    @app.get("/health/ready")
    async def _ready():
        return {"status": "ok"}

    @app.get("/livez")
    async def _livez():
        return {"status": "ok"}

    @app.get("/healthz")
    async def _healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def _metrics():
        return "ok"

    @app.get("/v1/health/dependencies")
    async def _deps():
        return {"status": "healthy"}

    @app.get("/v1/memories")
    async def _memories():
        return {"items": []}

    # A bare Variables is enough for get_logger; the rate-limit service is
    # supplied via the patched get_extension above.
    app.add_middleware(RateLimitMiddleware, v=Variables())
    return TestClient(app)


class TestMiddlewareExemptBypass:
    """The deny-all limiter trips normal paths but never the exempt ones."""

    @pytest.mark.parametrize(
        "path",
        ["/health/ready", "/livez", "/healthz", "/metrics", "/v1/health/dependencies"],
    )
    def test_exempt_paths_bypass_limiter(self, monkeypatch, path: str):
        client = _build_client(monkeypatch)
        resp = client.get(path)
        assert resp.status_code != 429, f"{path} was throttled but must be exempt"
        assert resp.status_code == 200

    def test_livez_bypasses_limiter(self, monkeypatch):
        # Explicit regression: /livez must return 200, not 429.
        client = _build_client(monkeypatch)
        resp = client.get("/livez")
        assert resp.status_code == 200

    def test_non_exempt_path_is_throttled(self, monkeypatch):
        # Sanity check that the deny-all limiter is actually wired and trips.
        client = _build_client(monkeypatch)
        resp = client.get("/v1/memories")
        assert resp.status_code == 429
