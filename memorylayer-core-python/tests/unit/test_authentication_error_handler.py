"""Unit tests for the global AuthenticationError → HTTP exception handler.

Regression guard: before the global handler was added to FastApiPlugin,
routes that did NOT have a per-endpoint ``except AuthenticationError`` block
would let the exception fall through to FastAPI's generic ``Exception``
handler and return HTTP 500.  The handler fixes this for all routes at once.

These tests use a *minimal* FastAPI app (no plugin stack, no service DI) that
replicates only the handler registration from ``FastApiPlugin.initialize``,
mirroring the pattern used in ``test_tenant_tag_middleware.py``.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from memorylayer_server.services.authentication.base import AuthenticationError


# ---------------------------------------------------------------------------
# Minimal app factory (mirrors the handler block in FastApiPlugin.initialize)
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Bare FastAPI app with only the AuthenticationError handler registered."""
    app = FastAPI()

    @app.exception_handler(AuthenticationError)
    async def _authentication_error_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @app.get("/raise-default")
    async def raise_default():
        """Route that raises AuthenticationError with the default 401 status."""
        raise AuthenticationError("nope")

    @app.get("/raise-custom")
    async def raise_custom():
        """Route that raises AuthenticationError with a custom status code."""
        raise AuthenticationError("forbidden", status_code=403)

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuthenticationErrorHandler:
    """Global AuthenticationError handler maps to the correct HTTP status."""

    def setup_method(self):
        self.client = TestClient(_make_app(), raise_server_exceptions=False)

    def test_default_status_is_401(self):
        """AuthenticationError with no explicit status_code → 401."""
        response = self.client.get("/raise-default")
        assert response.status_code == 401

    def test_default_body_contains_detail(self):
        """Response body is JSON with the error message under 'detail'."""
        response = self.client.get("/raise-default")
        assert response.json() == {"detail": "nope"}

    def test_custom_status_code_is_honored(self):
        """AuthenticationError with status_code=403 → 403, not 500 or 401."""
        response = self.client.get("/raise-custom")
        assert response.status_code == 403

    def test_custom_status_body_contains_detail(self):
        """Custom-status response carries the correct detail message."""
        response = self.client.get("/raise-custom")
        assert response.json() == {"detail": "forbidden"}
