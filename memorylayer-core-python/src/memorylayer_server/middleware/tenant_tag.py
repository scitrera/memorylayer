"""Tenant + OBO header span-attribute middleware.

Runs AFTER FastAPIInstrumentor's middleware so the request span is already
started and current.  Pulls well-known headers and attaches them as span
attributes so OTEL can filter by tenant and OBO context.

Header → attribute mapping:
    X-Auth-Tenant-ID          → scitrera.tenant
    X-Aether-Grant-ID         → scitrera.obo.grant_id
    X-Aether-Authority-Mode   → scitrera.obo.authority_mode
    X-Aether-Subject-Type     → scitrera.obo.subject_type
    X-Aether-Subject-ID       → scitrera.obo.subject_id

Header names match those sent by backend-future's MemoryLayerClient
(scitrera_app_server/tenant_api/ml_client.py lines 14-19).
"""

from collections.abc import Iterable

from fastapi import Request
from scitrera_app_framework import Variables
from scitrera_app_framework.api import Plugin
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..lifecycle.fastapi import EXT_FASTAPI_SERVER

try:
    from opentelemetry import trace as _otel_trace

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

EXT_TENANT_TAG_MIDDLEWARE = "memorylayer-server-fastapi-middleware-tenant-tag"

# (header-name, span-attribute-name) pairs
_HEADER_ATTR_MAP: tuple[tuple[str, str], ...] = (
    ("x-auth-tenant-id", "scitrera.tenant"),
    ("x-aether-grant-id", "scitrera.obo.grant_id"),
    ("x-aether-authority-mode", "scitrera.obo.authority_mode"),
    ("x-aether-subject-type", "scitrera.obo.subject_type"),
    ("x-aether-subject-id", "scitrera.obo.subject_id"),
)


class TenantTagMiddleware(BaseHTTPMiddleware):
    """Attach tenant + OBO headers as attributes on the active OTel span."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if HAS_OTEL:
            span = _otel_trace.get_current_span()
            if span.is_recording():
                for header, attr in _HEADER_ATTR_MAP:
                    value = request.headers.get(header)
                    if value:
                        span.set_attribute(attr, value)
        return await call_next(request)


class TenantTagMiddlewarePlugin(Plugin):
    """Plugin that adds :class:`TenantTagMiddleware` to the FastAPI app.

    Must run AFTER FastAPIInstrumentor so the request span is already active
    when TenantTagMiddleware executes.  Starlette/FastAPI processes middleware
    in LIFO order relative to add_middleware() calls, so we declare a dependency
    on the FastAPI extension point (which is where instrument_app is called) and
    let the plugin system ensure ordering.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TENANT_TAG_MIDDLEWARE

    def is_enabled(self, v: Variables) -> bool:
        return HAS_OTEL

    def initialize(self, v: Variables, logger) -> None:
        app = self.get_extension(EXT_FASTAPI_SERVER, v)
        app.add_middleware(TenantTagMiddleware)
        logger.info("TenantTagMiddleware registered")
        return None

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_FASTAPI_SERVER,)
