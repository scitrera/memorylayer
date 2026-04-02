"""Rate limiting middleware for MemoryLayer FastAPI server."""
import time
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from scitrera_app_framework import Variables, get_extension, get_logger
from scitrera_app_framework.api import Plugin
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..lifecycle.fastapi import EXT_FASTAPI_SERVER
from ..services._constants import EXT_RATE_LIMIT_SERVICE
from ..services.rate_limit.noop import NoopRateLimitService

# Health-check paths that bypass rate limiting
_HEALTH_PATHS = frozenset({"/health", "/healthz"})

EXT_RATE_LIMIT_MIDDLEWARE = 'memorylayer-server-fastapi-middleware-rate-limit'


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces per-key rate limits.

    The rate limit key is derived from the ``X-Auth-User-ID`` request header.
    If that header is absent the client IP address is used as a fallback.

    Health check endpoints (``/health``, ``/healthz``) are always exempt.

    Response headers added on every non-exempt request:
        ``X-RateLimit-Limit``      -- maximum requests per window
        ``X-RateLimit-Remaining``  -- requests remaining in current window
        ``X-RateLimit-Reset``      -- unix timestamp when the window resets

    When the limit is exceeded a ``429`` response is returned with::

        {"detail": "Rate limit exceeded", "retry_after": <seconds>}
    """

    def __init__(self, app: ASGIApp, v: Variables) -> None:
        super().__init__(app)
        self._v = v
        self._logger = get_logger(v, name=self.__class__.__name__)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check paths
        if request.url.path in _HEALTH_PATHS:
            return await call_next(request)

        # Resolve rate limit service lazily from Variables
        try:
            svc = get_extension(EXT_RATE_LIMIT_SERVICE, self._v)
        except Exception:
            self._logger.debug("Rate limit service not available, passing through")
            return await call_next(request)

        # Noop service: skip all processing for efficiency
        if isinstance(svc, NoopRateLimitService):
            return await call_next(request)

        # Derive key: prefer authenticated user id, fall back to client IP
        user_id = request.headers.get("X-Auth-User-ID")
        if user_id:
            key = f"user:{user_id}"
        else:
            client = request.client
            ip = client.host if client else "unknown"
            key = f"ip:{ip}"

        try:
            result = await svc.check_rate_limit(key)
        except Exception:
            self._logger.error(
                "Rate limit check failed for key %s, allowing request",
                key,
                exc_info=True,
            )
            return await call_next(request)

        # Build rate limit response headers
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at)),
        }

        if not result.allowed:
            retry_after = max(0, int(result.reset_at - time.time()))
            self._logger.debug(
                "Rate limit exceeded for key %s (retry_after=%s)",
                key,
                retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                headers=headers,
            )

        response = await call_next(request)

        # Attach rate limit headers to the actual response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        return response


class RateLimitMiddlewarePlugin(Plugin):
    """Plugin that adds :class:`RateLimitMiddleware` to the FastAPI app."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RATE_LIMIT_MIDDLEWARE

    def initialize(self, v: Variables, logger) -> None:
        app = self.get_extension(EXT_FASTAPI_SERVER, v)
        app.add_middleware(RateLimitMiddleware, v=v)
        logger.info("RateLimitMiddleware registered")
        return None

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_FASTAPI_SERVER,)
