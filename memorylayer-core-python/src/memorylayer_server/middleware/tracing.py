"""OpenTelemetry tracing middleware — active only when opentelemetry-api is installed."""
from typing import Iterable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from scitrera_app_framework import Variables, get_logger
from scitrera_app_framework.api import Plugin

from ..lifecycle.fastapi import EXT_FASTAPI_SERVER

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    from opentelemetry.propagate import extract, inject
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

EXT_TRACING_MIDDLEWARE = 'memorylayer-server-fastapi-middleware-tracing'


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that creates OpenTelemetry spans for each HTTP request."""

    def __init__(self, app: ASGIApp, tracer_name: str = "memorylayer") -> None:
        super().__init__(app)
        if HAS_OTEL:
            self._tracer = trace.get_tracer(tracer_name)
        else:
            self._tracer = None

    async def dispatch(self, request: Request, call_next):
        if not self._tracer:
            return await call_next(request)

        # Extract trace context from incoming headers (W3C Trace Context)
        ctx = extract(carrier=dict(request.headers))

        span_name = f"{request.method} {request.url.path}"

        with self._tracer.start_as_current_span(
            span_name,
            context=ctx,
            kind=trace.SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
            },
        ) as span:
            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 400:
                    span.set_status(StatusCode.ERROR)

                # Inject trace context into response headers for caller correlation
                response_headers = {}
                inject(carrier=response_headers)
                for key, value in response_headers.items():
                    response.headers[key] = value

                return response
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise


class TracingMiddlewarePlugin(Plugin):
    """Plugin that adds :class:`TracingMiddleware` to the FastAPI app.

    Registers only when the ``opentelemetry-api`` package is installed.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_TRACING_MIDDLEWARE

    def is_enabled(self, v: Variables) -> bool:
        return HAS_OTEL

    def initialize(self, v: Variables, logger) -> None:
        app = self.get_extension(EXT_FASTAPI_SERVER, v)
        app.add_middleware(TracingMiddleware)
        logger.info("TracingMiddleware registered")
        return None

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_FASTAPI_SERVER,)
