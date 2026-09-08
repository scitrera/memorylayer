"""Prometheus ``/metrics`` route for the embed-server.

Mirrors the upstream ``memorylayer_server.services.metrics.routes`` shape
but registers against ``memorylayer_embed_server.api.EXT_MULTI_API_ROUTERS``
so the embed-server's FastAPI app picks it up.

Gated on ``MEMORYLAYER_METRICS_SERVICE=prometheus`` — when the active
backend is the no-op service (default), this plugin is not registered
and the route never appears.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import Response
from memorylayer_server.config import MEMORYLAYER_METRICS_SERVICE
from scitrera_app_framework.api import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    include_in_schema=False,
    response_class=Response,
)
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in the standard text exposition format.

    An OpenTelemetry Collector configured with a Prometheus receiver
    can scrape this endpoint and forward to any OTLP backend, giving
    full OTel compatibility without a separate exporter library.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except ImportError as exc:
        raise RuntimeError(
            "prometheus_client is required to serve /metrics. Install with: pip install memorylayer-embed-server[observability]"
        ) from exc

    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


class PrometheusMetricsRoutePlugin(Plugin):
    """Register ``/metrics`` only when the active metrics backend is Prometheus."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        # MUST be False for a multi-extension router plugin: an enabled plugin
        # becomes the single-extension winner for EXT_MULTI_API_ROUTERS and
        # shadows every sibling router (embeddings, chat, score, transcription)
        # via get_extension(). Gate enablement in is_multi_extension() instead.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # Contribute the /metrics router only when the active metrics backend is
        # Prometheus; otherwise the framework omits this plugin entirely.
        return v.environ(MEMORYLAYER_METRICS_SERVICE, default="noop") == "prometheus"

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        # Enablement is already decided by is_multi_extension(); just contribute
        # the router.
        logger.info("Registering Prometheus /metrics route on embed-server")
        return router
