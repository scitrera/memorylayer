"""Prometheus /metrics endpoint - only active when metrics service is 'prometheus'."""
import logging

from fastapi import APIRouter
from fastapi.responses import Response
from scitrera_app_framework.api import Plugin, Variables

from ...api import EXT_MULTI_API_ROUTERS
from ...config import MEMORYLAYER_METRICS_SERVICE

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    include_in_schema=False,
    response_class=Response,
)
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in the standard text exposition format."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    except ImportError as exc:
        raise RuntimeError(
            "prometheus_client is required to serve /metrics. "
            "Install it with: pip install prometheus_client"
        ) from exc

    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


class PrometheusMetricsRoutePlugin(Plugin):
    """Plugin to register the /metrics route when Prometheus is the active provider."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        provider = v.environ(MEMORYLAYER_METRICS_SERVICE, default='noop')
        return provider == 'prometheus'

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        logger.info("Registering Prometheus /metrics route")
        return router
