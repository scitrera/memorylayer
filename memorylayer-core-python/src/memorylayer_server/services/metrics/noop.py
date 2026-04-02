"""No-op metrics service - discards all observations (OSS default)."""

from logging import Logger

from scitrera_app_framework.api import Variables

from .base import MetricsService, MetricsServicePluginBase


class NoopMetricsService(MetricsService):
    """No-op metrics service that discards all observations."""

    def counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        pass

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        pass

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        pass


class NoopMetricsServicePlugin(MetricsServicePluginBase):
    """Plugin for no-op metrics service."""

    PROVIDER_NAME = "noop"

    def initialize(self, v: Variables, logger: Logger) -> NoopMetricsService | None:
        return NoopMetricsService()
