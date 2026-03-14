"""Metrics Service - Pluggable metrics/observability interface."""
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_METRICS_SERVICE, DEFAULT_MEMORYLAYER_METRICS_SERVICE
from .._constants import EXT_METRICS_SERVICE

# Re-export for convenience
__all__ = ["MetricsService", "MetricsServicePluginBase", "EXT_METRICS_SERVICE"]


class MetricsService(ABC):
    """Abstract metrics service interface.

    Provides counter, histogram, and gauge primitives that can be implemented
    by different backends (no-op, Prometheus, StatsD, etc.).
    """

    @abstractmethod
    def counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment (default 1)
            labels: Optional label key/value pairs
        """
        ...

    @abstractmethod
    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional label key/value pairs
        """
        ...

    @abstractmethod
    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Current value
            labels: Optional label key/value pairs
        """
        ...

    @contextmanager
    def time(self, name: str, labels: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Context manager that records elapsed wall-clock time as a histogram.

        Works for both sync and async usage (uses time.perf_counter).

        Args:
            name: Histogram metric name
            labels: Optional label key/value pairs

        Example::

            with metrics.time("memorylayer_recall_latency_seconds"):
                result = await recall(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.histogram(name, elapsed, labels)


# noinspection PyAbstractClass
class MetricsServicePluginBase(Plugin):
    """Base plugin for metrics service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_METRICS_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_METRICS_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_METRICS_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        # Set a default value for MEMORYLAYER_METRICS_SERVICE; defaults are lower priority than .set(...) values
        v.set_default_value(MEMORYLAYER_METRICS_SERVICE, DEFAULT_MEMORYLAYER_METRICS_SERVICE)
