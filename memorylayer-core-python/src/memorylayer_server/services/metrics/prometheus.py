"""Prometheus metrics service - emits metrics in Prometheus exposition format.

Requires the optional ``prometheus_client`` package::

    pip install prometheus_client
"""
import threading
from logging import Logger
from typing import Optional

from scitrera_app_framework.api import Variables

from .base import MetricsService, MetricsServicePluginBase

# ---------------------------------------------------------------------------
# Standard MemoryLayer metric names
# ---------------------------------------------------------------------------
METRIC_RECALL_TOTAL = "memorylayer_recall_total"
METRIC_RECALL_LATENCY = "memorylayer_recall_latency_seconds"
METRIC_REMEMBER_TOTAL = "memorylayer_remember_total"
METRIC_REMEMBER_LATENCY = "memorylayer_remember_latency_seconds"
METRIC_SESSION_ACTIVE = "memorylayer_sessions_active"
METRIC_MEMORY_COUNT = "memorylayer_memories_total"
METRIC_REQUEST_TOTAL = "memorylayer_http_requests_total"
METRIC_REQUEST_LATENCY = "memorylayer_http_request_duration_seconds"


class PrometheusMetricsService(MetricsService):
    """Metrics service backed by ``prometheus_client``.

    Metric objects are created lazily on first use and cached by name.
    Label names are derived from the keys of the ``labels`` dict supplied
    on the first call for a given metric name; subsequent calls must pass
    the same label keys.

    ``prometheus_client`` handles its own internal locking, but the lazy
    creation cache is protected by a ``threading.Lock`` for safety.
    """

    def __init__(self) -> None:
        try:
            import prometheus_client as _pc  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "prometheus_client is required for PrometheusMetricsService. "
                "Install it with: pip install prometheus_client"
            ) from exc

        self._lock = threading.Lock()
        self._counters: dict = {}
        self._histograms: dict = {}
        self._gauges: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_names(labels: dict[str, str] | None) -> tuple[str, ...]:
        return tuple(labels.keys()) if labels else ()

    @staticmethod
    def _label_values(labels: dict[str, str] | None) -> tuple[str, ...]:
        return tuple(labels.values()) if labels else ()

    def _get_counter(self, name: str, label_names: tuple[str, ...]):
        import prometheus_client as pc
        if name not in self._counters:
            with self._lock:
                if name not in self._counters:
                    self._counters[name] = pc.Counter(name, name, list(label_names))
        return self._counters[name]

    def _get_histogram(self, name: str, label_names: tuple[str, ...]):
        import prometheus_client as pc
        if name not in self._histograms:
            with self._lock:
                if name not in self._histograms:
                    self._histograms[name] = pc.Histogram(name, name, list(label_names))
        return self._histograms[name]

    def _get_gauge(self, name: str, label_names: tuple[str, ...]):
        import prometheus_client as pc
        if name not in self._gauges:
            with self._lock:
                if name not in self._gauges:
                    self._gauges[name] = pc.Gauge(name, name, list(label_names))
        return self._gauges[name]

    # ------------------------------------------------------------------
    # MetricsService interface
    # ------------------------------------------------------------------

    def counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a Prometheus Counter."""
        label_names = self._label_names(labels)
        metric = self._get_counter(name, label_names)
        if label_names:
            metric.labels(*self._label_values(labels)).inc(value)
        else:
            metric.inc(value)

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe a value in a Prometheus Histogram."""
        label_names = self._label_names(labels)
        metric = self._get_histogram(name, label_names)
        if label_names:
            metric.labels(*self._label_values(labels)).observe(value)
        else:
            metric.observe(value)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a Prometheus Gauge."""
        label_names = self._label_names(labels)
        metric = self._get_gauge(name, label_names)
        if label_names:
            metric.labels(*self._label_values(labels)).set(value)
        else:
            metric.set(value)


class PrometheusMetricsServicePlugin(MetricsServicePluginBase):
    """Plugin for Prometheus metrics service."""
    PROVIDER_NAME = 'prometheus'

    def initialize(self, v: Variables, logger: Logger) -> Optional[PrometheusMetricsService]:
        logger.info("Initializing PrometheusMetricsService")
        return PrometheusMetricsService()
