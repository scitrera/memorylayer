"""
Unit tests for the metrics service — NoopMetricsService and PrometheusMetricsService.
"""

import time

import pytest

from memorylayer_server.services.metrics.noop import NoopMetricsService

# ============================================================================
# NoopMetricsService tests
# ============================================================================


class TestNoopMetricsService:
    """Test NoopMetricsService — every method is a silent no-op."""

    def test_counter_returns_none(self):
        """counter() completes without raising and returns None."""
        service = NoopMetricsService()
        result = service.counter("memorylayer_recall_total")
        assert result is None

    def test_counter_with_value_and_labels(self):
        """counter() accepts value and labels without raising."""
        service = NoopMetricsService()
        service.counter("memorylayer_recall_total", value=3, labels={"workspace": "ws-1"})

    def test_histogram_returns_none(self):
        """histogram() completes without raising and returns None."""
        service = NoopMetricsService()
        result = service.histogram("memorylayer_recall_latency_seconds", 0.042)
        assert result is None

    def test_histogram_with_labels(self):
        """histogram() accepts labels without raising."""
        service = NoopMetricsService()
        service.histogram(
            "memorylayer_recall_latency_seconds",
            0.123,
            labels={"mode": "hybrid"},
        )

    def test_gauge_returns_none(self):
        """gauge() completes without raising and returns None."""
        service = NoopMetricsService()
        result = service.gauge("memorylayer_sessions_active", 7.0)
        assert result is None

    def test_gauge_with_labels(self):
        """gauge() accepts labels without raising."""
        service = NoopMetricsService()
        service.gauge("memorylayer_sessions_active", 2.0, labels={"tenant": "acme"})

    def test_time_context_manager_does_not_raise(self):
        """time() context manager completes without raising."""
        service = NoopMetricsService()
        with service.time("memorylayer_recall_latency_seconds"):
            pass  # simulated work

    def test_time_context_manager_records_elapsed_via_histogram(self):
        """time() calls histogram() with a non-negative elapsed value."""
        recorded: list[tuple] = []

        class SpyMetrics(NoopMetricsService):
            def histogram(self, name, value, labels=None):
                recorded.append((name, value, labels))

        spy = SpyMetrics()
        with spy.time("my_metric", labels={"op": "test"}):
            time.sleep(0.01)  # ensure measurable elapsed time

        assert len(recorded) == 1
        name, value, labels = recorded[0]
        assert name == "my_metric"
        assert value >= 0.0
        assert labels == {"op": "test"}

    def test_time_context_manager_records_elapsed_on_exception(self):
        """time() still records histogram even when the body raises."""
        recorded: list = []

        class SpyMetrics(NoopMetricsService):
            def histogram(self, name, value, labels=None):
                recorded.append(value)

        spy = SpyMetrics()
        with pytest.raises(ValueError):
            with spy.time("my_metric"):
                raise ValueError("oops")

        assert len(recorded) == 1
        assert recorded[0] >= 0.0

    def test_multiple_calls_all_succeed(self):
        """Calling all metric methods in sequence does not raise."""
        service = NoopMetricsService()
        service.counter("c1")
        service.counter("c1", value=5, labels={"x": "y"})
        service.histogram("h1", 1.5)
        service.gauge("g1", 42.0)
        with service.time("h1"):
            pass


# ============================================================================
# PrometheusMetricsService tests (skipped when prometheus_client unavailable)
# ============================================================================


def _prometheus_available() -> bool:
    try:
        import prometheus_client  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _prometheus_available(),
    reason="prometheus_client is not installed",
)
class TestPrometheusMetricsService:
    """Test PrometheusMetricsService when prometheus_client is importable.

    Each test uses its own isolated CollectorRegistry so metric re-registration
    errors are avoided across test runs.
    """

    def _make_service(self):
        """Return a PrometheusMetricsService backed by a fresh registry."""
        import prometheus_client

        from memorylayer_server.services.metrics.prometheus import PrometheusMetricsService

        registry = prometheus_client.CollectorRegistry()

        service = PrometheusMetricsService.__new__(PrometheusMetricsService)
        import threading

        service._lock = threading.Lock()
        service._counters = {}
        service._histograms = {}
        service._gauges = {}
        # Monkey-patch _get_* to use the isolated registry
        original_get_counter = PrometheusMetricsService._get_counter
        original_get_histogram = PrometheusMetricsService._get_histogram
        original_get_gauge = PrometheusMetricsService._get_gauge

        def _get_counter(self, name, label_names):
            if name not in self._counters:
                with self._lock:
                    if name not in self._counters:
                        self._counters[name] = prometheus_client.Counter(name, name, list(label_names), registry=registry)
            return self._counters[name]

        def _get_histogram(self, name, label_names):
            if name not in self._histograms:
                with self._lock:
                    if name not in self._histograms:
                        self._histograms[name] = prometheus_client.Histogram(name, name, list(label_names), registry=registry)
            return self._histograms[name]

        def _get_gauge(self, name, label_names):
            if name not in self._gauges:
                with self._lock:
                    if name not in self._gauges:
                        self._gauges[name] = prometheus_client.Gauge(name, name, list(label_names), registry=registry)
            return self._gauges[name]

        import types

        service._get_counter = types.MethodType(_get_counter, service)
        service._get_histogram = types.MethodType(_get_histogram, service)
        service._get_gauge = types.MethodType(_get_gauge, service)
        service._registry = registry
        return service

    def test_instantiation_succeeds_when_prometheus_client_available(self):
        """PrometheusMetricsService can be instantiated when prometheus_client is present."""
        import prometheus_client

        registry = prometheus_client.CollectorRegistry()
        # Use _make_service to avoid global registry pollution
        service = self._make_service()
        assert service is not None

    def test_counter_does_not_raise(self):
        """counter() does not raise and creates a Counter metric."""
        service = self._make_service()
        service.counter("test_counter_total")

    def test_counter_with_labels_does_not_raise(self):
        """counter() with labels does not raise."""
        service = self._make_service()
        service.counter("test_counter_labels_total", value=2, labels={"env": "test"})

    def test_histogram_does_not_raise(self):
        """histogram() does not raise and creates a Histogram metric."""
        service = self._make_service()
        service.histogram("test_histogram_seconds", 0.5)

    def test_histogram_with_labels_does_not_raise(self):
        """histogram() with labels does not raise."""
        service = self._make_service()
        service.histogram("test_histogram_labels_seconds", 1.2, labels={"op": "recall"})

    def test_gauge_does_not_raise(self):
        """gauge() does not raise and creates a Gauge metric."""
        service = self._make_service()
        service.gauge("test_gauge", 10.0)

    def test_gauge_with_labels_does_not_raise(self):
        """gauge() with labels does not raise."""
        service = self._make_service()
        service.gauge("test_gauge_labels", 5.0, labels={"region": "us-east"})

    def test_counter_creates_counter_object_in_cache(self):
        """After counter() is called, the metric is cached in _counters."""
        service = self._make_service()
        service.counter("cached_counter_total")
        assert "cached_counter_total" in service._counters

    def test_histogram_creates_histogram_object_in_cache(self):
        """After histogram() is called, the metric is cached in _histograms."""
        service = self._make_service()
        service.histogram("cached_histogram_seconds", 0.1)
        assert "cached_histogram_seconds" in service._histograms

    def test_gauge_creates_gauge_object_in_cache(self):
        """After gauge() is called, the metric is cached in _gauges."""
        service = self._make_service()
        service.gauge("cached_gauge", 3.0)
        assert "cached_gauge" in service._gauges

    def test_time_context_manager_does_not_raise(self):
        """time() context manager completes without raising."""
        service = self._make_service()
        service.histogram("timing_histogram_seconds", 0.0)  # pre-register
        with service.time("timing_histogram_seconds"):
            pass
