"""Unit tests for TenantTagMiddleware.

Verifies that X-Auth-Tenant-ID and OBO headers are attached as span
attributes on the active OpenTelemetry span.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from memorylayer_server.middleware.tenant_tag import TenantTagMiddleware


# OTel's global TracerProvider can only be set once per process.  Use a single
# module-scoped provider + exporter and clear spans between tests.
@pytest.fixture(scope="module")
def otel_exporter():
    """Set up a module-scoped in-memory OTel exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.shutdown()


@pytest.fixture()
def exporter(otel_exporter):
    """Per-test helper: clear spans before each test and return the exporter."""
    otel_exporter.clear()
    return otel_exporter


def _make_app():
    """Minimal FastAPI app with TenantTagMiddleware and a test route."""
    app = FastAPI()
    app.add_middleware(TenantTagMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


class TestTenantTagMiddleware:
    """Tenant + OBO headers are attached to the active span."""

    def test_tenant_header_sets_span_attribute(self, exporter):
        """X-Auth-Tenant-ID → scitrera.tenant on the active span."""
        app = _make_app()
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-root"):
            client = TestClient(app, raise_server_exceptions=True)
            client.get("/ping", headers={"X-Auth-Tenant-ID": "alice"})

        spans = exporter.get_finished_spans()
        root = next((s for s in spans if s.name == "test-root"), None)
        assert root is not None, f"root span not found; spans={[s.name for s in spans]}"
        assert root.attributes.get("scitrera.tenant") == "alice"

    def test_obo_headers_set_span_attributes(self, exporter):
        """All five OBO-related headers are surfaced as span attributes."""
        app = _make_app()
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-obo"):
            client = TestClient(app, raise_server_exceptions=True)
            client.get(
                "/ping",
                headers={
                    "X-Auth-Tenant-ID": "botwinick",
                    "X-Aether-Grant-ID": "g_abc123",
                    "X-Aether-Authority-Mode": "delegate",
                    "X-Aether-Subject-Type": "user",
                    "X-Aether-Subject-ID": "u_xyz",
                },
            )

        spans = exporter.get_finished_spans()
        root = next((s for s in spans if s.name == "test-obo"), None)
        assert root is not None, f"root span not found; spans={[s.name for s in spans]}"
        attrs = root.attributes
        assert attrs.get("scitrera.tenant") == "botwinick"
        assert attrs.get("scitrera.obo.grant_id") == "g_abc123"
        assert attrs.get("scitrera.obo.authority_mode") == "delegate"
        assert attrs.get("scitrera.obo.subject_type") == "user"
        assert attrs.get("scitrera.obo.subject_id") == "u_xyz"

    def test_missing_headers_not_set(self, exporter):
        """Missing headers do not produce empty/None span attributes."""
        app = _make_app()
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-empty"):
            client = TestClient(app, raise_server_exceptions=True)
            client.get("/ping")  # no headers

        spans = exporter.get_finished_spans()
        root = next((s for s in spans if s.name == "test-empty"), None)
        assert root is not None, f"root span not found; spans={[s.name for s in spans]}"
        attrs = root.attributes
        assert "scitrera.tenant" not in attrs
        assert "scitrera.obo.grant_id" not in attrs
