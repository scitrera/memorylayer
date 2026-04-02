"""OpenTelemetry SDK initialization — configures TracerProvider and exporter."""
import logging
from typing import Iterable

from scitrera_app_framework import Plugin, Variables, ext_parse_bool

# Config constants
MEMORYLAYER_OTEL_ENABLED = 'MEMORYLAYER_OTEL_ENABLED'
MEMORYLAYER_OTEL_EXPORTER = 'MEMORYLAYER_OTEL_EXPORTER'  # 'otlp', 'console', 'none'
MEMORYLAYER_OTEL_ENDPOINT = 'MEMORYLAYER_OTEL_ENDPOINT'  # e.g., 'http://localhost:4317'
MEMORYLAYER_OTEL_SERVICE_NAME = 'MEMORYLAYER_OTEL_SERVICE_NAME'

DEFAULT_MEMORYLAYER_OTEL_SERVICE_NAME = 'memorylayer'
DEFAULT_MEMORYLAYER_OTEL_ENDPOINT = 'http://localhost:4317'

EXT_OTEL_INIT = 'memorylayer-server-otel-init'

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    HAS_OTEL_SDK = True
except ImportError:
    HAS_OTEL_SDK = False

# Optional OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False


class OTelInitPlugin(Plugin):
    """Plugin that initializes the OpenTelemetry SDK TracerProvider.

    Must run before any middleware that uses ``trace.get_tracer()``.  Registers
    only when ``opentelemetry-sdk`` is installed *and*
    ``MEMORYLAYER_OTEL_ENABLED`` is set to a truthy value.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_OTEL_INIT

    def is_enabled(self, v: Variables) -> bool:
        if not HAS_OTEL_SDK:
            return False
        return v.environ(MEMORYLAYER_OTEL_ENABLED, default=False, type_fn=ext_parse_bool)

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return ()

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        service_name = v.environ(
            MEMORYLAYER_OTEL_SERVICE_NAME,
            default=DEFAULT_MEMORYLAYER_OTEL_SERVICE_NAME,
        )
        exporter_type = v.environ(MEMORYLAYER_OTEL_EXPORTER, default='none').lower()

        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        if exporter_type == 'console':
            exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTel initialized: exporter=%s", exporter_type)
        elif exporter_type == 'otlp':
            if not HAS_OTLP:
                logger.warning(
                    "OTel exporter=otlp requested but opentelemetry-exporter-otlp-proto-grpc "
                    "is not installed; spans will not be exported"
                )
            else:
                endpoint = v.environ(
                    MEMORYLAYER_OTEL_ENDPOINT,
                    default=DEFAULT_MEMORYLAYER_OTEL_ENDPOINT,
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info("OTel initialized: exporter=%s, endpoint=%s", exporter_type, endpoint)
        else:
            logger.info("OTel initialized: exporter=%s (spans stay in-process)", exporter_type)

        trace.set_tracer_provider(provider)
        return provider

    def shutdown(self, v: Variables, logger: logging.Logger, value: object | None) -> None:
        if value is not None:
            value.shutdown()
            logger.info("OTel TracerProvider shut down")
