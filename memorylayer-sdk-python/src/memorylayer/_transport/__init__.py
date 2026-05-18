"""Transport abstraction for the MemoryLayer SDK.

Phase 5 of the Aether convergence introduced a small Transport protocol so
the SDK can issue requests via either:

* ``HttpTransport`` — direct ``httpx.AsyncClient`` calls (default; the legacy
  behaviour OSS deployments rely on).
* ``AetherTransport`` — ``proxy_http_async`` calls through an existing
  ``scitrera_aether_client`` connection (an ``AsyncAgentClient`` or
  ``AsyncServiceClient``).  Lets cowork agents and enterprise integrations
  reuse one Aether connection for everything instead of opening a parallel
  HTTP path to MemoryLayer.

Both transports return objects matching the ``TransportResponse`` protocol
so the rest of the client code does not branch.
"""

from .aether import AetherTransport, AetherTransportResponse
from .base import Transport, TransportResponse
from .http import HttpTransport

__all__ = [
    "Transport",
    "TransportResponse",
    "HttpTransport",
    "AetherTransport",
    "AetherTransportResponse",
]
