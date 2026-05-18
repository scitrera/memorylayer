"""Transport protocol — minimal contract shared by HTTP and Aether transports."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class TransportResponse(Protocol):
    """Subset of ``httpx.Response`` the SDK's request layer relies on."""

    status_code: int

    @property
    def content(self) -> bytes: ...
    @property
    def text(self) -> str: ...
    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...


@runtime_checkable
class Transport(Protocol):
    """Async transport that issues a single MemoryLayer-API request.

    Implementations:
    - ``HttpTransport`` wraps a long-lived ``httpx.AsyncClient``.
    - ``AetherTransport`` wraps a pre-existing Aether SDK client and routes
      via ``proxy_http_async`` against the MemoryLayer service topic.

    The ``json`` and ``content`` parameters are mutually exclusive: ``json``
    is JSON-encoded and sent with ``content-type: application/json``;
    ``content`` is sent as-is (str → utf-8) and the caller is responsible
    for setting any required ``content-type`` via ``headers``.
    """

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse: ...

    async def aclose(self) -> None: ...
