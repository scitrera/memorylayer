"""HTTP transport — wraps ``httpx.AsyncClient`` with the SDK's expected headers."""
from __future__ import annotations

from typing import Any, Mapping

import httpx


class HttpTransport:
    """Default SDK transport: one ``httpx.AsyncClient`` per SDK instance."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if session_id:
            headers["X-Session-ID"] = session_id
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        """Underlying ``httpx.AsyncClient``.

        Exposed so the small set of bespoke SDK paths that need raw httpx
        (file uploads, NDJSON streaming) can keep working unchanged.  Aether
        transport intentionally does NOT expose an equivalent — those bespoke
        paths raise ``NotImplementedError`` when running on Aether transport.
        """
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {"params": params, "headers": headers}
        if json is not None:
            kwargs["json"] = json
        if content is not None:
            kwargs["content"] = content
        return await self._client.request(method, path, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_session(self, session_id: str) -> None:
        """Update the ``X-Session-ID`` header on the underlying httpx client."""
        self._client.headers["X-Session-ID"] = session_id

    def clear_session(self) -> None:
        if "X-Session-ID" in self._client.headers:
            del self._client.headers["X-Session-ID"]
