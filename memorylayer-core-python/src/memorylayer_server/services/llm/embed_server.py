"""LLM provider that delegates to a ``memorylayer-embed-server`` peer.

Uses the existing :class:`EmbedServerClient` so the same HTTP / Aether
transport plumbing that powers embedding traffic carries chat
completions too. Each profile can optionally override the embed-server
URL, transport, Aether target, and timeout — so a deployment can fan
multiple LLM profiles out to multiple embed-server peers (one HTTP,
one Aether, mixed across regions, etc.).

The provider serializes canonical ``LLMRequest`` to OpenAI-shape JSON,
forwards it to ``/v1/chat/completions`` via the client, and
re-materializes the response into ``LLMResponse`` / ``LLMStreamChunk``.
"""
from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator
from logging import Logger
from typing import Any, Optional

from scitrera_app_framework import Variables, get_extension, get_logger

from ...config import (
    DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
    DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
    DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT,
    DEFAULT_MEMORYLAYER_EMBED_TRANSPORT,
)
from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from .._constants import EXT_AETHER_SERVICE_CONNECTION, EXT_EMBED_SERVER_CLIENT
from ..document.embed_client import EmbedServerClient, TRANSPORT_AETHER, TRANSPORT_HTTP
from .base import LLMProvider


def _role_str(role) -> str:
    return role.value if isinstance(role, LLMRole) else str(role)


def _message_to_openai_dict(msg: LLMMessage) -> dict:
    """Mirror of ``services/llm/openai._message_to_openai_dict``.

    Kept inline to avoid an import cycle when ``openai.py`` later wants
    to import embed_server.py for any reason (and so this provider is
    self-contained — no openai-SDK dependency is implied).
    """
    role = _role_str(msg.role)
    out: dict = {"role": role}
    if msg.tool_calls is not None:
        out["tool_calls"] = msg.tool_calls
        out["content"] = msg.content if msg.content else None
    else:
        out["content"] = msg.content or ""
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        out["name"] = msg.name
    return out


def _build_chat_payload(request: LLMRequest, resolved_model: str, *, stream: bool,
                        max_tokens: int | None, temperature: float | None) -> dict:
    """Assemble the OpenAI-shape POST body for ``/v1/chat/completions``."""
    payload: dict = {
        "model": resolved_model,
        "messages": [_message_to_openai_dict(m) for m in request.messages],
    }
    if stream:
        payload["stream"] = True
    if request.stop is not None:
        payload["stop"] = request.stop
    effective_max = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else max_tokens
    )
    if effective_max is not None:
        payload["max_completion_tokens"] = effective_max
    if temperature is not None:
        payload["temperature"] = temperature
    if request.tools is not None:
        payload["tools"] = request.tools
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.extra_body is not None:
        # Merge extras into the body so callers can override or add
        # vendor-specific fields without modifying the canonical schema.
        payload.update(request.extra_body)
    return payload


def _materialize_response(data: dict, fallback_model: str) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    tool_calls = message.get("tool_calls") or None
    return LLMResponse(
        content=message.get("content") or "",
        model=data.get("model") or fallback_model,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
        finish_reason=choice.get("finish_reason") or "stop",
        tool_calls=tool_calls,
        reasoning_content=message.get("reasoning_content"),
    )


async def _parse_sse_stream(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[LLMStreamChunk]:
    """Convert an SSE byte stream into ``LLMStreamChunk`` events.

    The upstream embed-server emits OpenAI-format ``data: {...}\\n\\n``
    chunks plus a terminal ``data: [DONE]\\n\\n``. We accumulate raw bytes
    across the iterator, split on the SSE record separator (``\\n\\n``),
    and yield one chunk per record.
    """
    buffer = b""
    final_finish_reason: str | None = None
    async for raw in byte_iter:
        buffer += raw
        while b"\n\n" in buffer:
            record, buffer = buffer.split(b"\n\n", 1)
            for chunk in _yield_records_from_sse_record(record):
                yield chunk

    # Flush trailing partial (no terminator) — rare but possible if the
    # upstream closes without a final blank line.
    if buffer.strip():
        for chunk in _yield_records_from_sse_record(buffer):
            yield chunk

    # Emit a final marker if the stream didn't include one.
    yield LLMStreamChunk(
        content="",
        is_final=True,
        finish_reason=final_finish_reason or "stop",
    )


def _yield_records_from_sse_record(record: bytes):
    """Parse one SSE record (one or more ``data:`` lines) into LLMStreamChunks."""
    for line in record.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            # Don't emit; the parent generator yields the terminal chunk.
            return
        try:
            data = _json.loads(payload)
        except _json.JSONDecodeError:
            continue

        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        content_delta = delta.get("content") or ""
        tool_calls_delta = delta.get("tool_calls") or None
        reasoning_delta = delta.get("reasoning_content") or None

        if content_delta:
            yield LLMStreamChunk(content=content_delta, is_final=False)
        if tool_calls_delta:
            yield LLMStreamChunk(
                content="",
                is_final=False,
                tool_calls_delta=tool_calls_delta,
            )
        if reasoning_delta:
            yield LLMStreamChunk(
                content="",
                is_final=False,
                reasoning_content_delta=reasoning_delta,
            )
        if finish_reason:
            yield LLMStreamChunk(
                content="",
                is_final=True,
                finish_reason=finish_reason,
            )


class EmbedServerLLMProvider(LLMProvider):
    """LLM provider that proxies to a ``memorylayer-embed-server`` peer.

    By default uses the singleton :class:`EmbedServerClient` registered via
    ``EXT_EMBED_SERVER_CLIENT``. When any of the per-profile overrides
    (URL, transport, Aether target, timeout) are set, the provider builds
    a dedicated client at construction time so multiple LLM profiles can
    fan out to multiple embed-server peers independently.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        embed_server_url: str | None = None,
        embed_server_transport: str | None = None,
        embed_server_aether_target: str | None = None,
        embed_server_timeout: float | None = None,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        v: Variables = None,
    ):
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._v = v
        self.logger = get_logger(v, name=self.__class__.__name__)

        # If any per-profile override is set, build a dedicated client now;
        # otherwise we lazily resolve the shared singleton via
        # ``EXT_EMBED_SERVER_CLIENT`` on first use.
        has_override = any(
            x is not None for x in (
                embed_server_url, embed_server_transport,
                embed_server_aether_target, embed_server_timeout,
            )
        )
        self._dedicated_client: EmbedServerClient | None = None
        if has_override:
            transport = (embed_server_transport or DEFAULT_MEMORYLAYER_EMBED_TRANSPORT).lower()
            aether_connection = None
            if transport == TRANSPORT_AETHER:
                aether_connection = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
                if aether_connection is None:
                    raise RuntimeError(
                        "EmbedServerLLMProvider with embed_server_transport=aether "
                        "requires EXT_AETHER_SERVICE_CONNECTION to be initialized."
                    )
            self._dedicated_client = EmbedServerClient(
                base_url=embed_server_url or "http://localhost:61051",
                timeout=float(
                    embed_server_timeout
                    if embed_server_timeout is not None
                    else DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT
                ),
                logger=self.logger,
                transport=transport,
                aether_connection=aether_connection,
                aether_target=embed_server_aether_target or DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
                aether_stream_idle_timeout_ms=DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
            )
            self._dedicated_needs_connect = (transport == TRANSPORT_HTTP)
        else:
            self._dedicated_needs_connect = False

        self.logger.info(
            "Initialized EmbedServerLLMProvider: model=%s, dedicated_client=%s, url=%s, transport=%s",
            model, self._dedicated_client is not None,
            embed_server_url, embed_server_transport,
        )

    async def _get_client(self) -> EmbedServerClient:
        """Return the configured client, lazily connecting the dedicated HTTP one."""
        if self._dedicated_client is not None:
            if self._dedicated_needs_connect:
                if getattr(self._dedicated_client, "_client", None) is None:
                    await self._dedicated_client.connect()
                self._dedicated_needs_connect = False
            return self._dedicated_client
        client = get_extension(EXT_EMBED_SERVER_CLIENT, self._v)
        if client is None:
            raise RuntimeError(
                "EmbedServerLLMProvider requires either per-profile overrides or "
                "the EXT_EMBED_SERVER_CLIENT extension to be registered."
            )
        # The shared client's plugin connects via ``async_ready``; nothing to do.
        return client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        max_tokens, temperature = self.resolve_params(request)
        resolved_model = request.model or self.model or "unknown"
        payload = _build_chat_payload(
            request, resolved_model, stream=False,
            max_tokens=max_tokens, temperature=temperature,
        )

        client = await self._get_client()
        self.logger.debug(
            "embed_server LLM request: model=%s, messages=%d",
            resolved_model, len(payload["messages"]),
        )
        data = await client.chat_completions(payload, stream=False)
        return _materialize_response(data, fallback_model=resolved_model)

    async def complete_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        max_tokens, temperature = self.resolve_params(request)
        resolved_model = request.model or self.model or "unknown"
        payload = _build_chat_payload(
            request, resolved_model, stream=True,
            max_tokens=max_tokens, temperature=temperature,
        )

        client = await self._get_client()
        byte_iter = await client.chat_completions(payload, stream=True)
        async for chunk in _parse_sse_stream(byte_iter):
            yield chunk

    @property
    def default_model(self) -> str | None:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True


__all__ = ["EmbedServerLLMProvider"]
