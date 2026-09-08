"""OpenAI-compatible LLM provider."""

from collections.abc import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from ..api_key_store import resolve_api_key_ref
from .attribution import task_attribution_headers
from .base import LLMProvider

DEFAULT_LLM_OPENAI_MODEL = "gpt-5-nano"


def _role_str(role) -> str:
    return role.value if isinstance(role, LLMRole) else str(role)


def _message_to_openai_dict(msg: LLMMessage) -> dict:
    """Serialize ``LLMMessage`` to an OpenAI chat-completions message dict.

    Handles assistant messages with ``tool_calls`` (content may be empty)
    and tool-result messages (role=tool with ``tool_call_id``).
    """
    role = _role_str(msg.role)
    out: dict = {"role": role}
    if msg.tool_calls is not None:
        out["tool_calls"] = msg.tool_calls
        # OpenAI accepts content=null on assistant messages that only
        # request tool calls; sending "" is also accepted.
        out["content"] = msg.content if msg.content else None
    elif getattr(msg, "images", None):
        # Multimodal (vision) message: serialize as the OpenAI content-block
        # list — a text block followed by one ``image_url`` block per image.
        # Each image is either a raw base64 string (assumed PNG) or an already
        # formed ``data:`` URI. Used by the image-backed fact-decomposition
        # path (scanned/OCR-free document pages) against vision-capable models.
        blocks: list[dict] = [{"type": "text", "text": msg.content or ""}]
        for img in msg.images:
            uri = img if img.startswith("data:") else f"data:image/png;base64,{img}"
            blocks.append({"type": "image_url", "image_url": {"url": uri}})
        out["content"] = blocks
    else:
        out["content"] = msg.content or ""
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        out["name"] = msg.name
    return out


def _openai_tool_call_to_dict(tc) -> dict:
    """Convert an OpenAI SDK ChatCompletionMessageToolCall to a plain dict."""
    return {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def _openai_tool_call_delta_to_dict(tc) -> dict:
    """Serialize a streaming tool-call delta. Any field may be ``None`` mid-stream."""
    out: dict = {"index": tc.index}
    if getattr(tc, "id", None) is not None:
        out["id"] = tc.id
    if getattr(tc, "type", None) is not None:
        out["type"] = tc.type
    fn = getattr(tc, "function", None)
    if fn is not None:
        fn_out: dict = {}
        if getattr(fn, "name", None) is not None:
            fn_out["name"] = fn.name
        if getattr(fn, "arguments", None) is not None:
            fn_out["arguments"] = fn.arguments
        if fn_out:
            out["function"] = fn_out
    return out


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with OpenAI API, Azure OpenAI, Ollama, vLLM, and any
    OpenAI-compatible endpoint by configuring the base URL.
    """

    # Default key name resolved from the API key store when no literal
    # ``api_key`` is given and no per-profile override is configured.
    DEFAULT_API_KEY_NAME = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = None,
        model: str = DEFAULT_LLM_OPENAI_MODEL,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        api_key_name: str | None = None,
        api_key_store=None,
        default_extra_body: dict | None = None,
        default_reasoning_effort: str | None = None,
        default_headers: dict[str, str] | None = None,
        stamp_identity: bool = False,
        v: Variables = None,
    ):
        # ``api_key`` (literal) wins and skips the store; otherwise resolve the
        # key by name through the store on each use (picks up rotations).
        self.api_key, self._key_ref = resolve_api_key_ref(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            default_name=self.DEFAULT_API_KEY_NAME,
        )
        self.base_url = base_url
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        # Provider-level extra_body merged into every request (per-request
        # extra_body wins on key conflicts). Used e.g. to send
        # {"enable_thinking": false} to a reasoning model for the whole profile.
        self.default_extra_body = default_extra_body or None
        # Provider-level reasoning_effort default (per-request wins). "none"
        # turns off thinking for reasoning models like Fireworks qwen3 (the
        # hosted endpoint rejects enable_thinking/chat_template_kwargs, but
        # honors reasoning_effort).
        self.default_reasoning_effort = default_reasoning_effort or None
        # Whether this endpoint may receive caller-asserted identity headers.
        # Decided by the registry from the host allowlist (see
        # attribution.host_allows_identity) because only the registry knows the
        # configured policy. Defaults to False so a provider constructed directly
        # — in a test, a script, or future code — cannot leak identity by omission.
        self.stamp_identity = bool(stamp_identity)
        # Static per-client HTTP headers (e.g. caller-asserted attribution:
        # X-Scitrera-Source / X-Scitrera-Tenant) installed on the AsyncOpenAI
        # client and sent on every request. Per-request headers (from
        # request.extra_headers / the ambient task attribution) are merged on
        # top at call time — see _build_kwargs.
        #
        # Held behind the same gate as the task header: passing headers without
        # permission to stamp is a caller error, so they are dropped rather than
        # quietly sent.
        self.default_headers = (default_headers or None) if self.stamp_identity else None
        self._client = None
        self._client_key = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized OpenAILLMProvider: base_url=%s, model=%s", base_url, model)

    def _build_client(self, api_key):
        """Construct an OpenAI async client for ``api_key``."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        client_kwargs: dict = {"api_key": api_key, "base_url": self.base_url}
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        return AsyncOpenAI(**client_kwargs)

    def _get_client(self):
        """Lazy-load OpenAI async client from the static key (no store)."""
        if self._client is None:
            self._client = self._build_client(self.api_key)
            self._client_key = self.api_key
        return self._client

    async def _ensure_client(self):
        """Return a client built with the currently-resolved key.

        Static key path returns the cached client. When a key store is bound,
        the key is resolved per call and the client is rebuilt only when the
        resolved key changes (supports rotation).
        """
        if self._key_ref is None:
            return self._get_client()
        key = await self._key_ref.resolve()
        if self._client is None or key != self._client_key:
            self._client = self._build_client(key)
            self._client_key = key
        return self._client

    def _build_kwargs(self, request: LLMRequest, *, stream: bool) -> dict:
        """Assemble kwargs for ``chat.completions.create()``."""
        messages = [_message_to_openai_dict(msg) for msg in request.messages]
        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        kwargs: dict = {"model": model, "messages": messages}
        if stream:
            kwargs["stream"] = True
        if request.stop is not None:
            kwargs["stop"] = request.stop
        effective_max = request.max_completion_tokens if request.max_completion_tokens is not None else max_tokens
        if effective_max is not None:
            kwargs["max_completion_tokens"] = effective_max
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.tools is not None:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        effective_reasoning = request.reasoning_effort if request.reasoning_effort is not None else self.default_reasoning_effort
        if effective_reasoning is not None:
            kwargs["reasoning_effort"] = effective_reasoning
        # Merge provider-level default extra_body with any per-request extra_body
        # (request keys win). Lets a profile disable thinking globally while a
        # specific call can still override.
        if self.default_extra_body is not None or request.extra_body is not None:
            kwargs["extra_body"] = {**(self.default_extra_body or {}), **(request.extra_body or {})}
        # Per-request attribution headers: the ambient task header
        # (X-Scitrera-Task-Id, set while a task handler runs) merged with any
        # explicit request.extra_headers (explicit request keys win). Passed as
        # ``extra_headers`` so the OpenAI SDK sends them on top of the client's
        # static default_headers (X-Scitrera-Source / X-Scitrera-Tenant). Only
        # set when non-empty so the wire shape is unchanged for plain calls.
        #
        # The task header is gated on the SAME allowlist decision as the static
        # ones: it is an internal identifier, and gating only the static set would
        # still hand every public provider a running trace of our task ids.
        # request.extra_headers is NOT gated — that is an explicit per-call choice
        # by the caller, not ambient identity this module asserts.
        ambient = task_attribution_headers() if self.stamp_identity else {}
        extra_headers = {**ambient, **(request.extra_headers or {})}
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        return kwargs

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        client = await self._ensure_client()
        kwargs = self._build_kwargs(request, stream=False)
        self.logger.debug(
            "LLM request: model=%s, messages=%d, tools=%s",
            kwargs["model"],
            len(kwargs["messages"]),
            (len(request.tools) if request.tools else 0),
        )

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        tool_calls = None
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = [_openai_tool_call_to_dict(tc) for tc in raw_tool_calls]

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
        )

    async def complete_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using OpenAI API."""
        client = await self._ensure_client()
        kwargs = self._build_kwargs(request, stream=True)

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
                yield LLMStreamChunk(
                    content=delta.content,
                    is_final=False,
                )

            raw_tc_deltas = getattr(delta, "tool_calls", None)
            if raw_tc_deltas:
                yield LLMStreamChunk(
                    content="",
                    is_final=False,
                    tool_calls_delta=[_openai_tool_call_delta_to_dict(tc) for tc in raw_tc_deltas],
                )

            if choice.finish_reason:
                yield LLMStreamChunk(
                    content="",
                    is_final=True,
                    finish_reason=choice.finish_reason,
                )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
