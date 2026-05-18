"""Google GenAI (Gemini) LLM provider."""

import json as _json
from collections.abc import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from .base import LLMProvider

DEFAULT_LLM_GOOGLE_MODEL = "gemini-3-flash-preview"

# Google finish_reason -> our finish_reason
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}

# OpenAI o-series reasoning_effort → Google "thinking budget" tokens.
_REASONING_EFFORT_TO_BUDGET = {
    "minimal": 256,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}


def _role_str(role) -> str:
    return role.value if isinstance(role, LLMRole) else str(role)


class GoogleLLMProvider(LLMProvider):
    """Google GenAI (Gemini) LLM provider.

    Uses the google-genai SDK for completions and streaming.

    Note on tools / structured output: Google's SDK uses native ``Tool`` /
    ``FunctionDeclaration`` types and is not OpenAI-compatible at the
    SDK layer. The provider forwards :attr:`LLMRequest.tools` to the
    SDK as-is; callers must construct Google-shape tools (or merge them
    in via ``extra_body``).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_GOOGLE_MODEL,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        v: Variables = None,
    ):
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._client = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized GoogleLLMProvider: model=%s", model)

    def _get_client(self):
        """Lazy-load Google GenAI client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        return self._client

    @staticmethod
    def _extract_messages(request: LLMRequest):
        """Extract system instruction and format messages for Google GenAI.

        Google GenAI uses 'model' role instead of 'assistant', and takes
        system_instruction as a config parameter.

        Returns:
            Tuple of (system_text or None, list of (role, content) tuples)
        """
        system_text = None
        messages = []
        for msg in request.messages:
            role_str = _role_str(msg.role)
            if role_str == "system":
                if system_text is None:
                    system_text = msg.content
                else:
                    system_text += "\n" + msg.content
            else:
                # Google uses "model" instead of "assistant".
                role = "model" if role_str == "assistant" else role_str
                messages.append((role, msg.content))
        return system_text, messages

    @staticmethod
    def _build_request(
        system_text,
        messages,
        request: LLMRequest,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """Build Google GenAI SDK types from extracted messages.

        Requires google-genai to be installed. Called only at API call time.
        """
        from google.genai import types

        contents = [
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=content)],
            )
            for role, content in messages
        ]

        config_kwargs: dict = {}
        if system_text is not None:
            config_kwargs["system_instruction"] = system_text
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if request.stop:
            config_kwargs["stop_sequences"] = request.stop
        if request.tools is not None:
            # Caller is responsible for providing Google-shape Tool
            # objects (or dicts the SDK will coerce).
            config_kwargs["tools"] = request.tools
        if request.response_format is not None:
            # Map OpenAI-shape {"type": "json_object"} / {"type": "json_schema", ...}
            # to Google's response_mime_type / response_schema fields.
            rf = request.response_format
            if isinstance(rf, dict):
                if rf.get("type") == "json_object":
                    config_kwargs["response_mime_type"] = "application/json"
                elif rf.get("type") == "json_schema":
                    config_kwargs["response_mime_type"] = "application/json"
                    schema = rf.get("json_schema", {}).get("schema")
                    if schema is not None:
                        config_kwargs["response_schema"] = schema
        if request.reasoning_effort is not None:
            budget = _REASONING_EFFORT_TO_BUDGET.get(request.reasoning_effort, 2048)
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=budget,
                )
            except (AttributeError, TypeError):
                # Older google-genai versions don't expose ThinkingConfig;
                # silently skip rather than break the call.
                pass
        if request.extra_body is not None:
            config_kwargs.update(request.extra_body)

        config = types.GenerateContentConfig(**config_kwargs)
        return contents, config

    @staticmethod
    def _map_finish_reason(finish_reason) -> str:
        """Map Google GenAI finish reason to our standard finish_reason."""
        if finish_reason is None:
            return "stop"
        reason_str = str(finish_reason)
        return _FINISH_REASON_MAP.get(reason_str, "stop")

    @staticmethod
    def _extract_tool_calls(response) -> list[dict] | None:
        """Pull ``function_call`` parts out of a Gemini response → OpenAI-shape tool_calls."""
        out: list[dict] = []
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                name = getattr(fc, "name", "") or ""
                args = getattr(fc, "args", None) or {}
                # Gemini sometimes hands back a Mapping-like object.
                try:
                    args_dict = dict(args)
                except (TypeError, ValueError):
                    args_dict = {}
                out.append({
                    "id": getattr(fc, "id", None) or f"call_{len(out)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _json.dumps(args_dict),
                    },
                })
        return out or None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Google GenAI API."""
        client = self._get_client()

        system_text, messages = self._extract_messages(request)
        max_tokens, temperature = self.resolve_params(request)
        effective_max = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else max_tokens
        )
        contents, config = self._build_request(
            system_text,
            messages,
            request,
            max_tokens=effective_max,
            temperature=temperature,
        )
        model = request.model or self.model

        self.logger.debug("LLM request: model=%s, contents=%d", model, len(contents))

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        content = response.text or ""

        # Extract usage metadata
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        total_tokens = usage.total_token_count if usage else (prompt_tokens + completion_tokens)

        # Extract finish reason from first candidate
        finish_reason = "stop"
        if response.candidates:
            finish_reason = self._map_finish_reason(response.candidates[0].finish_reason)

        tool_calls = self._extract_tool_calls(response)
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    async def complete_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using Google GenAI API.

        Text deltas stream as they arrive. Tool calls are aggregated and
        emitted on the terminal chunk (Gemini's streaming surface is
        primarily text-oriented).
        """
        client = self._get_client()

        system_text, messages = self._extract_messages(request)
        max_tokens, temperature = self.resolve_params(request)
        effective_max = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else max_tokens
        )
        contents, config = self._build_request(
            system_text,
            messages,
            request,
            max_tokens=effective_max,
            temperature=temperature,
        )
        model = request.model or self.model

        tool_call_accum: list[dict] = []

        async for chunk in await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            text = chunk.text or ""
            if text:
                yield LLMStreamChunk(
                    content=text,
                    is_final=False,
                )
            chunk_tool_calls = self._extract_tool_calls(chunk)
            if chunk_tool_calls:
                tool_call_accum.extend(chunk_tool_calls)

        if tool_call_accum:
            yield LLMStreamChunk(
                content="",
                is_final=False,
                tool_calls_delta=tool_call_accum,
            )

        # Final chunk to signal completion.
        yield LLMStreamChunk(
            content="",
            is_final=True,
            finish_reason="tool_calls" if tool_call_accum else "stop",
        )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
