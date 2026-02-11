"""Google GenAI (Gemini) LLM provider."""
from typing import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import LLMProvider
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk

DEFAULT_LLM_GOOGLE_MODEL = 'gemini-3-flash-preview'

# Google finish_reason -> our finish_reason
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


class GoogleLLMProvider(LLMProvider):
    """Google GenAI (Gemini) LLM provider.

    Uses the google-genai SDK for completions and streaming.
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
        self.logger.info(
            "Initialized GoogleLLMProvider: model=%s", model
        )

    def _get_client(self):
        """Lazy-load Google GenAI client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )
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
            if msg.role == "system":
                if system_text is None:
                    system_text = msg.content
                else:
                    system_text += "\n" + msg.content
            else:
                # Google uses "model" instead of "assistant"
                role = "model" if msg.role == "assistant" else msg.role
                messages.append((role, msg.content))
        return system_text, messages

    @staticmethod
    def _build_request(system_text, messages, request: LLMRequest,
                       max_tokens: int | None = None, temperature: float | None = None):
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

        config_kwargs = {}
        if system_text is not None:
            config_kwargs["system_instruction"] = system_text
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if request.stop:
            config_kwargs["stop_sequences"] = request.stop

        config = types.GenerateContentConfig(**config_kwargs)
        return contents, config

    @staticmethod
    def _map_finish_reason(finish_reason) -> str:
        """Map Google GenAI finish reason to our standard finish_reason."""
        if finish_reason is None:
            return "stop"
        reason_str = str(finish_reason)
        return _FINISH_REASON_MAP.get(reason_str, "stop")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Google GenAI API."""
        client = self._get_client()

        system_text, messages = self._extract_messages(request)
        max_tokens, temperature = self.resolve_params(request)
        contents, config = self._build_request(
            system_text, messages, request, max_tokens=max_tokens, temperature=temperature,
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

        return LLMResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    async def complete_stream(
            self,
            request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using Google GenAI API."""
        client = self._get_client()

        system_text, messages = self._extract_messages(request)
        max_tokens, temperature = self.resolve_params(request)
        contents, config = self._build_request(
            system_text, messages, request, max_tokens=max_tokens, temperature=temperature,
        )
        model = request.model or self.model

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

        # Final chunk to signal completion
        yield LLMStreamChunk(
            content="",
            is_final=True,
            finish_reason="stop",
        )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
