"""Anthropic Claude LLM provider."""
from typing import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import LLMProvider
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk

DEFAULT_LLM_ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'

# Anthropic stop_reason -> our finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude LLM provider.

    Uses the Anthropic Messages API for completions and streaming.
    """

    def __init__(
            self,
            api_key: str,
            model: str = DEFAULT_LLM_ANTHROPIC_MODEL,
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
            "Initialized AnthropicLLMProvider: model=%s", model
        )

    def _get_client(self):
        """Lazy-load Anthropic async client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    @staticmethod
    def _prepare_messages(request: LLMRequest):
        """Extract system message and format messages for Anthropic API.

        Anthropic requires system messages as a separate parameter,
        not within the messages list.

        Returns:
            Tuple of (system_message_text or None, formatted_messages_list)
        """
        system_text = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                # Anthropic takes system as a top-level parameter;
                # concatenate multiple system messages if present.
                if system_text is None:
                    system_text = msg.content
                else:
                    system_text += "\n" + msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})
        return system_text, messages

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic Messages API."""
        client = self._get_client()

        system_text, messages = self._prepare_messages(request)
        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        self.logger.debug("LLM request: model=%s, messages=%d", model, len(messages))

        kwargs = dict(
            model=model,
            messages=messages,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if system_text is not None:
            kwargs["system"] = system_text
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        response = await client.messages.create(**kwargs)

        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=_STOP_REASON_MAP.get(response.stop_reason, response.stop_reason or "stop"),
        )

    async def complete_stream(
            self,
            request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using Anthropic Messages API."""
        client = self._get_client()

        system_text, messages = self._prepare_messages(request)
        model = request.model or self.model

        max_tokens, temperature = self.resolve_params(request)

        kwargs = dict(
            model=model,
            messages=messages,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if system_text is not None:
            kwargs["system"] = system_text
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield LLMStreamChunk(
                    content=text,
                    is_final=False,
                )

            message = await stream.get_final_message()
            yield LLMStreamChunk(
                content="",
                is_final=True,
                finish_reason=_STOP_REASON_MAP.get(message.stop_reason, message.stop_reason or "stop"),
            )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
