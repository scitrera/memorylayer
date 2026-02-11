"""OpenAI-compatible LLM provider."""
from typing import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import LLMProvider
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk

DEFAULT_LLM_OPENAI_MODEL = 'gpt-5-nano'


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with OpenAI API, Azure OpenAI, Ollama, vLLM, and any
    OpenAI-compatible endpoint by configuring the base URL.
    """

    def __init__(
            self,
            api_key: str,
            base_url: str = None,
            model: str = DEFAULT_LLM_OPENAI_MODEL,
            default_max_tokens: int | None = None,
            default_temperature: float | None = None,
            v: Variables = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._client = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized OpenAILLMProvider: base_url=%s, model=%s",
            base_url, model
        )

    def _get_client(self):
        """Lazy-load OpenAI async client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        client = self._get_client()

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        self.logger.debug("LLM request: model=%s, messages=%d", model, len(messages))

        kwargs = dict(
            model=model,
            messages=messages,
            stop=request.stop,
        )
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=choice.finish_reason or "stop",
        )

    async def complete_stream(
            self,
            request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using OpenAI API."""
        client = self._get_client()

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        kwargs = dict(
            model=model,
            messages=messages,
            stop=request.stop,
            stream=True,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield LLMStreamChunk(
                        content=delta.content,
                        is_final=False,
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
