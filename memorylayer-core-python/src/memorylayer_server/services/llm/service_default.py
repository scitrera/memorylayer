"""Default LLM service implementation."""
from logging import Logger
from typing import AsyncIterator, Optional, List

from scitrera_app_framework import get_logger, Variables

from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk, LLMMessage, LLMRole
from .base import LLMProvider, EXT_LLM_REGISTRY, LLMServicePluginBase
from .registry import LLMProviderRegistry


class LLMService:
    """High-level LLM service wrapping provider registry.

    Similar to EmbeddingService wrapping EmbeddingProvider.
    Adds convenience methods for common patterns like synthesis.
    Supports profile-based routing through the registry.
    """

    def __init__(self, registry: LLMProviderRegistry, v: Variables = None):
        self.registry = registry
        self.logger = get_logger(v, name=self.__class__.__name__)

    @property
    def provider(self) -> LLMProvider:
        """Default provider for backward compatibility."""
        return self.registry.get_provider("default")

    async def complete(self, request: LLMRequest, profile: str = "default") -> LLMResponse:
        """Route completion to the provider for the given profile."""
        return await self.registry.complete(request, profile=profile)

    async def complete_stream(
            self,
            request: LLMRequest,
            profile: str = "default",
    ) -> AsyncIterator[LLMStreamChunk]:
        """Route streaming completion to the provider for the given profile."""
        async for chunk in self.registry.complete_stream(request, profile=profile):
            yield chunk

    async def synthesize(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = None,
            temperature: float = None,
            temperature_factor: float = None,
            profile: str = "default",
    ) -> str:
        """Simple synthesis - prompt with optional context.

        Args:
            prompt: User prompt/question
            context: Optional context to include
            max_tokens: Maximum response tokens (None = provider default)
            temperature: Explicit sampling temperature (overrides factor)
            temperature_factor: Multiplier against provider's default temperature
            profile: LLM provider profile to use

        Returns:
            Generated text
        """
        messages = []

        if context:
            messages.append(LLMMessage(
                role=LLMRole.SYSTEM,
                content=f"Use this context to inform your response:\n\n{context}"
            ))

        messages.append(LLMMessage(role=LLMRole.USER, content=prompt))

        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            temperature_factor=temperature_factor,
        )

        response = await self.registry.complete(request, profile=profile)
        return response.content

    async def answer_question(
            self,
            question: str,
            memories: List[str],
            max_tokens: int = 500,
            profile: str = "default",
    ) -> str:
        """Answer question using memories as context.

        Args:
            question: User question
            memories: List of memory contents
            max_tokens: Maximum response tokens
            profile: LLM provider profile to use

        Returns:
            Generated answer
        """
        context = "\n\n".join([f"- {m}" for m in memories])

        system_prompt = f"""You are a helpful assistant with access to the user's memories.
Answer questions based on the provided memories. If the memories don't contain
relevant information, say so.

Memories:
{context}"""

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            LLMMessage(role=LLMRole.USER, content=question),
        ]

        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature_factor=0.7,  # Lower temp for factual answers
        )

        response = await self.registry.complete(request, profile=profile)
        return response.content

    @property
    def default_model(self) -> str:
        """Default model from provider."""
        return self.provider.default_model

    @property
    def supports_streaming(self) -> bool:
        """Streaming support from provider."""
        return self.provider.supports_streaming


class DefaultLLMServicePlugin(LLMServicePluginBase):
    """Plugin for default LLM service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> LLMService:
        registry: LLMProviderRegistry = self.get_extension(EXT_LLM_REGISTRY, v)
        return LLMService(registry=registry, v=v)
