"""LLM Service - Pluggable LLM provider interface."""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk

from .._constants import EXT_LLM_PROVIDER, EXT_LLM_SERVICE, EXT_LLM_REGISTRY

# Registry config constants
MEMORYLAYER_LLM_REGISTRY = 'MEMORYLAYER_LLM_REGISTRY'
DEFAULT_MEMORYLAYER_LLM_REGISTRY = 'default'

# Service config constants
MEMORYLAYER_LLM_SERVICE = 'MEMORYLAYER_LLM_SERVICE'
DEFAULT_MEMORYLAYER_LLM_SERVICE = 'default'


class LLMProvider(ABC):
    """Abstract LLM provider interface.

    Provides low-level access to LLM completions.
    Similar to EmbeddingProvider, this is the actual API client.
    """

    # Subclasses should set these; used by resolve_params().
    default_max_tokens: int | None = None
    default_temperature: float | None = None

    def resolve_params(self, request: LLMRequest) -> tuple[int | None, float | None]:
        """Resolve effective max_tokens and temperature for a request.

        Resolution order for max_tokens:
            request.max_tokens > self.default_max_tokens

        Resolution order for temperature:
            request.temperature > (request.temperature_factor * default) > self.default_temperature
        """
        max_tokens = request.max_tokens if request.max_tokens is not None else self.default_max_tokens
        if request.temperature is not None:
            temperature = request.temperature
        elif request.temperature_factor is not None and self.default_temperature is not None:
            temperature = self.default_temperature * request.temperature_factor
        else:
            temperature = self.default_temperature
        return max_tokens, temperature

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM.

        Args:
            request: LLM request with messages and parameters

        Returns:
            LLM response with content and token counts
        """
        pass

    @abstractmethod
    async def complete_stream(
            self,
            request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion.

        Args:
            request: LLM request with stream=True

        Yields:
            LLMStreamChunk for each token/chunk
        """
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name for this provider."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming."""
        pass


# noinspection PyAbstractClass
class LLMProviderRegistryPluginBase(Plugin):
    """Base plugin for LLM provider registry."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_LLM_REGISTRY}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_LLM_REGISTRY

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_LLM_REGISTRY, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_LLM_REGISTRY, DEFAULT_MEMORYLAYER_LLM_REGISTRY)


# noinspection PyAbstractClass
class LLMServicePluginBase(Plugin):
    """Base plugin for LLM service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_LLM_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_LLM_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_LLM_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_LLM_SERVICE, DEFAULT_MEMORYLAYER_LLM_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_LLM_REGISTRY,)
