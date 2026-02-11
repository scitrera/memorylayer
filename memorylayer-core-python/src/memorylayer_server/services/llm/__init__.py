"""LLM service package."""
from .base import (
    LLMProvider,
    LLMProviderRegistryPluginBase,
    LLMServicePluginBase,
    EXT_LLM_PROVIDER,
    EXT_LLM_REGISTRY,
    EXT_LLM_SERVICE,
)
from .registry import LLMProviderRegistry
from .service_default import LLMService
from .noop import LLMNotConfiguredError

from scitrera_app_framework import Variables, get_extension


def get_llm_registry(v: Variables = None) -> LLMProviderRegistry:
    """Get the LLM provider registry instance."""
    return get_extension(EXT_LLM_REGISTRY, v)


def get_llm_provider(v: Variables = None) -> LLMProvider:
    """Get the default LLM provider instance.

    Backward-compatible convenience function that delegates through the registry.
    """
    return get_llm_registry(v).get_provider("default")


def get_llm_service(v: Variables = None) -> LLMService:
    """Get the LLM service instance."""
    return get_extension(EXT_LLM_SERVICE, v)


__all__ = (
    'LLMProvider',
    'LLMProviderRegistry',
    'LLMProviderRegistryPluginBase',
    'LLMService',
    'LLMServicePluginBase',
    'get_llm_provider',
    'get_llm_registry',
    'get_llm_service',
    'EXT_LLM_PROVIDER',
    'EXT_LLM_REGISTRY',
    'EXT_LLM_SERVICE',
    'LLMNotConfiguredError',
)
