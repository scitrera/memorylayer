"""LLM service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_LLM_REGISTRY,
    EXT_LLM_SERVICE,
    LLMProvider,
    LLMProviderRegistryPluginBase,
    LLMServicePluginBase,
)
from .noop import LLMNotConfiguredError
from .registry import LLMProviderRegistry
from .service_default import LLMService


def get_llm_registry(v: Variables = None) -> LLMProviderRegistry:
    """Get the LLM provider registry instance."""
    return get_extension(EXT_LLM_REGISTRY, v)


def get_llm_service(v: Variables = None) -> LLMService:
    """Get the LLM service instance."""
    return get_extension(EXT_LLM_SERVICE, v)


__all__ = (
    "LLMProvider",
    "LLMProviderRegistry",
    "LLMProviderRegistryPluginBase",
    "LLMService",
    "LLMServicePluginBase",
    "get_llm_registry",
    "get_llm_service",
    "EXT_LLM_REGISTRY",
    "EXT_LLM_SERVICE",
    "LLMNotConfiguredError",
)
