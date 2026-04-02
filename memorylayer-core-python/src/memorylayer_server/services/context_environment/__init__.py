"""Context environment service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_CONTEXT_ENVIRONMENT_SERVICE,
    ContextEnvironmentService,
    ContextEnvironmentServicePluginBase,
)


def get_context_environment_service(v: Variables = None) -> ContextEnvironmentService:
    """Get the context environment service instance."""
    return get_extension(EXT_CONTEXT_ENVIRONMENT_SERVICE, v)


__all__ = (
    "ContextEnvironmentService",
    "ContextEnvironmentServicePluginBase",
    "get_context_environment_service",
    "EXT_CONTEXT_ENVIRONMENT_SERVICE",
)
