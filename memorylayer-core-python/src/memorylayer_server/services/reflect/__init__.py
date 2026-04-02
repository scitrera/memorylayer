"""Reflect service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_REFLECT_SERVICE,
    ReflectServicePluginBase,
)
from .default import ReflectService


def get_reflect_service(v: Variables = None) -> ReflectService:
    """Get the reflect service instance."""
    return get_extension(EXT_REFLECT_SERVICE, v)


__all__ = (
    "ReflectService",
    "ReflectServicePluginBase",
    "get_reflect_service",
    "EXT_REFLECT_SERVICE",
)
