"""Reflect service package."""
from .base import (
    ReflectServicePluginBase,
    EXT_REFLECT_SERVICE,
)
from .default import ReflectService

from scitrera_app_framework import Variables, get_extension


def get_reflect_service(v: Variables = None) -> ReflectService:
    """Get the reflect service instance."""
    return get_extension(EXT_REFLECT_SERVICE, v)


__all__ = (
    'ReflectService',
    'ReflectServicePluginBase',
    'get_reflect_service',
    'EXT_REFLECT_SERVICE',
)
