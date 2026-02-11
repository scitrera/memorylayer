"""Contradiction service package."""
from scitrera_app_framework import Variables, get_extension

from .base import (
    ContradictionService,
    ContradictionServicePluginBase,
    EXT_CONTRADICTION_SERVICE,
)


def get_contradiction_service(v: Variables = None) -> ContradictionService:
    """Get the contradiction service instance."""
    return get_extension(EXT_CONTRADICTION_SERVICE, v)


__all__ = (
    'ContradictionService',
    'ContradictionServicePluginBase',
    'get_contradiction_service',
    'EXT_CONTRADICTION_SERVICE',
)
