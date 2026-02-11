"""Decay service package."""
from scitrera_app_framework import Variables, get_extension

from .base import (
    DecayService,
    DecayServicePluginBase,
    EXT_DECAY_SERVICE,
)


def get_decay_service(v: Variables = None) -> DecayService:
    """Get the decay service instance."""
    return get_extension(EXT_DECAY_SERVICE, v)


__all__ = (
    'DecayService',
    'DecayServicePluginBase',
    'get_decay_service',
    'EXT_DECAY_SERVICE',
)
