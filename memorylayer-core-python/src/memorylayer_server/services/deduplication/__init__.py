"""Deduplication service package."""
from .base import (
    DeduplicationServicePluginBase,
    EXT_DEDUPLICATION_SERVICE,
    DeduplicationService,
    DeduplicationAction,
    DeduplicationResult,
)

from scitrera_app_framework import Variables, get_extension


def get_deduplication_service(v: Variables = None) -> DeduplicationService:
    """Get the deduplication service instance."""
    return get_extension(EXT_DEDUPLICATION_SERVICE, v)


__all__ = (
    'DeduplicationService',
    'DeduplicationServicePluginBase',
    'get_deduplication_service',
    'EXT_DEDUPLICATION_SERVICE',
    'DeduplicationAction',
    'DeduplicationResult',
)
