"""Tier generation service package."""
from scitrera_app_framework import Variables, get_extension

from .base import (
    SemanticTieringService,
    SemanticTieringServicePluginBase,
    EXT_SEMANTIC_TIERING_SERVICE,
)


def get_semantic_tiering_service(v: Variables = None) -> SemanticTieringService:
    """Get the semantic tier generation service instance."""
    return get_extension(EXT_SEMANTIC_TIERING_SERVICE, v)


__all__ = (
    'SemanticTieringService',
    'SemanticTieringServicePluginBase',
    'get_semantic_tiering_service',
    'EXT_SEMANTIC_TIERING_SERVICE',
)
