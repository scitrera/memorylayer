"""
Reranker Service - Document reranking for improved retrieval quality.

Extension Points:
- reranker-provider: Low-level reranking model implementations
- reranker-service: High-level service wrapping providers
"""

from scitrera_app_framework import get_extension, Variables

from .base import (
    EXT_RERANKER_PROVIDER,
    EXT_RERANKER_SERVICE,
    RerankerProvider,
    MultimodalRerankerProvider,
    RerankerService,
    RerankResult,
    RerankerProviderPluginBase,
    RerankerServicePluginBase,
)


def get_reranker_provider(v: Variables) -> RerankerProvider:
    """Get the active reranker provider."""
    return get_extension(EXT_RERANKER_PROVIDER, v=v)


def get_reranker_service(v: Variables) -> RerankerService:
    """Get the active reranker service."""
    return get_extension(EXT_RERANKER_SERVICE, v=v)


__all__ = [
    # Extension points
    'EXT_RERANKER_PROVIDER',
    'EXT_RERANKER_SERVICE',
    # Base classes
    'RerankerProvider',
    'MultimodalRerankerProvider',
    'RerankerService',
    'RerankResult',
    # Plugin bases
    'RerankerProviderPluginBase',
    'RerankerServicePluginBase',
    # Getters
    'get_reranker_provider',
    'get_reranker_service',
]
