from .base import EmbeddingProvider, EXT_EMBEDDING_PROVIDER, EXT_EMBEDDING_SERVICE
from .service_default import EmbeddingService

from scitrera_app_framework import Variables, get_extension


def get_embedding_provider(v: Variables = None) -> EmbeddingProvider:
    return get_extension(EXT_EMBEDDING_PROVIDER, v)


def get_embedding_service(v: Variables = None) -> EmbeddingService:
    return get_extension(EXT_EMBEDDING_SERVICE, v)


__all__ = (
    'EmbeddingProvider',
    'EmbeddingService',
    'get_embedding_provider',
    'get_embedding_service',
    'EXT_EMBEDDING_PROVIDER',
    'EXT_EMBEDDING_SERVICE',
)
