"""Local reranker provider using sentence-transformers CrossEncoder."""
from .provider import LocalRerankerProvider, LocalRerankerProviderPlugin

__all__ = [
    'LocalRerankerProvider',
    'LocalRerankerProviderPlugin',
]
