"""
Disabled Reranker Provider (no-op).

Returns uniform scores - effectively disables reranking while maintaining API compatibility.
"""

from logging import Logger

from scitrera_app_framework import Variables

from ..base import RerankerProvider, RerankerProviderPluginBase


class NoneRerankerProvider(RerankerProvider):
    """
    No-op reranker that returns uniform scores.

    Use when reranking is not needed or to disable it for testing.
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """Return uniform scores (1.0) for all documents."""
        return [1.0] * len(documents)


class NoneRerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin for disabled reranker provider."""

    PROVIDER_NAME = "none"

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return NoneRerankerProvider(v=v)
