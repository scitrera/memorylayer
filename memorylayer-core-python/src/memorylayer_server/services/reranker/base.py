"""
Reranker Service Base - Abstract interfaces for document reranking.

Rerankers score query-document relevance for improved retrieval quality.
They are used after initial retrieval to re-order results by relevance.

Extension Points:
- reranker-provider: Low-level reranking model implementations
- reranker-service: High-level service wrapping providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Optional, Union, Any

from scitrera_app_framework.api import Variables, Plugin, enabled_option_pattern
from scitrera_app_framework import get_logger

from ...config import (
    MEMORYLAYER_RERANKER_PROVIDER, DEFAULT_MEMORYLAYER_RERANKER_PROVIDER,
    MEMORYLAYER_RERANKER_SERVICE, DEFAULT_MEMORYLAYER_RERANKER_SERVICE,
    MEMORYLAYER_RERANKER_PRELOAD_ENABLED, DEFAULT_MEMORYLAYER_RERANKER_PRELOAD_ENABLED,
)

from .._constants import EXT_RERANKER_PROVIDER, EXT_RERANKER_SERVICE


@dataclass
class RerankResult:
    """Result of a reranking operation."""
    index: int  # Original index in input list
    score: float  # Relevance score (0-1)
    document: Any  # Original document


class RerankerProvider(ABC):
    """
    Abstract reranker provider interface.

    Implementations provide low-level document reranking against queries.
    Can be text-only or multimodal (supporting images/video).
    """

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def preload(self):
        """
        Preload model resources.

        Optional - implementations can override for large models.
        """
        pass

    @abstractmethod
    async def rerank(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """
        Score documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional task-specific instruction

        Returns:
            List of relevance scores (0-1) for each document, same order as input
        """
        pass

    async def rerank_with_indices(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
            top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Score documents and return sorted indices with scores.

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional task-specific instruction
            top_k: Return only top K results (None for all)

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        scores = await self.rerank(query, documents, instruction)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


class MultimodalRerankerProvider(RerankerProvider):
    """
    Reranker provider that supports multimodal queries and documents.

    Extends base RerankerProvider with image/video support.
    """

    @abstractmethod
    async def rerank_multimodal(
            self,
            query: Union[str, dict],
            documents: list[Union[str, dict]],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """
        Score multimodal documents by relevance to a multimodal query.

        Args:
            query: Query string or dict with text/image/video keys
            documents: List of document strings or dicts with text/image/video keys
            instruction: Optional task-specific instruction

        Returns:
            List of relevance scores (0-1) for each document
        """
        pass


class RerankerService:
    """
    High-level reranker service wrapping a provider.

    Provides convenience methods for common reranking patterns.
    """

    def __init__(self, provider: RerankerProvider, v: Variables = None):
        self.provider = provider
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def rerank(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """Score documents by relevance to query."""
        return await self.provider.rerank(query, documents, instruction)

    async def rerank_with_indices(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
            top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Score documents and return sorted indices with scores."""
        return await self.provider.rerank_with_indices(query, documents, instruction, top_k)

    async def rerank_objects(
            self,
            query: str,
            objects: list[Any],
            content_fn,
            instruction: Optional[str] = None,
            top_k: Optional[int] = None,
    ) -> list[RerankResult]:
        """
        Rerank arbitrary objects using a content extraction function.

        Args:
            query: The search query
            objects: List of objects to rerank
            content_fn: Function to extract text content from each object
            instruction: Optional task-specific instruction
            top_k: Return only top K results

        Returns:
            List of RerankResult with original objects, sorted by relevance
        """
        if not objects:
            return []

        # Extract content from objects
        documents = [content_fn(obj) for obj in objects]

        # Get ranked indices
        ranked = await self.provider.rerank_with_indices(
            query, documents, instruction, top_k
        )

        # Build results with original objects
        results = [
            RerankResult(index=idx, score=score, document=objects[idx])
            for idx, score in ranked
        ]

        return results

    async def rerank_objects_adaptive(
            self,
            query: str,
            objects: list[Any],
            content_fn,
            score_fn,
            requested_k: int,
            instruction: Optional[str] = None,
    ) -> list[RerankResult]:
        """
        Rerank objects with adaptive candidate sizing based on initial scores.

        Subclasses can override for more sophisticated adaptive behavior.

        Args:
            query: The search query
            objects: List of objects to rerank
            content_fn: Function to extract text content from each object
            score_fn: Function to get initial similarity score from each object
            requested_k: Number of final results to return
            instruction: Optional task-specific instruction

        Returns:
            List of RerankResult with reranked objects
        """
        # Default implementation: just use rerank_objects with top_k
        return await self.rerank_objects(
            query=query,
            objects=objects,
            content_fn=content_fn,
            instruction=instruction,
            top_k=requested_k,
        )

    @property
    def supports_multimodal(self) -> bool:
        """Check if provider supports multimodal reranking."""
        return isinstance(self.provider, MultimodalRerankerProvider)


# Plugin base classes

class RerankerProviderPluginBase(Plugin):
    """Base Plugin for reranker providers."""
    PROVIDER_NAME: str = ''

    def name(self) -> str:
        return f"{EXT_RERANKER_PROVIDER}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RERANKER_PROVIDER

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_RERANKER_PROVIDER, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_RERANKER_PROVIDER, DEFAULT_MEMORYLAYER_RERANKER_PROVIDER)

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        """Preload reranker model if configured."""
        from scitrera_app_framework import ext_parse_bool

        # noinspection PyTypeChecker
        provider: RerankerProvider = value
        preload = v.environ(
            MEMORYLAYER_RERANKER_PRELOAD_ENABLED,
            default=DEFAULT_MEMORYLAYER_RERANKER_PRELOAD_ENABLED,
            type_fn=ext_parse_bool
        )

        if preload:
            provider.logger.info("Preloading reranker model in background")
            await provider.preload()


class RerankerServicePluginBase(Plugin):
    """Base Plugin for reranker service implementations."""
    PROVIDER_NAME: str = ''

    def name(self) -> str:
        return f"{EXT_RERANKER_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RERANKER_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_RERANKER_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_RERANKER_SERVICE, DEFAULT_MEMORYLAYER_RERANKER_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_RERANKER_PROVIDER,)
