"""
Default Reranker Service Implementation.

Provides high-level reranking with adaptive candidate list sizing.
"""

from logging import Logger
from typing import Any, Optional

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_RERANKER_PROVIDER,
    RerankerProvider,
    RerankerService,
    RerankerServicePluginBase,
    RerankResult,
)


# Adaptive reranking configuration
ADAPTIVE_MIN_CANDIDATES = 10  # Minimum candidates to consider
ADAPTIVE_MAX_CANDIDATES = 50  # Maximum candidates to consider
ADAPTIVE_QUALITY_THRESHOLD = 0.7  # Score threshold for "good" results
ADAPTIVE_EXPANSION_FACTOR = 1.5  # How much to expand when quality is low


class DefaultRerankerService(RerankerService):
    """
    Default reranker service with adaptive candidate list sizing.

    Features:
    - Wraps any RerankerProvider
    - Adaptive candidate expansion based on initial result quality
    - Convenience methods for reranking objects
    """

    def __init__(self, provider: RerankerProvider, v: Variables = None):
        super().__init__(provider, v)
        self.min_candidates = ADAPTIVE_MIN_CANDIDATES
        self.max_candidates = ADAPTIVE_MAX_CANDIDATES
        self.quality_threshold = ADAPTIVE_QUALITY_THRESHOLD
        self.expansion_factor = ADAPTIVE_EXPANSION_FACTOR

    def compute_adaptive_k(
        self,
        initial_scores: list[float],
        requested_k: int,
        available_count: int,
    ) -> int:
        """
        Compute adaptive candidate list size based on initial result quality.

        If top scores are low (below quality threshold), expand the candidate
        list to give the reranker more options to find relevant documents.

        Args:
            initial_scores: Similarity scores from initial retrieval (0-1)
            requested_k: Number of final results requested
            available_count: Total documents available

        Returns:
            Recommended number of candidates to rerank
        """
        if not initial_scores:
            return min(self.min_candidates, available_count)

        # Check quality of top results
        top_k = min(requested_k, len(initial_scores))
        top_scores = sorted(initial_scores, reverse=True)[:top_k]
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0

        # Compute base candidate count
        base_candidates = max(requested_k * 3, self.min_candidates)

        # Expand if quality is low
        if avg_top_score < self.quality_threshold:
            quality_ratio = avg_top_score / self.quality_threshold
            expansion = self.expansion_factor * (1 - quality_ratio)
            base_candidates = int(base_candidates * (1 + expansion))

        # Clamp to bounds
        candidates = max(self.min_candidates, min(base_candidates, self.max_candidates))
        candidates = min(candidates, available_count)

        self.logger.debug(
            "Adaptive rerank: avg_top_score=%.3f, candidates=%d (of %d available)",
            avg_top_score,
            candidates,
            available_count,
        )

        return candidates

    async def rerank_with_adaptive_k(
        self,
        query: str,
        documents: list[str],
        initial_scores: list[float],
        requested_k: int,
        instruction: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank with adaptive candidate list sizing.

        Args:
            query: The search query
            documents: Full list of candidate documents
            initial_scores: Similarity scores from initial retrieval
            requested_k: Number of final results to return
            instruction: Optional task-specific instruction

        Returns:
            List of (original_index, rerank_score) for top results
        """
        if not documents:
            return []

        # Compute adaptive candidate count
        candidates_k = self.compute_adaptive_k(
            initial_scores, requested_k, len(documents)
        )

        # Get top candidates by initial score
        indexed_initial = list(enumerate(initial_scores))
        indexed_initial.sort(key=lambda x: x[1], reverse=True)
        top_candidates = indexed_initial[:candidates_k]

        # Extract documents for reranking
        candidate_indices = [idx for idx, _ in top_candidates]
        candidate_docs = [documents[idx] for idx in candidate_indices]

        # Rerank candidates
        rerank_scores = await self.provider.rerank(query, candidate_docs, instruction)

        # Map back to original indices and sort by rerank score
        results = [
            (candidate_indices[i], score)
            for i, score in enumerate(rerank_scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:requested_k]

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
        Rerank objects with adaptive candidate sizing.

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
        if not objects:
            return []

        # Extract content and initial scores
        documents = [content_fn(obj) for obj in objects]
        initial_scores = [score_fn(obj) for obj in objects]

        # Rerank with adaptive sizing
        ranked = await self.rerank_with_adaptive_k(
            query, documents, initial_scores, requested_k, instruction
        )

        # Build results
        results = [
            RerankResult(index=idx, score=score, document=objects[idx])
            for idx, score in ranked
        ]

        return results


class DefaultRerankerServicePlugin(RerankerServicePluginBase):
    """Plugin for default reranker service."""

    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        provider = self.get_extension(EXT_RERANKER_PROVIDER, v)
        return DefaultRerankerService(provider=provider, v=v)
