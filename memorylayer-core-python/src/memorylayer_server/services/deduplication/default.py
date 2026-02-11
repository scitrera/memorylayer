"""
Default Deduplication Service implementation.

Prevents duplicate memories during session extraction and manual remember operations.
Uses content hashing for exact matches and embedding similarity for semantic matches.
"""
from typing import Optional
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from ..embedding import EXT_EMBEDDING_SERVICE, EmbeddingService
from .base import (
    DeduplicationService,
    DeduplicationServicePluginBase,
    DeduplicationAction,
    DeduplicationResult,
    MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD,
    DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD,
    MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD,
    DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD,
)


class DefaultDeduplicationService(DeduplicationService):
    """Default deduplication service implementation."""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_service: EmbeddingService,
        v: Variables = None
    ):
        """
        Initialize deduplication service.

        Args:
            storage: Storage backend for memory access
            embedding_service: Embedding service for similarity computation
            v: Variables for logging context
        """
        self.storage = storage
        self.embedding_service = embedding_service
        self.logger = get_logger(v, name=self.__class__.__name__)

        # Get thresholds from config with defaults
        self.similarity_threshold = v.get(
            MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD,
            DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD
        )
        self.merge_threshold = v.get(
            MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD,
            DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD
        )

        self.logger.info(
            "Initialized DefaultDeduplicationService with thresholds: similarity=%.2f, merge=%.2f",
            self.similarity_threshold, self.merge_threshold
        )

    async def check_duplicate(
        self,
        content: str,
        content_hash: str,
        embedding: list[float],
        workspace_id: str
    ) -> DeduplicationResult:
        """
        Check if a memory is a duplicate.

        Checks in order:
        1. Exact content hash match (SKIP)
        2. High embedding similarity >= similarity_threshold (UPDATE)
        3. Moderate similarity >= merge_threshold (MERGE candidate)
        4. Otherwise (CREATE)

        Args:
            content: Memory content
            content_hash: SHA-256 hash of content
            embedding: Content embedding vector
            workspace_id: Workspace to check against

        Returns:
            DeduplicationResult with action and details
        """
        # 1. Check exact hash match
        existing = await self.storage.get_memory_by_hash(workspace_id, content_hash)
        if existing:
            self.logger.debug("Found exact duplicate: %s", existing.id)
            return DeduplicationResult(
                action=DeduplicationAction.SKIP,
                existing_memory_id=existing.id,
                similarity_score=1.0,
                reason="Exact content duplicate"
            )

        # 2. Check embedding similarity
        similar_memories = await self.storage.search_memories(
            workspace_id=workspace_id,
            query_embedding=embedding,
            limit=5,
            min_relevance=self.merge_threshold
        )

        if similar_memories:
            top_match, top_score = similar_memories[0]

            if top_score >= self.similarity_threshold:
                self.logger.debug(
                    "Found semantic duplicate: %s (similarity: %.3f)",
                    top_match.id, top_score
                )
                return DeduplicationResult(
                    action=DeduplicationAction.UPDATE,
                    existing_memory_id=top_match.id,
                    similarity_score=top_score,
                    reason=f"Semantic duplicate (similarity: {top_score:.3f})"
                )
            elif top_score >= self.merge_threshold:
                self.logger.debug(
                    "Found merge candidate: %s (similarity: %.3f)",
                    top_match.id, top_score
                )
                return DeduplicationResult(
                    action=DeduplicationAction.MERGE,
                    existing_memory_id=top_match.id,
                    similarity_score=top_score,
                    reason=f"Potential merge candidate (similarity: {top_score:.3f})"
                )

        # 3. No duplicates found
        return DeduplicationResult(
            action=DeduplicationAction.CREATE,
            reason="New unique memory"
        )

    async def deduplicate_batch(
        self,
        candidates: list[tuple[str, str, list[float]]],
        workspace_id: str
    ) -> list[DeduplicationResult]:
        """
        Check multiple memories for duplicates.

        Args:
            candidates: List of (content, content_hash, embedding) tuples
            workspace_id: Workspace to check against

        Returns:
            List of DeduplicationResult in same order as candidates
        """
        results = []
        for content, content_hash, embedding in candidates:
            result = await self.check_duplicate(
                content, content_hash, embedding, workspace_id
            )
            results.append(result)

        self.logger.info(
            "Deduplicated batch of %s candidates: %s create, %s skip, %s update, %s merge",
            len(candidates),
            sum(1 for r in results if r.action == DeduplicationAction.CREATE),
            sum(1 for r in results if r.action == DeduplicationAction.SKIP),
            sum(1 for r in results if r.action == DeduplicationAction.UPDATE),
            sum(1 for r in results if r.action == DeduplicationAction.MERGE),
        )

        return results


class DefaultDeduplicationServicePlugin(DeduplicationServicePluginBase):
    """Default deduplication service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> DeduplicationService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        embedding_service: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)
        return DefaultDeduplicationService(
            storage=storage,
            embedding_service=embedding_service,
            v=v
        )
