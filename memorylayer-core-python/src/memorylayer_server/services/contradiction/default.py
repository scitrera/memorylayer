"""Default contradiction service implementation."""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import ContradictionService, ContradictionServicePluginBase, ContradictionRecord
from ..storage import EXT_STORAGE_BACKEND
from ..storage.base import StorageBackend

# Negation pairs used for simple textual contradiction detection.
# For each pair, if text_a contains one term and text_b contains the other,
# a negation-type contradiction is flagged.
NEGATION_PAIRS = [
    ("use", "don't use"), ("use", "do not use"), ("use", "avoid"),
    ("enable", "disable"), ("add", "remove"),
    ("true", "false"), ("always", "never"),
    ("should", "should not"), ("should", "shouldn't"),
    ("must", "must not"), ("must", "mustn't"),
    ("can", "cannot"), ("can", "can't"),
    ("is", "is not"), ("is", "isn't"),
    ("prefer", "avoid"), ("recommended", "not recommended"),
    ("include", "exclude"), ("allow", "deny"), ("allow", "block"),
]


class DefaultContradictionService(ContradictionService):
    """Default contradiction implementation using storage backend directly."""

    def __init__(self, storage: StorageBackend, v: Variables = None):
        self._storage = storage
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def check_new_memory(self, workspace_id: str, memory_id: str) -> list[ContradictionRecord]:
        """Find contradictions between a new memory and existing memories.

        1. Get the new memory from storage
        2. Search for similar memories using embedding similarity
        3. Check for negation patterns between the new memory and each similar memory
        4. Store and return any detected contradictions
        """
        new_memory = await self._storage.get_memory(workspace_id, memory_id, track_access=False)
        if not new_memory:
            self.logger.warning("Memory %s not found in workspace %s", memory_id, workspace_id)
            return []

        if not new_memory.embedding:
            self.logger.debug("Memory %s has no embedding, skipping contradiction check", memory_id)
            return []

        # Search for similar memories
        similar_memories = await self._storage.search_memories(
            workspace_id,
            query_embedding=new_memory.embedding,
            limit=20,
            min_relevance=0.7,
        )

        contradictions = []
        for existing_memory, relevance in similar_memories:
            # Skip self-comparison
            if existing_memory.id == memory_id:
                continue

            if self._has_negation_pattern(new_memory.content, existing_memory.content):
                record = ContradictionRecord(
                    workspace_id=workspace_id,
                    memory_a_id=memory_id,
                    memory_b_id=existing_memory.id,
                    contradiction_type='negation',
                    confidence=relevance,
                    detection_method='negation_pattern',
                )
                stored = await self._storage.create_contradiction(record)
                contradictions.append(stored)
                self.logger.info(
                    "Contradiction detected between %s and %s (confidence=%.2f)",
                    memory_id, existing_memory.id, relevance,
                )

        return contradictions

    async def get_unresolved(self, workspace_id: str, limit: int = 10) -> list[ContradictionRecord]:
        """Get unresolved contradictions for a workspace."""
        return await self._storage.get_unresolved_contradictions(workspace_id, limit)

    async def resolve(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: Optional[str] = None,
    ) -> Optional[ContradictionRecord]:
        """Resolve a contradiction by applying the chosen resolution strategy.

        Args:
            workspace_id: Workspace boundary
            contradiction_id: ID of the contradiction to resolve
            resolution: One of "keep_a", "keep_b", "keep_both", "merge"
            merged_content: Required when resolution is "merge"

        Returns:
            Updated contradiction record, or None if not found
        """
        record = await self._storage.get_contradiction(workspace_id, contradiction_id)
        if not record:
            self.logger.warning("Contradiction %s not found in workspace %s", contradiction_id, workspace_id)
            return None

        if resolution == 'keep_a':
            # Soft-delete memory B
            await self._storage.delete_memory(workspace_id, record.memory_b_id, hard=False)
            self.logger.info("Resolved contradiction %s: keeping memory %s, soft-deleted %s",
                             contradiction_id, record.memory_a_id, record.memory_b_id)

        elif resolution == 'keep_b':
            # Soft-delete memory A
            await self._storage.delete_memory(workspace_id, record.memory_a_id, hard=False)
            self.logger.info("Resolved contradiction %s: keeping memory %s, soft-deleted %s",
                             contradiction_id, record.memory_b_id, record.memory_a_id)

        elif resolution == 'merge' and merged_content:
            # Update memory A with merged content, soft-delete memory B
            await self._storage.update_memory(workspace_id, record.memory_a_id, content=merged_content)
            await self._storage.delete_memory(workspace_id, record.memory_b_id, hard=False)
            self.logger.info("Resolved contradiction %s: merged into %s, soft-deleted %s",
                             contradiction_id, record.memory_a_id, record.memory_b_id)

        elif resolution == 'keep_both':
            self.logger.info("Resolved contradiction %s: keeping both memories", contradiction_id)

        else:
            self.logger.warning("Unknown resolution strategy: %s", resolution)
            return None

        # Mark contradiction as resolved in storage
        return await self._storage.resolve_contradiction(
            workspace_id, contradiction_id, resolution, merged_content
        )

    @staticmethod
    def _has_negation_pattern(text_a: str, text_b: str) -> bool:
        """Check for negation patterns between two texts.

        For each pair, checks if text_a contains one term and text_b contains
        the other (in either direction).

        Args:
            text_a: First text to compare
            text_b: Second text to compare

        Returns:
            True if a negation pattern is detected
        """
        lower_a = text_a.lower()
        lower_b = text_b.lower()

        for term_pos, term_neg in NEGATION_PAIRS:
            # Check both directions: a has positive and b has negative, or vice versa
            if (term_pos in lower_a and term_neg in lower_b) or \
               (term_neg in lower_a and term_pos in lower_b):
                return True

        return False


class DefaultContradictionServicePlugin(ContradictionServicePluginBase):
    """Plugin that creates the default contradiction service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> ContradictionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultContradictionService(storage=storage, v=v)
