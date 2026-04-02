"""Default contradiction service implementation."""

import re
from datetime import UTC
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...utils import dot_product as _dot_product_util
from ..storage import EXT_STORAGE_BACKEND
from ..storage.base import StorageBackend
from .base import (
    CONTRADICTION_TYPE_NEGATION,
    CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT,
    ContradictionRecord,
    ContradictionService,
    ContradictionServicePluginBase,
)

# Negation pairs used for simple textual contradiction detection.
# For each pair, if text_a contains one term and text_b contains the other,
# a negation-type contradiction is flagged.
NEGATION_PAIRS = [
    ("use", "don't use"),
    ("use", "do not use"),
    ("use", "avoid"),
    ("enable", "disable"),
    ("add", "remove"),
    ("true", "false"),
    ("always", "never"),
    ("should", "should not"),
    ("should", "shouldn't"),
    ("must", "must not"),
    ("must", "mustn't"),
    ("can", "cannot"),
    ("can", "can't"),
    ("is", "is not"),
    ("is", "isn't"),
    ("prefer", "avoid"),
    ("recommended", "not recommended"),
    ("include", "exclude"),
    ("allow", "deny"),
    ("allow", "block"),
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

            # Determine which memory is newer for temporal ordering
            newer_id = self._determine_newer_memory(new_memory, existing_memory)

            if self._has_negation_pattern(new_memory.content, existing_memory.content):
                record = ContradictionRecord(
                    workspace_id=workspace_id,
                    memory_a_id=memory_id,
                    memory_b_id=existing_memory.id,
                    contradiction_type=CONTRADICTION_TYPE_NEGATION,
                    confidence=relevance,
                    detection_method="negation_pattern",
                    newer_memory_id=newer_id,
                )
                stored = await self._storage.create_contradiction(record)
                contradictions.append(stored)
                self.logger.info(
                    "Contradiction detected between %s and %s (confidence=%.2f)",
                    memory_id,
                    existing_memory.id,
                    relevance,
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
        merged_content: str | None = None,
    ) -> ContradictionRecord | None:
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

        if resolution == "keep_a":
            # Soft-delete memory B
            await self._storage.delete_memory(workspace_id, record.memory_b_id, hard=False)
            self.logger.info(
                "Resolved contradiction %s: keeping memory %s, soft-deleted %s", contradiction_id, record.memory_a_id, record.memory_b_id
            )

        elif resolution == "keep_b":
            # Soft-delete memory A
            await self._storage.delete_memory(workspace_id, record.memory_a_id, hard=False)
            self.logger.info(
                "Resolved contradiction %s: keeping memory %s, soft-deleted %s", contradiction_id, record.memory_b_id, record.memory_a_id
            )

        elif resolution == "merge" and merged_content:
            # Update memory A with merged content, soft-delete memory B
            await self._storage.update_memory(workspace_id, record.memory_a_id, content=merged_content)
            await self._storage.delete_memory(workspace_id, record.memory_b_id, hard=False)
            self.logger.info(
                "Resolved contradiction %s: merged into %s, soft-deleted %s", contradiction_id, record.memory_a_id, record.memory_b_id
            )

        elif resolution == "keep_both":
            self.logger.info("Resolved contradiction %s: keeping both memories", contradiction_id)

        else:
            self.logger.warning("Unknown resolution strategy: %s", resolution)
            return None

        # Mark contradiction as resolved in storage
        return await self._storage.resolve_contradiction(workspace_id, contradiction_id, resolution, merged_content)

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
            if (term_pos in lower_a and term_neg in lower_b) or (term_neg in lower_a and term_pos in lower_b):
                return True

        return False

    @staticmethod
    def _extract_entity_values(text: str) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, value) triples from text using regex patterns.

        Patterns match constructions like:
          "<subject> is <value>"
          "<subject> uses <value>"
          "<subject> runs <value>"

        Args:
            text: Text to extract entity-value pairs from

        Returns:
            List of (subject, predicate, value) tuples (all lowercased)
        """
        patterns = [
            r"(\w[\w\s]{1,30}?)\s+(is|uses|runs|has|uses|requires|needs)\s+([\w][\w\s\-\.]{0,40})",
        ]
        results = []
        lower_text = text.lower()
        for pattern in patterns:
            for match in re.finditer(pattern, lower_text):
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                value = match.group(3).strip()
                # Filter out very short or very long subjects/values
                if 2 <= len(subject) <= 50 and 1 <= len(value) <= 60:
                    results.append((subject, predicate, value))
        return results

    @staticmethod
    def _dot_product(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute dot product similarity between two vectors (assumed unit-normalized)."""
        return _dot_product_util(vec_a, vec_b)

    @staticmethod
    def _determine_newer_memory(memory_a, memory_b) -> str | None:
        """Determine which memory is newer based on created_at timestamp.

        Returns the ID of the newer memory, or None if timestamps are unavailable.
        """
        created_a = getattr(memory_a, "created_at", None)
        created_b = getattr(memory_b, "created_at", None)
        if created_a is None or created_b is None:
            return None
        return memory_a.id if created_a >= created_b else memory_b.id

    async def check_semantic_conflict(self, memory_a, memory_b) -> ContradictionRecord | None:
        """Check if two memories have a semantic value conflict.

        Detection logic:
        1. Compute embedding similarity; must be in range [0.7, 0.9] (similar topic, not identical)
        2. Extract entity-value triples from each memory
        3. If two triples share the same subject+predicate but have different values → conflict

        Args:
            memory_a: First memory object
            memory_b: Second memory object

        Returns:
            ContradictionRecord if a conflict is found, None otherwise
        """
        # Both memories need embeddings for similarity check
        emb_a = getattr(memory_a, "embedding", None)
        emb_b = getattr(memory_b, "embedding", None)
        if not emb_a or not emb_b:
            return None

        similarity = self._dot_product(emb_a, emb_b)

        # Similarity in [0.7, 0.9]: related topic but different enough to conflict
        if not (0.7 <= similarity <= 0.9):
            return None

        triples_a = self._extract_entity_values(memory_a.content)
        triples_b = self._extract_entity_values(memory_b.content)

        if not triples_a or not triples_b:
            return None

        # Build lookup: (subject, predicate) -> value for memory_b
        lookup_b: dict[tuple[str, str], str] = {(subj, pred): val for subj, pred, val in triples_b}

        for subj_a, pred_a, val_a in triples_a:
            key = (subj_a, pred_a)
            if key in lookup_b:
                val_b = lookup_b[key]
                if val_a != val_b:
                    newer_id = self._determine_newer_memory(memory_a, memory_b)
                    record = ContradictionRecord(
                        workspace_id=memory_a.workspace_id if hasattr(memory_a, "workspace_id") else "",
                        memory_a_id=memory_a.id,
                        memory_b_id=memory_b.id,
                        contradiction_type=CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT,
                        confidence=similarity,
                        detection_method="entity_value_extraction",
                        newer_memory_id=newer_id,
                    )
                    self.logger.debug(
                        "Semantic conflict: subject=%r predicate=%r val_a=%r val_b=%r",
                        subj_a,
                        pred_a,
                        val_a,
                        val_b,
                    )
                    return record

        return None

    async def scan_workspace(
        self,
        workspace_id: str,
        batch_size: int = 50,
    ) -> list[ContradictionRecord]:
        """Scan all memories in workspace for contradictions using pairwise comparison.

        For each memory with an embedding, searches for similar memories and checks
        both negation patterns and semantic value conflicts. Skips pairs that already
        have a recorded contradiction.

        Args:
            workspace_id: Workspace to scan
            batch_size: Memories per batch for embedding search

        Returns:
            List of newly created contradiction records
        """
        self.logger.info("Starting workspace contradiction scan for workspace %s", workspace_id)

        # Collect all existing contradiction pairs to avoid duplicates
        existing = await self._storage.get_unresolved_contradictions(workspace_id, limit=10000)
        existing_pairs: set[frozenset] = {frozenset([c.memory_a_id, c.memory_b_id]) for c in existing}

        # Get workspace stats to understand scale
        try:
            stats = await self._storage.get_workspace_stats(workspace_id)
            total_memories = stats.get("total_memories", 0)
        except Exception:
            total_memories = 0

        self.logger.info("Workspace %s has ~%d memories to scan", workspace_id, total_memories)

        new_contradictions: list[ContradictionRecord] = []
        seen_pairs: set[frozenset] = set(existing_pairs)

        # Use recent memories as scan seeds - fetch in batches
        offset = 0
        from datetime import datetime

        # Use a far-back date to get all memories
        epoch = datetime(2000, 1, 1, tzinfo=UTC)

        while True:
            batch = await self._storage.get_recent_memories(
                workspace_id,
                created_after=epoch,
                limit=batch_size,
                detail_level="full",
                offset=offset,
            )
            if not batch:
                break

            offset += len(batch)

            # Convert batch dicts to memory objects if needed
            memory_objects = []
            for item in batch:
                if isinstance(item, dict):
                    # get_recent_memories returns dicts; fetch the full Memory object
                    mem_id = item.get("id")
                    if mem_id:
                        mem = await self._storage.get_memory(workspace_id, mem_id, track_access=False)
                        if mem and mem.embedding:
                            memory_objects.append(mem)
                else:
                    if getattr(item, "embedding", None):
                        memory_objects.append(item)

            for memory in memory_objects:
                if not memory.embedding:
                    continue

                # Search for similar memories to this one
                similar = await self._storage.search_memories(
                    workspace_id,
                    query_embedding=memory.embedding,
                    limit=20,
                    min_relevance=0.7,
                )

                for candidate, relevance in similar:
                    if candidate.id == memory.id:
                        continue

                    pair = frozenset([memory.id, candidate.id])
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # Check negation pattern
                    if self._has_negation_pattern(memory.content, candidate.content):
                        newer_id = self._determine_newer_memory(memory, candidate)
                        record = ContradictionRecord(
                            workspace_id=workspace_id,
                            memory_a_id=memory.id,
                            memory_b_id=candidate.id,
                            contradiction_type=CONTRADICTION_TYPE_NEGATION,
                            confidence=relevance,
                            detection_method="negation_pattern",
                            newer_memory_id=newer_id,
                        )
                        stored = await self._storage.create_contradiction(record)
                        new_contradictions.append(stored)
                        self.logger.info(
                            "Scan found negation contradiction: %s vs %s (%.2f)",
                            memory.id,
                            candidate.id,
                            relevance,
                        )
                        continue

                    # Check semantic value conflict
                    conflict = await self.check_semantic_conflict(memory, candidate)
                    if conflict:
                        conflict.workspace_id = workspace_id
                        stored = await self._storage.create_contradiction(conflict)
                        new_contradictions.append(stored)
                        self.logger.info(
                            "Scan found semantic conflict: %s vs %s (%.2f)",
                            memory.id,
                            candidate.id,
                            relevance,
                        )

            # If batch was smaller than batch_size, we've reached the end
            if len(batch) < batch_size:
                break

        self.logger.info(
            "Workspace scan complete for %s: found %d new contradictions",
            workspace_id,
            len(new_contradictions),
        )
        return new_contradictions


class DefaultContradictionServicePlugin(ContradictionServicePluginBase):
    """Plugin that creates the default contradiction service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> ContradictionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultContradictionService(storage=storage, v=v)
