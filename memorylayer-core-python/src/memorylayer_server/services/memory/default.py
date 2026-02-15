"""
Memory Service - Core business logic for memory operations.

Operations:
- remember: Store new memory with automatic embedding and classification
- recall: Query memories with vector search and optional LLM enhancement
- forget: Soft or hard delete memories
- decay: Reduce memory importance over time
- get: Retrieve single memory by ID
"""
import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger, DEBUG
from typing import Optional, Any, TYPE_CHECKING

from scitrera_app_framework import get_logger, get_extension, Variables

from ...models import RememberInput, Memory, RecallInput, RecallResult, RecallMode, MemoryType, MemoryStatus, SearchTolerance, DetailLevel
from ...utils import compute_content_hash, generate_id

from ..cache import CacheService, EXT_CACHE_SERVICE
from ..contradiction import ContradictionService, EXT_CONTRADICTION_SERVICE
from ..decay import DecayService, EXT_DECAY_SERVICE
from ..deduplication import DeduplicationService, EXT_DEDUPLICATION_SERVICE, DeduplicationAction
from ..extraction import EXT_EXTRACTION_SERVICE, ExtractionService
from ..llm import LLMService, EXT_LLM_SERVICE
from ..storage import StorageBackend, EXT_STORAGE_BACKEND
from ..embedding import EmbeddingService, EXT_EMBEDDING_SERVICE
from ..semantic_tiering import SemanticTieringService, EXT_SEMANTIC_TIERING_SERVICE
from ..reranker import RerankerService, EXT_RERANKER_SERVICE
from .._constants import EXT_TASK_SERVICE

if TYPE_CHECKING:
    from ..tasks import TaskService

from ..association import (
    AssociationService, EXT_ASSOCIATION_SERVICE,
    MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
)

from ...config import DEFAULT_CONTEXT_ID, GLOBAL_WORKSPACE_ID

from .base import (
    MemoryServicePluginBase,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
    DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
    MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
    DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
    MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
    DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
)
from ...config import (
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RECENCY_HALF_LIFE_HOURS,
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)

# Internal constant for LLM recall token budget
_INTERNAL_LLM_RECALL_TOKEN_BUDGET = 2048


@dataclass
class ScopeBoosts:
    """Configuration for locality-based score boosting."""
    same_context: float = 1.5  # 50% boost for same context
    same_workspace: float = 1.2  # 20% boost for same workspace
    global_workspace: float = 1.0  # No boost for global


class MemoryService:
    """
    Core memory service implementing remember/recall/forget operations.

    This service coordinates between:
    - Storage backend (PostgreSQL or SQLite)
    - Embedding service (for vector generation)
    - LLM service (for query rewriting and re-ranking)
    - Cache (for recent memories)
    """

    def __init__(
            self,
            storage: StorageBackend,
            embedding_service: EmbeddingService,
            deduplication_service: DeduplicationService,
            association_service: Optional[AssociationService] = None,
            cache: Optional[Any] = None,
            v: Variables = None,
            tier_generation_service: Optional[SemanticTieringService] = None,
            llm_service: Optional[LLMService] = None,
            reranker_service: Optional[RerankerService] = None,
            decay_service: Optional[DecayService] = None,
            contradiction_service: Optional[ContradictionService] = None,
            task_service: Optional["TaskService"] = None,
            extraction_service: Optional[ExtractionService] = None,
    ):
        self.storage = storage
        self.embedding = embedding_service
        self.deduplication = deduplication_service
        self.association_service = association_service
        self.cache = cache
        self.tier_generation_service = tier_generation_service
        self.llm_service = llm_service
        self.reranker_service = reranker_service
        self.decay_service = decay_service
        self.contradiction_service = contradiction_service
        self.task_service = task_service
        self.extraction_service = extraction_service
        self.v = v
        self.logger = get_logger(v, name=self.__class__.__name__)

        # Get auto-association threshold from config
        self.auto_association_threshold = v.get(
            MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
            DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
        )

        # Fact decomposition config
        self.fact_decomposition_enabled = v.get(
            MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
            DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        )
        self.fact_decomposition_min_length = v.get(
            MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
            DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
        )

        # Recall overfetch multiplier for reranker candidate pool
        self.recall_overfetch = v.get(
            MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
            DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
        )

        # Maximum memories discovered via association graph expansion
        self.max_graph_expansion = v.get(
            MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
            DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
        )

        # Default graph traversal settings for recall
        self.default_include_associations = v.get(
            MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
            DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
        )
        self.default_traverse_depth = v.get(
            MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
            DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
        )

        self.logger.info(
            "Initialized MemoryService (auto_association_threshold=%.2f, fact_decomposition=%s, recall_overfetch=%s, max_graph_expansion=%s, include_associations=%s, traverse_depth=%s)",
            self.auto_association_threshold,
            self.fact_decomposition_enabled,
            self.recall_overfetch,
            self.max_graph_expansion,
            self.default_include_associations,
            self.default_traverse_depth,
        )

    def _recall_cache_key(self, workspace_id: str, query: str, input: RecallInput) -> str:
        """Generate a deterministic cache key for recall results."""
        filter_data = json.dumps({
            "types": [t.value if hasattr(t, 'value') else str(t) for t in (input.types or [])],
            "subtypes": [s.value if hasattr(s, 'value') else str(s) for s in (input.subtypes or [])],
            "tags": sorted(input.tags or []),
            "mode": input.mode.value if input.mode and hasattr(input.mode, 'value') else str(input.mode),
            "tolerance": input.tolerance.value if input.tolerance and hasattr(input.tolerance, 'value') else str(input.tolerance),
            "limit": input.limit,
            "context_id": input.context_id,
            "detail_level": input.detail_level.value if input.detail_level and hasattr(input.detail_level, 'value') else str(
                input.detail_level),
        }, sort_keys=True)
        hash_input = f"{query}|{filter_data}"
        key_hash = compute_content_hash(hash_input)[:16]
        return f"recall:{workspace_id}:{key_hash}"

    # noinspection PyShadowingBuiltins
    async def remember(
            self,
            workspace_id: str,
            input: RememberInput,
            user_id: Optional[str] = None,
            inline: bool = False,
    ) -> Memory:
        """
        Store a new memory.

        Hot path (always synchronous):
        1. Generate content hash and embedding
        2. Check for duplicates (SKIP/UPDATE/MERGE/CREATE)
        3. Classify memory type if not provided
        4. Store in backend

        Post-store (conditional on decomposition):
        - If decomposable: schedule fact decomposition (background by default).
          The decomposition handler owns the per-fact pipeline for each
          extracted fact (dedup, store, associate, contradict, tier-gen).
        - If not decomposable: run post-store pipeline directly
          (cache invalidation, tier gen, contradiction check, auto-association).

        Args:
            workspace_id: Target workspace
            input: Memory content and metadata
            user_id: Optional user ID override
            inline: If True, run all post-store work synchronously including
                    fact decomposition. Default False (background/eventual).
        """
        self.logger.info(
            "Storing memory in workspace: %s, type: %s, content length: %s",
            workspace_id,
            input.type,
            len(input.content),
        )

        # 1. Generate content hash
        content_hash = compute_content_hash(input.content)

        # 2. Generate embedding (needed for deduplication)
        start_time = datetime.now(timezone.utc)
        embedding = await self.embedding.embed(input.content)
        if self.logger.isEnabledFor(DEBUG):
            self.logger.debug(
                "Generated embedding in %s ms",
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

        # 3. Check for duplicates using DeduplicationService
        dedup_result = await self.deduplication.check_duplicate(
            content=input.content,
            content_hash=content_hash,
            embedding=embedding,
            workspace_id=workspace_id
        )

        if dedup_result.action == DeduplicationAction.SKIP:
            # Exact duplicate found, return existing memory
            self.logger.info(
                "Found duplicate memory: %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason
            )
            existing = await self.storage.get_memory(workspace_id, dedup_result.existing_memory_id, track_access=False)
            return existing

        elif dedup_result.action == DeduplicationAction.UPDATE:
            # Semantic duplicate found, update existing memory
            self.logger.info(
                "Updating existing memory: %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason
            )
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=input.content,
                embedding=embedding,
                importance=max(input.importance, 0.5),  # Boost importance on update
            )
            return updated

        elif dedup_result.action == DeduplicationAction.MERGE:
            # Merge candidate - for now, treat as UPDATE
            # TODO: Implement proper merge logic (combine content, metadata, etc.)
            self.logger.info(
                "Merging with existing memory: %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason
            )
            existing = await self.storage.get_memory(workspace_id, dedup_result.existing_memory_id, track_access=False)
            # Simple merge: append new content to existing
            merged_content = f"{existing.content}\n\n---\n\n{input.content}"
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=merged_content,
                embedding=embedding,
                importance=max(existing.importance, input.importance),
            )
            return updated

        # 4. No duplicate - proceed with creating new memory
        self.logger.debug("Creating new memory (%s)", dedup_result.reason)

        # 5. Track whether type was auto-classified (for LLM reclassification later)
        type_was_auto = input.type is None

        # 6. Classify memory type if not provided
        memory_type = input.type
        if memory_type is None:
            memory_type = await self._classify_memory_type(input.content)
            self.logger.debug("Auto-classified memory type: %s", memory_type)

        # 7. Create memory object with generated fields
        memory_data = RememberInput(
            content=input.content,
            type=memory_type,
            subtype=input.subtype,
            importance=input.importance,
            tags=input.tags,
            metadata=input.metadata,
            associations=input.associations,
            context_id=input.context_id,
            user_id=user_id or input.user_id,
        )

        # Store in backend (backend will create Memory object)
        memory = await self.storage.create_memory(workspace_id, memory_data)

        # Update with embedding (if backend doesn't handle it)
        if memory.embedding is None:
            memory = await self.storage.update_memory(
                workspace_id,
                memory.id,
                embedding=embedding,
            )

        self.logger.info("Stored memory: %s", memory.id)

        # Post-store: conditional on decomposition
        # If the memory will be decomposed, only schedule decomposition —
        # the decomposition handler owns the per-fact pipeline.
        # Otherwise, run the post-store pipeline directly on this memory.
        should_decompose = self._should_decompose(memory.content, memory.type)

        if should_decompose:
            if inline:
                await self._decompose_and_process_inline(workspace_id, memory, embedding)
            elif self.task_service:
                try:
                    await self.task_service.schedule_task(
                        'decompose_facts',
                        {'memory_id': memory.id, 'workspace_id': workspace_id},
                    )
                    self.logger.debug("Scheduled fact decomposition for memory %s", memory.id)
                except Exception as e:
                    self.logger.warning(
                        "Failed to schedule decomposition for %s, running post-store on composite: %s",
                        memory.id, e,
                    )
                    # Fallback: run post-store pipeline on composite
                    await self._post_store_pipeline(workspace_id, memory, embedding, inline=False)
            else:
                # No task service and not inline — run post-store on composite as fallback
                await self._post_store_pipeline(workspace_id, memory, embedding, inline=False)
        else:
            await self._post_store_pipeline(workspace_id, memory, embedding, inline=inline, classify_type=type_was_auto)

        return memory

    async def _post_store_pipeline(
            self,
            workspace_id: str,
            memory: Memory,
            embedding: list[float],
            inline: bool = False,
            classify_type: bool = False,
    ) -> None:
        """Run post-store normalization: cache invalidation, association, contradiction, tier gen.

        This is the reusable pipeline that runs after a memory is stored.
        Used by both remember() (for non-decomposable memories) and ingest_fact()
        (for each decomposed fact).

        Args:
            workspace_id: Target workspace
            memory: The stored memory
            embedding: Pre-computed embedding vector
            inline: If True, run all steps synchronously.
                    If False (default), schedule as background tasks where possible.
            classify_type: If True, request LLM-based type reclassification
                    in the auto-enrich task.
        """
        # Cache invalidation (always inline, trivial cost)
        if self.cache:
            try:
                await self.cache.clear_prefix(f"recall:{workspace_id}:")
                await self.cache.clear_prefix(f"assoc:{workspace_id}:")
            except Exception as e:
                self.logger.debug("Cache invalidation failed: %s", e)

        # Tier generation
        if self.tier_generation_service:
            try:
                if inline:
                    await self.tier_generation_service.generate_tiers(memory.id, workspace_id)
                else:
                    await self.tier_generation_service.request_tier_generation(memory.id, workspace_id)
            except Exception as e:
                self.logger.warning("Failed tier generation for %s: %s", memory.id, e)

        # Contradiction check
        if self.contradiction_service:
            try:
                await self.contradiction_service.check_new_memory(workspace_id, memory.id)
            except Exception as e:
                self.logger.warning("Failed contradiction check for %s: %s", memory.id, e)

        # Auto-enrich (association + optional type classification)
        if inline or not self.task_service:
            await self._inline_auto_enrich(workspace_id, memory, embedding, classify_type=classify_type)
        else:
            try:
                await self.task_service.schedule_task('auto_enrich', {
                    'memory_id': memory.id,
                    'workspace_id': workspace_id,
                    'content': memory.content,
                    'embedding': embedding,
                    'classify_type': classify_type,
                })
            except Exception as e:
                self.logger.warning(
                    "Failed to schedule auto-enrich for %s, falling back to inline: %s",
                    memory.id, e,
                )
                await self._inline_auto_enrich(workspace_id, memory, embedding, classify_type=classify_type)

    async def ingest_fact(
            self,
            workspace_id: str,
            input: RememberInput,
            embedding: list[float] | None = None,
            source_memory_id: str | None = None,
            inline: bool = False,
    ) -> Memory | None:
        """Process a single memory through the full pipeline: dedup, store, post-store.

        This is the isolated per-fact pipeline. Called by:
        - FactDecompositionTaskHandler for each decomposed fact (background)
        - _decompose_and_process_inline() for inline decomposition
        - Any future scenario needing atomic memory ingestion with full treatment

        Args:
            workspace_id: Target workspace
            input: Memory content and metadata
            embedding: Pre-computed embedding (generated if None)
            source_memory_id: Parent memory ID (for decomposed facts)
            inline: If True, run post-store pipeline synchronously

        Returns:
            The stored Memory, or None if deduplicated away (SKIP)
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embedding.embed(input.content)

        # Dedup check
        content_hash = compute_content_hash(input.content)
        dedup_result = await self.deduplication.check_duplicate(
            content=input.content,
            content_hash=content_hash,
            embedding=embedding,
            workspace_id=workspace_id,
        )

        if dedup_result.action == DeduplicationAction.SKIP:
            self.logger.info(
                "Fact is duplicate (SKIP): %s (%s)",
                dedup_result.existing_memory_id, dedup_result.reason,
            )
            return None

        if dedup_result.action == DeduplicationAction.UPDATE:
            self.logger.info(
                "Fact updates existing memory: %s (%s)",
                dedup_result.existing_memory_id, dedup_result.reason,
            )
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=input.content,
                embedding=embedding,
                importance=max(input.importance, 0.5),
            )
            await self._post_store_pipeline(workspace_id, updated, embedding, inline=inline)
            return updated

        if dedup_result.action == DeduplicationAction.MERGE:
            self.logger.info(
                "Fact merges with existing memory: %s (%s)",
                dedup_result.existing_memory_id, dedup_result.reason,
            )
            existing = await self.storage.get_memory(
                workspace_id, dedup_result.existing_memory_id, track_access=False,
            )
            merged_content = f"{existing.content}\n\n---\n\n{input.content}"
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=merged_content,
                embedding=embedding,
                importance=max(existing.importance, input.importance),
            )
            await self._post_store_pipeline(workspace_id, updated, embedding, inline=inline)
            return updated

        # CREATE: store new memory
        memory_type = input.type
        if memory_type is None:
            memory_type = await self._classify_memory_type(input.content)

        memory_data = RememberInput(
            content=input.content,
            type=memory_type,
            subtype=input.subtype,
            importance=input.importance,
            tags=input.tags,
            metadata=input.metadata,
            associations=input.associations,
            context_id=input.context_id,
            user_id=input.user_id,
        )

        memory = await self.storage.create_memory(workspace_id, memory_data)

        if memory.embedding is None:
            memory = await self.storage.update_memory(
                workspace_id, memory.id, embedding=embedding,
            )

        # Set source_memory_id if this fact came from decomposition
        if source_memory_id:
            memory = await self.storage.update_memory(
                workspace_id, memory.id, source_memory_id=source_memory_id,
            )

        self.logger.info("Stored fact memory: %s", memory.id)

        # Run post-store pipeline (no decomposition — facts are atomic)
        await self._post_store_pipeline(workspace_id, memory, embedding, inline=inline)

        return memory

    async def _decompose_and_process_inline(
            self,
            workspace_id: str,
            memory: Memory,
            embedding: list[float],
    ) -> list[Memory]:
        """Decompose a composite memory and process each fact inline.

        Used when remember(inline=True) and the memory qualifies for decomposition.
        Runs the full per-fact pipeline (dedup, store, associate, etc.) synchronously
        for each decomposed fact, then archives the parent.

        Args:
            workspace_id: Target workspace
            memory: The composite memory to decompose
            embedding: Pre-computed embedding of the composite

        Returns:
            List of created fact memories
        """
        facts = await self.extraction_service.decompose_to_facts(memory.content)

        if len(facts) <= 1:
            # Already atomic — just run post-store pipeline on the original
            await self._post_store_pipeline(workspace_id, memory, embedding, inline=True)
            return [memory]

        from ...models.association import AssociateInput

        created = []
        for fact in facts:
            # Determine type/subtype: prefer fact-level overrides, fall back to parent
            fact_type = memory.type
            fact_subtype = memory.subtype
            try:
                if fact.get("type"):
                    fact_type = MemoryType(fact["type"])
            except ValueError:
                pass
            try:
                from ...models import MemorySubtype
                if fact.get("subtype"):
                    fact_subtype = MemorySubtype(fact["subtype"])
            except (ValueError, ImportError):
                pass

            fact_input = RememberInput(
                content=fact["content"],
                type=fact_type,
                subtype=fact_subtype,
                importance=memory.importance,
                tags=memory.tags,
                metadata={**(memory.metadata or {}), "decomposed_from": memory.id},
                context_id=memory.context_id,
                user_id=memory.user_id,
            )
            result = await self.ingest_fact(
                workspace_id, fact_input,
                source_memory_id=memory.id,
                inline=True,
            )
            if result:
                created.append(result)

        # Create PART_OF associations from each fact to the parent
        for fact_mem in created:
            try:
                assoc_input = AssociateInput(
                    source_id=fact_mem.id,
                    target_id=memory.id,
                    relationship="part_of",
                    strength=1.0,
                    metadata={"auto_generated": True, "source": "fact_decomposition"},
                )
                await self.storage.create_association(workspace_id, assoc_input)
            except Exception as e:
                self.logger.warning(
                    "Failed PART_OF association %s->%s: %s",
                    fact_mem.id, memory.id, e,
                )

        # Archive the parent memory
        try:
            await self.storage.update_memory(
                workspace_id, memory.id, status=MemoryStatus.ARCHIVED.value,
            )
            self.logger.info(
                "Decomposed memory %s into %d facts inline, archived parent",
                memory.id, len(created),
            )
        except Exception as e:
            self.logger.warning("Failed to archive parent memory %s: %s", memory.id, e)

        return created

    async def _inline_auto_enrich(
            self,
            workspace_id: str,
            memory: Memory,
            embedding: list[float],
            classify_type: bool = False,
    ) -> None:
        """
        Fallback inline auto-enrich: association + optional type classification.

        Used when the task service is not available to schedule background
        auto-enrich with LLM classification.
        """
        if self.association_service:
            try:
                similar_memories = await self.storage.search_memories(
                    workspace_id=workspace_id,
                    query_embedding=embedding,
                    limit=5,
                    min_relevance=self.auto_association_threshold,
                )
            except Exception as e:
                self.logger.warning("Failed to search similar memories for %s: %s", memory.id, e)
                similar_memories = []

            for similar_memory, score in (similar_memories or []):
                if similar_memory.id != memory.id:  # Don't self-associate
                    try:
                        await self.association_service.auto_associate(
                            workspace_id=workspace_id,
                            new_memory_id=memory.id,
                            similar_memories=[(similar_memory.id, score)],
                            threshold=self.auto_association_threshold,
                        )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to auto-associate %s with %s: %s",
                            memory.id, similar_memory.id, e,
                        )

        # Type classification
        if classify_type and self.extraction_service:
            try:
                classified_type, classified_subtype = await self.extraction_service.classify_content(
                    memory.content,
                )
                if classified_type != memory.type:
                    update_kwargs: dict = {'type': classified_type.value}
                    if classified_subtype is not None:
                        update_kwargs['subtype'] = classified_subtype.value
                    await self.storage.update_memory(
                        workspace_id=workspace_id,
                        memory_id=memory.id,
                        **update_kwargs,
                    )
                    self.logger.info(
                        "Reclassified memory %s from %s to %s",
                        memory.id, memory.type, classified_type,
                    )
            except Exception as e:
                self.logger.debug("Inline type classification skipped for %s: %s", memory.id, e)

    async def recall(
            self,
            workspace_id: str,
            input: RecallInput,
            user_id: Optional[str] = None,
    ) -> RecallResult:
        """
        Query memories using vector similarity and optional filters.

        Modes:
        - RAG: Pure vector similarity (fast, ~30ms)
        - LLM: Query rewriting + tiered search (accurate, ~500ms)
        - HYBRID: RAG first, LLM if insufficient (balanced)
        """
        self.logger.info(
            "Recalling memories in workspace: %s, mode: %s, query: %s",
            workspace_id,
            input.mode,
            input.query[:50]
        )

        start_time = datetime.now(timezone.utc)

        # Resolve None → server defaults for mode, tolerance, and detail_level
        effective_mode = input.mode if input.mode is not None else RecallMode.RAG
        effective_tolerance = input.tolerance if input.tolerance is not None else SearchTolerance.MODERATE
        effective_detail_level = input.detail_level if input.detail_level is not None else DetailLevel.FULL

        # Determine effective tolerance threshold
        relevance_threshold = self._get_relevance_threshold(effective_tolerance, input.min_relevance)

        # Phase 4: Check recall cache
        cache_key = None
        if self.cache:
            cache_key = self._recall_cache_key(workspace_id, input.query, input)
            try:
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    self.logger.debug("Recall cache hit for key: %s", cache_key)
                    return RecallResult(**cached)
            except Exception as e:
                self.logger.debug("Cache get failed: %s", e)

        # RAG mode: Pass 1 - Pure vector similarity
        if effective_mode == RecallMode.RAG:
            result = await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
            )
            result.mode_used = RecallMode.RAG

        # LLM mode: Query rewriting + enhanced search
        elif effective_mode == RecallMode.LLM:
            result = await self._recall_llm(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
            )
            result.mode_used = RecallMode.LLM

        # HYBRID mode: Try RAG first, fall back to LLM if insufficient
        else:
            result = await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
            )

            # Check if RAG results are sufficient
            if not result.memories or (result.memories and result.memories[0].importance < input.rag_threshold):
                self.logger.debug("RAG insufficient, trying LLM mode")

                result = await self._recall_llm(
                    workspace_id=workspace_id,
                    input=input,
                    relevance_threshold=relevance_threshold,
                )
                result.mode_used = RecallMode.LLM
            else:
                result.mode_used = RecallMode.RAG

        # Calculate search-only latency (vector/LLM search phase)
        search_latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        result.search_latency_ms = search_latency_ms

        # Resolve None → server defaults for graph traversal
        effective_include_associations = (
            input.include_associations if input.include_associations is not None
            else self.default_include_associations
        )
        effective_traverse_depth = (
            input.traverse_depth if input.traverse_depth is not None
            else self.default_traverse_depth
        )
        effective_max_expansion = (
            input.max_expansion if input.max_expansion is not None
            else self.max_graph_expansion
        )

        # Association expansion (Phase 3A)
        assoc_ms = 0
        if effective_include_associations or effective_traverse_depth > 0:
            t0 = datetime.now(timezone.utc)
            result.memories = await self._expand_with_associations(
                workspace_id=workspace_id,
                memories=result.memories,
                traverse_depth=effective_traverse_depth,
                include_associations=effective_include_associations,
                max_expansion=effective_max_expansion,
            )
            assoc_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

        # Reranking across all modes (Phase 3B)
        # Skip reranking for wildcard/trivial queries where ranking is meaningless
        rerank_ms = 0
        trivial_query = input.query.strip() in ("*", "", "**")
        if self.reranker_service and len(result.memories) > input.limit and not trivial_query:
            t0 = datetime.now(timezone.utc)
            result.memories = await self._apply_reranking(
                query=input.query,
                memories=result.memories,
                limit=input.limit,
            )
            rerank_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        elif len(result.memories) > input.limit:
            # Truncate without reranking (trivial query or no reranker)
            result.memories = result.memories[:input.limit]

        # Apply detail_level filtering if requested
        detail_ms = 0
        if effective_detail_level != DetailLevel.FULL:
            t0 = datetime.now(timezone.utc)
            filtered_memories = self._apply_detail_level(
                result.memories,
                effective_detail_level
            )
            result.memories = filtered_memories
            detail_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

        # Increment access counts (and boost importance) in parallel
        access_ms = 0
        if result.memories:
            t0 = datetime.now(timezone.utc)
            access_tasks = [self.increment_access(workspace_id, m.id) for m in result.memories]
            await asyncio.gather(*access_tasks, return_exceptions=True)
            access_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

        # Calculate total latency
        total_latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        self.logger.info(
            "Recalled %s memories in %s ms "
            "(search: %s ms, associations: %s ms, rerank: %s ms, detail_filter: %s ms, access_tracking: %s ms) "
            "using %s mode (detail_level: %s)",
            len(result.memories),
            total_latency_ms,
            search_latency_ms,
            assoc_ms,
            rerank_ms,
            detail_ms,
            access_ms,
            result.mode_used,
            effective_detail_level.value
        )

        # Phase 4: Cache recall result
        if self.cache and cache_key:
            try:
                await self.cache.set(cache_key, result.model_dump(), ttl_seconds=300)
            except Exception as e:
                self.logger.debug("Cache set failed: %s", e)

        return result

    async def _recall_rag(
            self,
            workspace_id: str,
            input: RecallInput,
            relevance_threshold: float,
    ) -> RecallResult:
        """Pure vector similarity search."""
        # Generate query embedding
        query_embedding = await self.embedding.embed(input.query)

        # Overfetch to give the reranker a larger candidate pool
        overfetch_limit = input.limit * self.recall_overfetch

        # Search memories in current workspace
        include_archived = getattr(input, 'include_archived', False)
        results = await self.storage.search_memories(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=overfetch_limit,
            offset=input.offset,
            min_relevance=relevance_threshold,
            types=[t.value for t in input.types] if input.types else None,
            subtypes=[s.value for s in input.subtypes] if input.subtypes else None,
            tags=input.tags if input.tags else None,
            include_archived=include_archived,
        )

        # Search _global workspace if enabled and not already searching it
        global_results = []
        if input.include_global and workspace_id != GLOBAL_WORKSPACE_ID:
            global_results = await self.storage.search_memories(
                workspace_id=GLOBAL_WORKSPACE_ID,
                query_embedding=query_embedding,
                limit=overfetch_limit,
                offset=input.offset,
                min_relevance=relevance_threshold,
                types=[t.value for t in input.types] if input.types else None,
                subtypes=[s.value for s in input.subtypes] if input.subtypes else None,
                tags=input.tags if input.tags else None,
                include_archived=include_archived,
            )

        # Combine results
        all_results = results + global_results

        # Apply scope boosts
        context_id = input.context_id if input.context_id else DEFAULT_CONTEXT_ID
        boosted_memories = self.apply_scope_boosts(
            all_results,
            query_context_id=context_id,
            query_workspace_id=workspace_id,
            boosts=None  # Use default boosts
        )

        # Apply recency boost
        effective_recency_weight = input.recency_weight if input.recency_weight is not None else DEFAULT_RECENCY_WEIGHT
        boosted_memories = self.apply_recency_boost(
            boosted_memories,
            recency_weight=effective_recency_weight,
        )

        # Take top limit results after boosting
        memories = boosted_memories[:input.limit]

        return RecallResult(
            memories=memories,
            total_count=len(all_results),
            query_tokens=0,
            search_latency_ms=0,  # Will be set by caller
            mode_used=RecallMode.RAG,
        )

    async def _recall_llm(
            self,
            workspace_id: str,
            input: RecallInput,
            relevance_threshold: float,
    ) -> RecallResult:
        """
        LLM-enhanced retrieval with query rewriting.

        Steps:
        1. Use LLM to rewrite query for better semantic match
        2. Perform RAG search with rewritten query

        Note: Re-ranking is handled at the top level in recall() for all modes.
        """
        import time
        start_time = time.time()

        # Check if LLM service is available
        if not self.llm_service:
            self.logger.warning("LLM service not available, falling back to RAG")
            return await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
            )

        # Step 1: LLM Query Rewriting
        # TODO: configurable LLM query rewriting
        # rewritten_query = await self._rewrite_query_with_llm(input.query, input.context)
        rewritten_query = input.query

        # self.logger.info(
        #     "LLM query rewrite: '%s' -> '%s'",
        #     input.query[:50],
        #     rewritten_query[:50] if rewritten_query != input.query else "(unchanged)"
        # )

        # Step 2: Search with rewritten query (fetch more candidates for re-ranking)
        search_input = RecallInput(
            query=rewritten_query,
            types=input.types,
            subtypes=input.subtypes,
            tags=input.tags,
            context_id=input.context_id,
            mode=RecallMode.RAG,  # Use RAG for initial retrieval
            tolerance=input.tolerance,
            limit=min(input.limit * self.recall_overfetch, 50),  # Overfetch for reranker
            min_relevance=max(0.2, relevance_threshold - 0.3),  # Lower threshold for candidates
            include_associations=input.include_associations,
            traverse_depth=input.traverse_depth,
            created_after=input.created_after,
            created_before=input.created_before,
            context=input.context,
            rag_threshold=input.rag_threshold,
            detail_level=input.detail_level,
        )

        rag_result = await self._recall_rag(
            workspace_id=workspace_id,
            input=search_input,
            relevance_threshold=max(0.2, relevance_threshold - 0.3),
        )

        if not rag_result.memories:
            # No candidates found
            return RecallResult(
                memories=[],
                total_count=0,
                search_latency_ms=int((time.time() - start_time) * 1000),
                mode_used=RecallMode.LLM,
                query_rewritten=rewritten_query,
                sufficiency_reached=False
            )

        # Reranking is now handled at the top level in recall() for all modes
        search_latency_ms = int((time.time() - start_time) * 1000)

        self.logger.info(
            "LLM recall complete: %d candidates in %d ms",
            len(rag_result.memories),
            search_latency_ms
        )

        return RecallResult(
            memories=rag_result.memories,
            total_count=len(rag_result.memories),
            search_latency_ms=search_latency_ms,
            mode_used=RecallMode.LLM,
            query_rewritten=rewritten_query,
            sufficiency_reached=len(rag_result.memories) >= input.limit
        )

    async def _rewrite_query_with_llm(
            self,
            query: str,
            context: Optional[str] = None
    ) -> str:
        """
        Use LLM to rewrite query for better semantic search.

        Expands abbreviations, adds synonyms, clarifies intent.
        """
        prompt = f"""Rewrite the following search query to improve semantic search results.
Expand abbreviations, add relevant synonyms, and clarify the intent.
Keep it concise (under 100 words). Return ONLY the rewritten query, no explanation.

Original query: {query}"""

        if context:
            prompt += f"\n\nContext: {context}"

        try:
            rewritten = await self.llm_service.synthesize(
                prompt=prompt,
                max_tokens=_INTERNAL_LLM_RECALL_TOKEN_BUDGET,
                temperature_factor=0.4,  # Low temperature for consistency
                profile="default",
            )
            # Clean up response
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as e:
            self.logger.warning("Query rewriting failed: %s, using original", e)
            return query

    async def _rerank_with_llm(
            self,
            query: str,
            memories: list,
            limit: int
    ) -> list:
        """
        Re-rank memories by relevance to query.

        Uses dedicated RerankerService if available, falls back to LLM-based reranking.
        Returns top-k most relevant memories.
        """
        if len(memories) <= limit:
            return memories

        # Try dedicated reranker service first (faster, more accurate)
        if self.reranker_service:
            try:
                # Get initial scores for adaptive sizing
                initial_scores = [getattr(mem, 'relevance', 0.5) for mem in memories]

                # Use adaptive reranking
                results = await self.reranker_service.rerank_objects_adaptive(
                    query=query,
                    objects=memories,
                    content_fn=lambda m: m.content,
                    score_fn=lambda m: getattr(m, 'relevance', 0.5),
                    requested_k=limit,
                )

                if results:
                    self.logger.debug(
                        "Reranker service: %d candidates -> %d results",
                        len(memories),
                        len(results)
                    )
                    return [r.document for r in results]

            except Exception as e:
                self.logger.warning("Reranker service failed: %s, falling back to LLM", e)

        # Fall back to LLM-based reranking
        if not self.llm_service:
            self.logger.warning("No reranker or LLM service available, using original order")
            return memories[:limit]

        # Build context with memory summaries
        memory_summaries = []
        for i, mem in enumerate(memories[:20]):  # Limit to top 20 for LLM context
            content_preview = mem.content[:200] if len(mem.content) > 200 else mem.content
            memory_summaries.append(f"[{i}] {content_preview}")

        summaries_text = "\n".join(memory_summaries)

        prompt = f"""Given the search query and candidate memories, rank them by relevance.
Return ONLY a comma-separated list of indices (e.g., "3,0,5,2") for the {limit} most relevant memories.
Most relevant first. No explanation.

Query: {query}

Candidate memories:
{summaries_text}

Top {limit} indices (comma-separated):"""

        try:
            response = await self.llm_service.synthesize(
                prompt=prompt,
                max_tokens=50,
                temperature_factor=0.15,  # Very low for deterministic ranking
                profile="default",
            )

            # Parse indices from response
            indices = []
            for part in response.strip().split(","):
                try:
                    idx = int(part.strip().strip("[]"))
                    if 0 <= idx < len(memories):
                        indices.append(idx)
                except ValueError:
                    continue

            if indices:
                # Return memories in ranked order
                ranked = [memories[i] for i in indices[:limit]]
                # Fill remaining slots if needed
                if len(ranked) < limit:
                    remaining = [m for i, m in enumerate(memories) if i not in indices]
                    ranked.extend(remaining[:limit - len(ranked)])
                return ranked

        except Exception as e:
            self.logger.warning("LLM re-ranking failed: %s, using original order", e)

        # Fallback to original order
        return memories[:limit]

    async def _apply_reranking(
            self,
            query: str,
            memories: list[Memory],
            limit: int,
    ) -> list[Memory]:
        """Apply reranking to memories using the reranker service.

        Uses reranker_service if available, falls back to truncation.

        Args:
            query: The original search query
            memories: Memories to rerank
            limit: Maximum number to return

        Returns:
            Reranked and truncated list of memories
        """
        if not self.reranker_service or len(memories) <= limit:
            return memories[:limit]

        try:
            reranked = await self.reranker_service.rerank_objects_adaptive(
                query=query,
                objects=memories,
                content_fn=lambda m: m.content,
                score_fn=lambda m: getattr(m, 'boosted_score', None) or getattr(m, 'relevance_score', 0.5),
                requested_k=limit,
            )
            if reranked:
                return [r.document for r in reranked]
            return memories[:limit]
        except Exception as e:
            self.logger.warning("Reranking failed, falling back to truncation: %s", e)
            return memories[:limit]

    async def forget(
            self,
            workspace_id: str,
            memory_id: str,
            hard: bool = False,
            reason: Optional[str] = None,
    ) -> bool:
        """
        Delete or soft-delete a memory.

        Soft delete: Sets deleted_at timestamp
        Hard delete: Removes from database entirely
        """
        self.logger.info(
            "Forgetting memory: %s in workspace: %s, hard: %s",
            memory_id,
            workspace_id,
            hard
        )

        success = await self.storage.delete_memory(
            workspace_id=workspace_id,
            memory_id=memory_id,
            hard=hard
        )

        if success:
            self.logger.info("Memory forgotten: %s", memory_id)
        else:
            self.logger.warning("Failed to forget memory: %s", memory_id)

        return success

    async def decay(
            self,
            workspace_id: str,
            memory_id: str,
            decay_rate: float = 0.1,
    ) -> Optional[Memory]:
        """
        Reduce memory importance by decay_rate.

        Used for implementing memory decay over time.
        """
        self.logger.debug(
            "Decaying memory: %s by rate: %s",
            memory_id,
            decay_rate
        )

        # Get current memory
        memory = await self.storage.get_memory(workspace_id, memory_id)
        if not memory:
            self.logger.warning("Memory not found for decay: %s", memory_id)
            return None

        # Calculate new importance (apply decay directly)
        new_importance = max(0.0, memory.importance - decay_rate)

        # Update memory
        updated = await self.storage.update_memory(
            workspace_id=workspace_id,
            memory_id=memory_id,
            importance=new_importance
        )

        self.logger.debug(
            "Decayed memory: %s, new importance: %s",
            memory_id,
            new_importance
        )

        return updated

    async def get(
            self,
            workspace_id: str,
            memory_id: str,
    ) -> Optional[Memory]:
        """Get a single memory by ID within a workspace."""
        self.logger.debug("Getting memory: %s in workspace: %s", memory_id, workspace_id)
        return await self.storage.get_memory(workspace_id, memory_id)

    async def get_by_id(
            self,
            memory_id: str,
    ) -> Optional[Memory]:
        """Get a single memory by ID without workspace filter. Memory IDs are globally unique."""
        self.logger.debug("Getting memory by ID: %s", memory_id)
        return await self.storage.get_memory_by_id(memory_id)

    async def increment_access(
            self,
            workspace_id: str,
            memory_id: str,
    ) -> None:
        """Increment access count and update last_accessed_at."""
        try:
            memory = await self.storage.get_memory(workspace_id, memory_id)
            if memory:
                importance = await self.decay_service.calculate_access_boost(memory)
                await self.storage.update_memory(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    access_count=memory.access_count + 1,
                    last_accessed_at=datetime.now(timezone.utc),
                    importance=importance,
                )
        except Exception as e:
            self.logger.warning("Failed to increment access for %s: %s", memory_id, e)

    def _apply_detail_level(self, memories: list[Memory], detail_level: DetailLevel) -> list[Memory]:
        """
        Apply detail level filtering to memories.

        Args:
            memories: List of Memory objects to filter
            detail_level: Level of detail to return

        Returns:
            List of filtered memories
        """
        if detail_level == DetailLevel.FULL:
            # No filtering needed for full detail
            return memories

        filtered_memories = []

        for memory in memories:
            # Create a copy of the memory with modified content
            memory_dict = memory.model_dump()

            if detail_level == DetailLevel.ABSTRACT:
                # Use abstract field if available, else truncate to ~100 chars
                if memory.abstract:
                    memory_dict['content'] = memory.abstract
                else:
                    memory_dict['content'] = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content

            elif detail_level == DetailLevel.OVERVIEW:
                # Use overview field if available, else truncate to ~500 chars
                if memory.overview:
                    memory_dict['content'] = memory.overview
                else:
                    memory_dict['content'] = memory.content[:500] + "..." if len(memory.content) > 500 else memory.content

            filtered_memories.append(Memory(**memory_dict))

        return filtered_memories

    async def _expand_with_associations(
            self,
            workspace_id: str,
            memories: list[Memory],
            traverse_depth: int,
            include_associations: bool,
            max_expansion: int = 50,
    ) -> list[Memory]:
        """Expand recall results by traversing association graph.

        BFS-traverses associations up to traverse_depth hops from recalled memories.
        Discovered memories are scored relative to their parent:
          parent_score * association_strength * (0.8 ^ depth)

        Args:
            workspace_id: Workspace boundary
            memories: Initially recalled memories
            traverse_depth: Maximum BFS depth (0 = no expansion)
            include_associations: Whether to include associated memories
            max_expansion: Maximum number of graph-discovered memories to add

        Returns:
            Combined list of original + discovered memories, sorted by score descending
        """
        if not include_associations and traverse_depth <= 0:
            return memories

        effective_depth = max(traverse_depth, 1) if include_associations else traverse_depth

        # Phase 4: Check association expansion cache
        assoc_cache_key = None
        if self.cache and memories:
            # Cache key based on seed memory IDs and depth
            seed_ids = sorted(m.id for m in memories)
            assoc_cache_key = f"assoc:{workspace_id}:{compute_content_hash(':'.join(seed_ids))[:16]}:{effective_depth}"
            try:
                cached = await self.cache.get(assoc_cache_key)
                if cached is not None:
                    self.logger.debug("Association expansion cache hit")
                    return [Memory(**m) for m in cached]
            except Exception as e:
                self.logger.debug("Association cache get failed: %s", e)

        # Track already-seen memory IDs to avoid duplicates
        seen_ids: set[str] = {m.id for m in memories}
        discovered: list[tuple[Memory, float]] = []

        # BFS queue: (memory_id, parent_score, current_depth)
        queue: list[tuple[str, float, int]] = []
        for memory in memories:
            parent_score = getattr(memory, 'boosted_score', None) or getattr(memory, 'relevance_score', 0.5)
            queue.append((memory.id, parent_score, 0))

        while queue:
            if len(discovered) >= max_expansion:
                self.logger.debug(
                    "Graph expansion cap reached (%d memories), stopping BFS",
                    max_expansion,
                )
                break

            memory_id, parent_score, depth = queue.pop(0)

            if depth >= effective_depth:
                continue

            try:
                associations = await self.storage.get_associations(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    direction="both",
                )
            except Exception as e:
                self.logger.warning("Failed to get associations for %s: %s", memory_id, e)
                continue

            for assoc in associations:
                # Determine the other end of the association
                target_id = assoc.target_id if assoc.source_id == memory_id else assoc.source_id

                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                try:
                    target_memory = await self.storage.get_memory(workspace_id, target_id)
                except Exception:
                    continue

                if target_memory is None:
                    continue

                # Skip non-active memories
                if hasattr(target_memory, 'status') and target_memory.status != MemoryStatus.ACTIVE:
                    continue

                # Score: parent_score * strength * decay_per_hop
                hop_decay = 0.8 ** (depth + 1)
                score = parent_score * assoc.strength * hop_decay

                # Attach score metadata
                memory_dict = target_memory.model_dump()
                memory_dict['relevance_score'] = score
                memory_dict['boosted_score'] = score
                memory_dict['source_scope'] = 'association'
                scored_memory = Memory(**memory_dict)

                discovered.append((scored_memory, score))

                # Continue BFS from this memory
                queue.append((target_id, score, depth + 1))

        if not discovered:
            return memories

        # Combine original memories with discovered ones
        combined = list(memories)
        combined.extend([m for m, _ in discovered])

        # Sort by boosted_score descending
        combined.sort(
            key=lambda m: getattr(m, 'boosted_score', 0.0) or 0.0,
            reverse=True,
        )

        # Phase 4: Cache the expanded result
        if self.cache and assoc_cache_key:
            try:
                await self.cache.set(
                    assoc_cache_key,
                    [m.model_dump() for m in combined],
                    ttl_seconds=600,
                )
            except Exception as e:
                self.logger.debug("Association cache set failed: %s", e)

        return combined

    def _should_decompose(self, content: str, memory_type: Optional[MemoryType]) -> bool:
        """Determine whether a memory should be decomposed into atomic facts.

        Criteria:
        - Fact decomposition must be enabled
        - Content length must exceed the configured minimum
        - Content must contain multiple sentences (>1 period or semicolon)
        - Memory must not be a working type (working memories are transient)

        Args:
            content: The memory content
            memory_type: The memory type (if known)

        Returns:
            True if the memory should be decomposed
        """
        if not self.fact_decomposition_enabled:
            return False

        if memory_type == MemoryType.WORKING:
            return False

        if len(content) < self.fact_decomposition_min_length:
            return False

        # Check for multiple sentences (periods, semicolons, or question marks followed by space/end)
        sentence_terminators = re.findall(r'[.;?!]\s', content)
        # Also check for a terminator at the very end of the string
        if content and content[-1] in '.;?!':
            sentence_terminators.append(content[-1])
        if len(sentence_terminators) <= 1:
            return False

        return True

    async def _classify_memory_type(self, content: str) -> MemoryType:
        """
        Auto-classify memory type based on content.

        Simple heuristic-based classification.
        In production, could use LLM for more accurate classification.
        """
        # TODO: use LLM service (or another form of classifier) for more accurate classification
        content_lower = content.lower()

        # Procedural: How-to, steps, instructions
        if any(keyword in content_lower for keyword in [
            "how to", "steps", "procedure", "process", "method", "workflow"
        ]):
            return MemoryType.PROCEDURAL

        # Episodic: Time-based, events, specific instances
        if any(keyword in content_lower for keyword in [
            "when", "yesterday", "today", "occurred", "happened", "at that time"
        ]):
            return MemoryType.EPISODIC

        # Working: Current context, temporary
        if any(keyword in content_lower for keyword in [
            "currently", "working on", "in progress", "now", "right now"
        ]):
            return MemoryType.WORKING

        # Default to semantic (facts, concepts)
        return MemoryType.SEMANTIC

    def _get_relevance_threshold(
            self,
            tolerance: SearchTolerance,
            min_relevance: Optional[float]
    ) -> float:
        """
        Calculate effective relevance threshold.

        Priority:
        1. min_relevance is None: use tolerance-based floor (server default)
        2. min_relevance <= 0.0: bypass all thresholds (testing mode)
        3. Explicit value: respect caller's choice, applying tolerance floor as minimum
        """
        # TODO: Tolerance threshold values should be server-configurable
        # rather than hardcoded. Move these to service configuration so
        # operators can tune search sensitivity per deployment.
        tolerance_floors = {
            SearchTolerance.STRICT: 0.6,
            SearchTolerance.MODERATE: 0.3,
            SearchTolerance.LOOSE: 0.15,
        }
        floor = tolerance_floors.get(tolerance, tolerance_floors[SearchTolerance.MODERATE])

        # No explicit value: use tolerance-based floor as the server default
        if min_relevance is None:
            return floor

        # Testing mode: bypass all thresholds
        if min_relevance <= 0.0:
            return min_relevance

        # Explicit value: respect caller's choice, but enforce tolerance floor
        return max(min_relevance, floor)

    def apply_scope_boosts(
            self,
            memories: list,
            query_context_id: str,
            query_workspace_id: str,
            boosts: Optional[ScopeBoosts] = None
    ) -> list[Memory]:
        """
        Apply locality-based score boosts to recalled memories.

        Args:
            memories: List of (memory, score) tuples from storage
            query_context_id: The context the query originated from
            query_workspace_id: The workspace the query originated from
            boosts: ScopeBoosts configuration (uses defaults if None)

        Returns:
            List of Memory objects sorted by boosted score with source_scope added
        """
        if boosts is None:
            boosts = ScopeBoosts()

        boosted_memories = []

        for memory, base_score in memories:
            # Determine scope and boost
            memory_context_id = memory.context_id if memory.context_id else DEFAULT_CONTEXT_ID
            memory_workspace_id = memory.workspace_id

            if memory_context_id == query_context_id:
                source_scope = "same_context"
                boost = boosts.same_context
            elif memory_workspace_id == query_workspace_id:
                source_scope = "same_workspace"
                boost = boosts.same_workspace
            elif memory_workspace_id == GLOBAL_WORKSPACE_ID:
                source_scope = "global_workspace"
                boost = boosts.global_workspace
            else:
                source_scope = "other"
                boost = 1.0

            boosted_score = base_score * boost

            # Create new Memory object with ranking metadata
            memory_dict = memory.model_dump()
            memory_dict['source_scope'] = source_scope
            memory_dict['relevance_score'] = base_score
            memory_dict['boosted_score'] = boosted_score

            boosted_memory = Memory(**memory_dict)
            boosted_memories.append((boosted_memory, boosted_score))

        # Sort by boosted score descending
        boosted_memories.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in boosted_memories]

    def apply_recency_boost(
            self,
            memories: list[Memory],
            recency_weight: float,
            half_life_hours: float = DEFAULT_RECENCY_HALF_LIFE_HOURS,
    ) -> list[Memory]:
        """
        Apply time-based recency boost to recalled memories.

        Uses exponential decay based on memory's updated_at timestamp.
        Recent memories get higher scores; old memories decay toward
        (1 - recency_weight) of their boosted score.

        Args:
            memories: List of Memory objects with boosted_score already set
            recency_weight: How much recency affects ranking (0.0-1.0)
            half_life_hours: Hours until recency factor reaches 0.5

        Returns:
            List of Memory objects re-sorted by recency-adjusted boosted_score
        """
        if recency_weight <= 0.0 or not memories:
            return memories

        now = datetime.now(timezone.utc)
        decay_lambda = math.log(2) / half_life_hours

        for memory in memories:
            age_hours = max(0.0, (now - memory.updated_at).total_seconds() / 3600.0)
            recency_factor = math.exp(-decay_lambda * age_hours)
            # Blend: at weight=0 no effect, at weight=1 full decay
            adjusted_score = memory.boosted_score * (1.0 - recency_weight + recency_weight * recency_factor)
            memory.boosted_score = adjusted_score

        # Re-sort by adjusted boosted_score
        memories.sort(key=lambda m: m.boosted_score, reverse=True)
        return memories

    async def recall_with_global(
            self,
            workspace_id: str,
            context_id: str,
            query: str,
            include_global: bool = True,
            boosts: Optional[ScopeBoosts] = None,
            **kwargs
    ) -> list[Memory]:
        """
        Recall memories from workspace and optionally _global.

        Args:
            workspace_id: The workspace to search
            context_id: The context the query is from (for boosting)
            query: The search query
            include_global: Whether to include _global workspace
            boosts: ScopeBoosts configuration
            **kwargs: Additional recall parameters (limit, types, etc.)

        Returns:
            Combined and ranked memories with locality boosts applied
        """
        # Build RecallInput from query and kwargs
        recall_input = RecallInput(
            query=query,
            context_id=context_id,
            limit=kwargs.get('limit', 10),
            types=kwargs.get('types', []),
            subtypes=kwargs.get('subtypes', []),
            tags=kwargs.get('tags', []),
            mode=kwargs.get('mode', RecallMode.RAG),
            tolerance=kwargs.get('tolerance', SearchTolerance.MODERATE),
            min_relevance=kwargs.get('min_relevance'),
        )

        # Generate query embedding once
        query_embedding = await self.embedding.embed(query)

        # Get memories from current workspace
        workspace_results = await self.storage.search_memories(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=recall_input.limit,
            min_relevance=self._get_relevance_threshold(recall_input.tolerance, recall_input.min_relevance),
            types=[t.value for t in recall_input.types] if recall_input.types else None,
            subtypes=[s.value for s in recall_input.subtypes] if recall_input.subtypes else None,
            tags=recall_input.tags if recall_input.tags else None,
        )

        # Get memories from _global if enabled
        global_results = []
        if include_global and workspace_id != GLOBAL_WORKSPACE_ID:
            global_results = await self.storage.search_memories(
                workspace_id=GLOBAL_WORKSPACE_ID,
                query_embedding=query_embedding,
                limit=recall_input.limit,
                min_relevance=self._get_relevance_threshold(recall_input.tolerance, recall_input.min_relevance),
                types=[t.value for t in recall_input.types] if recall_input.types else None,
                subtypes=[s.value for s in recall_input.subtypes] if recall_input.subtypes else None,
                tags=recall_input.tags if recall_input.tags else None,
            )

        # Combine results
        all_memories = workspace_results + global_results

        # Apply scope boosts and return sorted
        ranked = self.apply_scope_boosts(
            all_memories,
            query_context_id=context_id,
            query_workspace_id=workspace_id,
            boosts=boosts
        )

        # Apply recency boost
        effective_recency_weight = kwargs.get('recency_weight', DEFAULT_RECENCY_WEIGHT)
        ranked = self.apply_recency_boost(
            ranked,
            recency_weight=effective_recency_weight,
        )

        return ranked


class DefaultMemoryServicePlugin(MemoryServicePluginBase):
    """Default memory service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> MemoryService:
        cache = self.get_extension(EXT_CACHE_SERVICE, v)
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        embedding: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)
        deduplication: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        association_service: AssociationService = self.get_extension(EXT_ASSOCIATION_SERVICE, v)
        tier_generation_service: SemanticTieringService = self.get_extension(EXT_SEMANTIC_TIERING_SERVICE, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        reranker_service: RerankerService = self.get_extension(EXT_RERANKER_SERVICE, v)
        decay_service: DecayService = self.get_extension(EXT_DECAY_SERVICE, v)
        contradiction_service: ContradictionService = self.get_extension(EXT_CONTRADICTION_SERVICE, v)
        extraction_service: ExtractionService = self.get_extension(EXT_EXTRACTION_SERVICE, v)

        # TaskService is optional -- auto-association works inline without it
        task_service: Optional["TaskService"] = None
        try:
            task_service = self.get_extension(EXT_TASK_SERVICE, v)
        except Exception:
            logger.debug("TaskService not available, auto-association will run inline")

        return MemoryService(
            storage=storage,
            embedding_service=embedding,
            deduplication_service=deduplication,
            association_service=association_service,
            cache=cache,
            tier_generation_service=tier_generation_service,
            llm_service=llm_service,
            reranker_service=reranker_service,
            decay_service=decay_service,
            contradiction_service=contradiction_service,
            task_service=task_service,
            extraction_service=extraction_service,
            v=v,
        )
