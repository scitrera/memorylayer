"""
Inference Service - Default LLM-based implementation.

Derives higher-order insights from accumulated memories about entities.
"""
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import InferenceServicePluginBase, InferenceResult
from ..storage import StorageBackend, EXT_STORAGE_BACKEND
from ..memory import MemoryService, EXT_MEMORY_SERVICE
from ..association import AssociationService, EXT_ASSOCIATION_SERVICE
from ..cache import CacheService, EXT_CACHE_SERVICE
from ..llm import LLMService, EXT_LLM_SERVICE, LLMNotConfiguredError
from ...models import (
    Memory, RememberInput, RecallInput, RecallMode,
    MemoryType, MemorySubtype, MemoryStatus,
)

# Cache TTL for insights (15 minutes)
INSIGHTS_CACHE_TTL = 900

DERIVE_SYSTEM_PROMPT = """You are an analytical reasoning system that derives insights about entities from accumulated observations.

Given a collection of memories/observations about an entity, identify higher-order patterns, behavioral tendencies, preferences, and conclusions that are NOT explicitly stated but can be inferred from the evidence.

Rules:
- Each insight must be supported by multiple observations
- Focus on patterns, not individual events
- Be specific and actionable, not vague
- Distinguish between strong patterns (many observations) and tentative ones (few observations)
- Output each insight on its own line, prefixed with importance score (0.0-1.0)

Format each insight as:
[importance] insight text

Example:
[0.8] Prefers concise, direct communication and becomes disengaged with verbose explanations
[0.6] Tends to approach problems methodically, breaking them into smaller sub-tasks before implementation"""


class DefaultInferenceService:
    """LLM-based inference service that derives insights from memory patterns."""

    def __init__(
            self,
            storage: StorageBackend,
            memory_service: MemoryService,
            llm_service: Optional[LLMService] = None,
            association_service: Optional[AssociationService] = None,
            cache_service: Optional[CacheService] = None,
            v: Variables = None,
    ):
        self.storage = storage
        self.memory_service = memory_service
        self.llm = llm_service
        self.association_service = association_service
        self.cache = cache_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized DefaultInferenceService")

    async def derive_insights(
            self,
            workspace_id: str,
            subject_id: str,
            observer_id: Optional[str] = None,
            force: bool = False,
    ) -> InferenceResult:
        """Derive higher-order insights about a subject from accumulated memories."""
        self.logger.info(
            "Deriving insights for subject=%s in workspace=%s (observer=%s, force=%s)",
            subject_id, workspace_id, observer_id, force
        )

        # Check cache unless forced
        if not force and self.cache:
            cache_key = self._cache_key(workspace_id, subject_id, observer_id)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                self.logger.debug("Returning cached inference result for subject=%s", subject_id)
                # Return existing insights from storage
                insights = await self.get_insights(workspace_id, subject_id, observer_id)
                return InferenceResult(
                    subject_id=subject_id,
                    workspace_id=workspace_id,
                    insights_created=0,
                    insights_updated=0,
                    source_memory_count=len(insights),
                    insights=insights,
                )

        # Recall all memories about this subject
        source_memories = await self._gather_subject_memories(
            workspace_id, subject_id, observer_id
        )

        if not source_memories:
            self.logger.info("No memories found for subject=%s, skipping inference", subject_id)
            return InferenceResult(
                subject_id=subject_id,
                workspace_id=workspace_id,
                source_memory_count=0,
            )

        self.logger.debug(
            "Gathered %d source memories for subject=%s", len(source_memories), subject_id
        )

        # Derive insights via LLM
        raw_insights = await self._derive_with_llm(source_memories, subject_id)

        if not raw_insights:
            self.logger.info("No insights derived for subject=%s", subject_id)
            return InferenceResult(
                subject_id=subject_id,
                workspace_id=workspace_id,
                source_memory_count=len(source_memories),
            )

        # Store insights as memories
        created_insights = []
        for importance, insight_text in raw_insights:
            memory = await self._store_insight(
                workspace_id=workspace_id,
                subject_id=subject_id,
                observer_id=observer_id,
                content=insight_text,
                importance=importance,
                source_memory_ids=[m.id for m in source_memories[:10]],
            )
            if memory:
                created_insights.append(memory)

        # Update cache
        if self.cache:
            cache_key = self._cache_key(workspace_id, subject_id, observer_id)
            await self.cache.set(cache_key, True, ttl_seconds=INSIGHTS_CACHE_TTL)

        result = InferenceResult(
            subject_id=subject_id,
            workspace_id=workspace_id,
            insights_created=len(created_insights),
            source_memory_count=len(source_memories),
            insights=created_insights,
        )

        self.logger.info(
            "Derived %d insights for subject=%s from %d source memories",
            len(created_insights), subject_id, len(source_memories)
        )

        return result

    async def get_insights(
            self,
            workspace_id: str,
            subject_id: str,
            observer_id: Optional[str] = None,
            limit: int = 20,
    ) -> list[Memory]:
        """Retrieve existing derived insights about a subject."""
        # Search for INFERENCE-subtype memories about this subject
        recall_input = RecallInput(
            query=f"insights about entity {subject_id}",
            subtypes=[MemorySubtype.INFERENCE],
            subject_id=subject_id,
            limit=limit,
            mode=RecallMode.RAG,
            min_relevance=0.0,
            include_associations=False,
        )

        if observer_id is not None:
            recall_input.observer_id = observer_id

        try:
            result = await self.memory_service.recall(
                workspace_id=workspace_id,
                input=recall_input,
            )
            return result.memories
        except Exception as e:
            self.logger.error("Failed to retrieve insights for subject=%s: %s", subject_id, e)
            return []

    async def _gather_subject_memories(
            self,
            workspace_id: str,
            subject_id: str,
            observer_id: Optional[str] = None,
    ) -> list[Memory]:
        """Gather all memories about a subject for analysis."""
        # Use a broad query to get all relevant memories
        recall_input = RecallInput(
            query=f"everything about entity {subject_id}",
            subject_id=subject_id,
            limit=100,
            mode=RecallMode.RAG,
            min_relevance=0.0,
            include_associations=True,
            traverse_depth=1,
        )

        if observer_id is not None:
            recall_input.observer_id = observer_id

        try:
            result = await self.memory_service.recall(
                workspace_id=workspace_id,
                input=recall_input,
            )
            # Exclude existing inferences to avoid circular derivation
            return [
                m for m in result.memories
                if m.subtype != MemorySubtype.INFERENCE
            ]
        except Exception as e:
            self.logger.error("Failed to gather memories for subject=%s: %s", subject_id, e)
            return []

    async def _derive_with_llm(
            self,
            memories: list[Memory],
            subject_id: str,
    ) -> list[tuple[float, str]]:
        """Use LLM to derive insights from memories."""
        if not self.llm:
            self.logger.warning("No LLM service available, using fallback inference")
            return self._derive_fallback(memories, subject_id)

        # Build context from memories
        context_parts = []
        for i, memory in enumerate(memories[:50], 1):  # Cap at 50 memories
            type_label = memory.type.value.upper()
            subtype_label = f" ({memory.subtype.value})" if memory.subtype else ""
            context_parts.append(
                f"[{i}] {type_label}{subtype_label}: {memory.content}"
            )

        context = "\n".join(context_parts)

        prompt = f"""{DERIVE_SYSTEM_PROMPT}

Analyze the following {len(context_parts)} observations about entity "{subject_id}" and derive higher-order insights.

Observations:
{context}

Derive insights about this entity's patterns, preferences, tendencies, and characteristics.
Output each insight on its own line with an importance score."""

        try:
            response = await self.llm.synthesize(
                prompt=prompt,
                max_tokens=2048,
                profile="inference",
            )
            return self._parse_insights(response)
        except LLMNotConfiguredError:
            self.logger.warning("LLM not configured, using fallback inference")
            return self._derive_fallback(memories, subject_id)
        except Exception as e:
            self.logger.error("LLM inference failed: %s", e)
            return self._derive_fallback(memories, subject_id)

    def _parse_insights(self, response: str) -> list[tuple[float, str]]:
        """Parse LLM response into (importance, insight_text) tuples."""
        insights = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse [importance] insight format
            if line.startswith("["):
                try:
                    bracket_end = line.index("]")
                    importance = float(line[1:bracket_end])
                    importance = max(0.0, min(1.0, importance))
                    text = line[bracket_end + 1:].strip()
                    if text:
                        insights.append((importance, text))
                except (ValueError, IndexError):
                    # If parsing fails, treat as 0.5 importance
                    insights.append((0.5, line))
            else:
                insights.append((0.5, line))

        return insights

    def _derive_fallback(
            self,
            memories: list[Memory],
            subject_id: str,
    ) -> list[tuple[float, str]]:
        """Simple fallback inference without LLM.

        Groups memories by type and generates summary-level insights.
        """
        if len(memories) < 2:
            return []

        insights = []

        # Group by type
        by_type: dict[str, list[Memory]] = {}
        for m in memories:
            key = m.type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(m)

        # Generate type-based insights
        for type_name, type_memories in by_type.items():
            if len(type_memories) >= 2:
                importance = min(0.8, 0.3 + (len(type_memories) * 0.05))
                insights.append((
                    importance,
                    f"Has {len(type_memories)} {type_name} memories indicating "
                    f"significant {type_name} patterns"
                ))

        # Group by subtype for more specific insights
        by_subtype: dict[str, list[Memory]] = {}
        for m in memories:
            if m.subtype:
                key = m.subtype.value
                if key not in by_subtype:
                    by_subtype[key] = []
                by_subtype[key].append(m)

        for subtype_name, subtype_memories in by_subtype.items():
            if len(subtype_memories) >= 2:
                importance = min(0.9, 0.4 + (len(subtype_memories) * 0.05))
                insights.append((
                    importance,
                    f"Recurring {subtype_name} pattern observed across "
                    f"{len(subtype_memories)} memories"
                ))

        return insights[:10]  # Cap at 10 insights

    async def _store_insight(
            self,
            workspace_id: str,
            subject_id: str,
            observer_id: Optional[str],
            content: str,
            importance: float,
            source_memory_ids: list[str],
    ) -> Optional[Memory]:
        """Store a derived insight as a memory."""
        try:
            input_data = RememberInput(
                content=content,
                type=MemoryType.SEMANTIC,
                subtype=MemorySubtype.INFERENCE,
                importance=importance,
                tags=["inference", "derived"],
                metadata={
                    "source_memory_count": len(source_memory_ids),
                    "derived_at": datetime.now(timezone.utc).isoformat(),
                },
                observer_id=observer_id,
                subject_id=subject_id,
            )

            memory = await self.memory_service.remember(
                workspace_id=workspace_id,
                input=input_data,
            )

            # Create associations to source memories
            if self.association_service and memory:
                for source_id in source_memory_ids[:5]:  # Cap associations
                    try:
                        from ...models.association import AssociateInput
                        await self.association_service.create_association(
                            workspace_id=workspace_id,
                            input_data=AssociateInput(
                                source_id=memory.id,
                                target_id=source_id,
                                relationship="builds_on",
                                strength=0.7,
                                metadata={"derivation": "inference"},
                            ),
                        )
                    except Exception as e:
                        self.logger.debug("Failed to create inference association: %s", e)

            return memory
        except Exception as e:
            self.logger.error("Failed to store insight: %s", e)
            return None

    @staticmethod
    def _cache_key(workspace_id: str, subject_id: str, observer_id: Optional[str]) -> str:
        obs = observer_id or "_any"
        return f"inference:{workspace_id}:{subject_id}:{obs}"


class DefaultInferenceServicePlugin(InferenceServicePluginBase):
    """Default inference service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> DefaultInferenceService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        memory: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        association_service: AssociationService = self.get_extension(EXT_ASSOCIATION_SERVICE, v)
        cache_service: CacheService = self.get_extension(EXT_CACHE_SERVICE, v)
        return DefaultInferenceService(
            storage=storage,
            memory_service=memory,
            llm_service=llm_service,
            association_service=association_service,
            cache_service=cache_service,
            v=v,
        )
