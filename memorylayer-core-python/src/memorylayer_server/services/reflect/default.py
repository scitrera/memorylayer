"""
Reflect Service - Memory synthesis and summarization.

Uses LLM to:
- Synthesize multiple memories into coherent summary
- Generate category summaries
- Answer complex queries requiring reasoning
"""
from datetime import datetime, timezone
from logging import Logger
from typing import Optional, Any

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import ReflectServicePluginBase
from ..storage import StorageBackend, EXT_STORAGE_BACKEND
from ..memory import MemoryService, EXT_MEMORY_SERVICE
from ..llm import LLMService, EXT_LLM_SERVICE, LLMNotConfiguredError
from ...models import ReflectInput, ReflectResult, RecallInput, RecallMode, DetailLevel

# Token budget mapping for detail levels
REFLECT_TOKEN_BUDGETS = {
    DetailLevel.ABSTRACT: 150,
    DetailLevel.OVERVIEW: 500,
    DetailLevel.FULL: 4096,
}


class ReflectService:
    """Service for LLM-powered memory synthesis."""

    def __init__(
            self,
            storage: StorageBackend,
            memory_service: MemoryService,
            llm_service: Optional[LLMService] = None,
            v: Variables = None
    ):
        self.storage = storage
        self.memory_service = memory_service
        self.llm = llm_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        if llm_service:
            self.logger.info("Initialized ReflectService with LLM service (model: %s)", llm_service.default_model)
        else:
            self.logger.info("Initialized ReflectService without LLM service")

    def _build_recall_input(
        self,
        query: str,
        *,
        types: Optional[list] = None,
        subtypes: Optional[list] = None,
        tags: Optional[list] = None,
        context_id: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: RecallMode = RecallMode.LLM,
        limit: int = 20,
        min_relevance: float = 0.5,
        include_associations: bool = True,
        traverse_depth: int = 1,
    ) -> RecallInput:
        """Build a RecallInput for reflection operations."""
        return RecallInput(
            query=query,
            types=types,
            subtypes=subtypes,
            tags=tags,
            context_id=context_id,
            user_id=user_id,
            mode=mode,
            limit=limit,
            min_relevance=min_relevance,
            include_associations=include_associations,
            traverse_depth=traverse_depth,
        )

    async def reflect(
            self,
            workspace_id: str,
            input: ReflectInput,
    ) -> ReflectResult:
        """
        Synthesize memories matching query into coherent reflection.

        Steps:
        1. Recall relevant memories
        2. Gather associated memories (depth=input.depth)
        3. Send to LLM with synthesis prompt
        4. Return reflection with source references
        """
        self.logger.info(
            "Generating reflection in workspace: %s, query: %s",
            workspace_id,
            input.query[:50]
        )

        start_time = datetime.now(timezone.utc)

        # 1. Recall relevant memories
        recall_input = self._build_recall_input(
            query=input.query,
            types=input.types,
            subtypes=input.subtypes,
            tags=input.tags,
            context_id=input.context_id,
            user_id=input.user_id,
            mode=RecallMode.LLM,  # Use LLM mode for best semantic matching
            limit=20,  # Get more memories for synthesis
            min_relevance=0.5,
            include_associations=True,
            traverse_depth=input.depth,
        )

        recall_result = await self.memory_service.recall(
            workspace_id=workspace_id,
            input=recall_input
        )

        if not recall_result.memories:
            self.logger.warning("No memories found for reflection query: %s", input.query)
            return ReflectResult(
                reflection="No relevant memories found to reflect upon.",
                source_memories=[],
                confidence=0.0,
                tokens_processed=0
            )

        self.logger.debug("Found %s memories for reflection", len(recall_result.memories))

        # 2. Gather associated memories (already handled by traverse_depth in recall)
        source_memory_ids = [m.id for m in recall_result.memories]

        # 3. Synthesize with LLM
        max_tokens = REFLECT_TOKEN_BUDGETS.get(input.detail_level, 4096)
        if self.llm:
            reflection, tokens_used = await self._synthesize_with_llm(
                memories=recall_result.memories,
                query=input.query,
                max_tokens=max_tokens
            )
            confidence = self._calculate_confidence(recall_result.memories)
        else:
            # Fallback: Simple concatenation if no LLM available
            self.logger.warning("No LLM client available, using simple synthesis")
            reflection, tokens_used, confidence = self._simple_synthesis(
                memories=recall_result.memories,
                query=input.query,
                max_tokens=max_tokens
            )

        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        self.logger.info(
            "Generated reflection in %s ms, %s tokens, confidence: %.2f",
            latency_ms,
            tokens_used,
            confidence
        )

        result = ReflectResult(
            reflection=reflection,
            source_memories=source_memory_ids if input.include_sources else [],
            confidence=confidence,
            tokens_processed=tokens_used
        )

        return result

    async def _synthesize_with_llm(
            self,
            memories: list,
            query: str,
            max_tokens: int
    ) -> tuple[str, int]:
        """
        Use LLM to synthesize memories into coherent reflection.

        In production, this would call OpenAI/Anthropic API.
        """
        self.logger.debug("Synthesizing %s memories with LLM", len(memories))

        # Build context from memories
        context_parts = []
        for i, memory in enumerate(memories, 1):
            context_parts.append(
                f"[{i}] {memory.type.value.upper()} - {memory.content}"
            )

        context = "\n\n".join(context_parts)

        # Build synthesis prompt
        prompt = f"""Based on the following memories, provide a synthesized reflection on: "{query}"

Memories:
{context}

Synthesize these memories into a coherent, insightful reflection that directly addresses the query. Focus on patterns, relationships, and key insights. Be concise but comprehensive.

Reflection:"""

        # In production, call actual LLM API
        # For now, return a placeholder
        if self.llm:
            try:
                # Placeholder - replace with actual LLM call
                reflection = await self._call_llm(prompt, max_tokens)
                tokens_used = len(prompt.split()) + len(reflection.split())  # Rough estimate

                return reflection, tokens_used
            except Exception as e:
                self.logger.error("LLM synthesis failed: %s", e)
                # Fall back to simple synthesis
                return self._simple_synthesis(memories, query, max_tokens)
        else:
            return self._simple_synthesis(memories, query, max_tokens)

    async def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM API using the configured LLM service."""
        if not self.llm:
            self.logger.warning("No LLM service available, using fallback")
            return "Memory synthesis requires LLM integration. Please configure an LLM provider."

        try:
            result = await self.llm.synthesize(
                prompt=prompt,
                max_tokens=max_tokens,
                # temperature=0.7,
                profile="reflection",
            )
            return result
        except LLMNotConfiguredError:
            self.logger.warning("LLM provider not configured, using fallback")
            return "Memory synthesis requires LLM integration. Please configure an LLM profile (set MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER and MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL)."
        except Exception as e:
            self.logger.error("LLM call failed: %s", e)
            return f"LLM synthesis failed: {e}"

    def _simple_synthesis(
            self,
            memories: list,
            query: str,
            max_tokens: int
    ) -> tuple[str, int, float]:
        """
        Simple synthesis without LLM.

        Returns: (reflection, tokens_used, confidence)
        """
        self.logger.debug("Using simple synthesis for %s memories", len(memories))

        # Group by type
        by_type = {}
        for memory in memories:
            type_name = memory.type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(memory)

        # Build reflection
        parts = [f"Reflection on: {query}\n"]

        for type_name, type_memories in by_type.items():
            parts.append(f"\n{type_name.upper()} memories ({len(type_memories)}):")
            for memory in type_memories[:5]:  # Limit to 5 per type
                parts.append(f"- {memory.content[:200]}...")  # Truncate

        reflection = "\n".join(parts)

        # Truncate to max_tokens (rough estimate: 1 token ~= 4 chars)
        max_chars = max_tokens * 4
        if len(reflection) > max_chars:
            reflection = reflection[:max_chars] + "..."

        tokens_used = len(reflection.split())
        confidence = min(1.0, len(memories) / 10.0)  # Simple confidence based on memory count

        return reflection, tokens_used, confidence

    def _calculate_confidence(self, memories: list) -> float:
        """
        Calculate confidence score based on memory quality.

        Factors:
        - Number of memories found
        - Average importance
        - Recency
        """
        if not memories:
            return 0.0

        # Number of memories (more is better, up to a point)
        count_factor = min(1.0, len(memories) / 10.0)

        # Average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)

        # Recency factor (memories accessed recently are more relevant)
        recency_factor = 0.0
        if memories[0].last_accessed_at:
            # Simple heuristic - if recently accessed, boost confidence
            recency_factor = 0.2

        confidence = (count_factor * 0.5) + (avg_importance * 0.4) + recency_factor

        return min(1.0, confidence)

    async def answer_question(
            self,
            workspace_id: str,
            question: str,
            context_memories: Optional[list[str]] = None,
    ) -> ReflectResult:
        """
        Answer a question using memories as knowledge base.

        If context_memories provided, use those; otherwise, recall relevant memories.
        """
        self.logger.info("Answering question: %s", question[:50])

        # Use provided context or recall memories
        if context_memories:
            memories = []
            for memory_id in context_memories:
                memory = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
                if memory:
                    memories.append(memory)
        else:
            # Recall relevant memories
            recall_input = self._build_recall_input(
                query=question,
                mode=RecallMode.LLM,
                limit=10,
                min_relevance=0.6,
                include_associations=False,
            )
            recall_result = await self.memory_service.recall(workspace_id, recall_input)
            memories = recall_result.memories

        if not memories:
            return ReflectResult(
                reflection="I don't have enough information to answer this question.",
                source_memories=[],
                confidence=0.0,
                tokens_processed=0
            )

        # Generate answer using reflection
        reflect_input = ReflectInput(
            query=question,
            detail_level=DetailLevel.OVERVIEW,
            include_sources=True,
            depth=1
        )

        result = await self.reflect(workspace_id, reflect_input)
        return result


class DefaultReflectServicePlugin(ReflectServicePluginBase):
    """Default reflect service plugin."""
    PROVIDER_NAME = 'default'

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_MEMORY_SERVICE, EXT_LLM_SERVICE)

    def initialize(self, v: Variables, logger: Logger) -> ReflectService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        memory: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        return ReflectService(
            storage=storage,
            memory_service=memory,
            llm_service=llm_service,
            v=v
        )
