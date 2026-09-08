"""
Default Tier Generation Service implementation.

Generates hierarchical summaries (abstract, overview) for memories using LLM.
"""

from logging import Logger

from scitrera_app_framework import ext_parse_bool, get_logger
from scitrera_app_framework.api import Variables

from ...config import (
    DEFAULT_MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED,
    DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_ENABLED,
    DEFAULT_MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS,
    DEFAULT_MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS,
    MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED,
    MEMORYLAYER_SEMANTIC_TIERING_ENABLED,
    MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS,
    MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS,
)
from ...models.generation import GenerationActivity
from ...models.llm import LLMMessage, LLMRequest, LLMRole
from ...models.memory import Memory
from ..extraction.deterministic import extractive_tiers
from ..llm import EXT_LLM_SERVICE, LLMNotConfiguredError, LLMService
from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from ..tasks.base import EXT_TASK_SERVICE, TaskService
from .base import SemanticTieringService, SemanticTieringServicePluginBase


class DefaultSemanticTieringService(SemanticTieringService):
    """Default tier generation service implementation using LLM provider."""

    # System prompts for tier generation (system + user message pattern)
    ABSTRACT_SYSTEM_PROMPT = (
        "You are a concise summarization assistant. Produce a single short "
        "sentence capturing the key factual point of the provided text. "
        "Be direct and specific — no filler, no speculation, no editorializing. "
        "Preserve important details like names, numbers, and technical specifics. "
        "Return ONLY the summary, nothing else."
    )

    OVERVIEW_SYSTEM_PROMPT = (
        "You are a concise summarization assistant. Produce a 2-3 sentence "
        "overview of the provided text. Stick strictly to the facts stated — "
        "no filler, no speculation, no editorializing. "
        "Preserve important details like names, numbers, and technical specifics. "
        "Return ONLY the overview, nothing else."
    )

    def __init__(
        self,
        llm_service: LLMService,
        storage: StorageBackend,
        v: Variables = None,
        enabled: bool = True,
        task_service: TaskService | None = None,
        abstract_skip_chars: int = DEFAULT_MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS,
        overview_skip_chars: int = DEFAULT_MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS,
        extractive_enabled: bool = DEFAULT_MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED,
    ):
        """
        Initialize tier generation service.

        Args:
            llm_service: LLM service for text generation
            storage: Storage backend for memory access
            v: Variables for logging context
            enabled: Whether tier generation is enabled
            task_service: Optional task service for background scheduling
            abstract_skip_chars: Skip the abstract LLM call when content is at or below
                this length. 0 disables skipping.
            overview_skip_chars: Skip the overview LLM call when content is at or below
                this length. 0 disables skipping.
        """
        self.llm_service = llm_service
        self.storage = storage
        self.enabled = enabled
        self.task_service = task_service
        self.abstract_skip_chars = abstract_skip_chars
        self.overview_skip_chars = overview_skip_chars
        self.extractive_enabled = extractive_enabled
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized DefaultTierGenerationService (enabled=%s, background=%s, skip_chars=%d/%d)",
            self.enabled,
            self.task_service is not None,
            self.abstract_skip_chars,
            self.overview_skip_chars,
        )

    def _generation_allowed(self) -> bool:
        checker = getattr(self.llm_service, "is_generation_allowed", None)
        return checker(GenerationActivity.SEMANTIC_TIERING) if checker is not None else True

    async def generate_abstract(self, content: str, max_tokens: int = 500) -> str:
        """
        Generate brief abstract (tier 1) from memory content.

        Skips the LLM entirely when the content is already at or below the abstract's own
        target length — see ``_should_skip``.

        Args:
            content: Full memory content
            max_tokens: Maximum tokens for abstract

        Returns:
            Brief abstract string
        """
        if self._should_skip(content, self.abstract_skip_chars, "abstract"):
            return content

        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=self.ABSTRACT_SYSTEM_PROMPT),
                LLMMessage(role=LLMRole.USER, content=f"Summarize this:\n\n{content}"),
            ],
            max_tokens=max_tokens,
            temperature_factor=0.7,
        )

        try:
            response = await self.llm_service.complete(
                request,
                profile="tier_generation",
                activity=GenerationActivity.SEMANTIC_TIERING,
            )
            return response.content.strip()
        except LLMNotConfiguredError as e:
            # Expected on a server with no LLM configured; the registry already says
            # so once at startup. Warning per stored memory would be pure noise.
            self.logger.debug("Skipping abstract generation: %s", e)
            return content[:100] + "..." if len(content) > 100 else content
        except Exception as e:
            self.logger.warning("Failed to generate abstract: %s", e)
            # Fallback: truncate content
            return content[:100] + "..." if len(content) > 100 else content

    def _should_skip(self, content: str, threshold: int, tier: str) -> bool:
        """Whether ``content`` is already short enough that summarizing it is pointless.

        The tier prompts ask for "a single short sentence" (abstract) and "a 2-3 sentence
        overview". When the source is already that short, the LLM cannot make it more
        compact — it can only paraphrase, at the cost of a call and a chance to drop a
        detail. Returning the content verbatim is both cheaper and lossless.

        A threshold of 0 disables skipping.
        """
        if threshold <= 0 or len(content) > threshold:
            return False
        self.logger.debug(
            "Skipping %s generation: content is %d chars, at or below the %d-char target",
            tier,
            len(content),
            threshold,
        )
        return True

    async def generate_overview(self, content: str, max_tokens: int = 500) -> str:
        """
        Generate overview (tier 2) from memory content.

        Skips the LLM entirely when the content is already at or below the overview's own
        target length — see ``_should_skip``.

        Args:
            content: Full memory content
            max_tokens: Maximum tokens for overview

        Returns:
            Overview string
        """
        if self._should_skip(content, self.overview_skip_chars, "overview"):
            return content

        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=self.OVERVIEW_SYSTEM_PROMPT),
                LLMMessage(role=LLMRole.USER, content=f"Provide an overview of this:\n\n{content}"),
            ],
            max_tokens=max_tokens,
            temperature_factor=0.7,
        )

        try:
            response = await self.llm_service.complete(
                request,
                profile="tier_generation",
                activity=GenerationActivity.SEMANTIC_TIERING,
            )
            return response.content.strip()
        except LLMNotConfiguredError as e:
            self.logger.debug("Skipping overview generation: %s", e)
            return content[:500] + "..." if len(content) > 500 else content
        except Exception as e:
            self.logger.warning("Failed to generate overview: %s", e)
            # Fallback: truncate content
            return content[:500] + "..." if len(content) > 500 else content

    async def generate_tiers(self, memory_id: str, workspace_id: str, force: bool = False) -> Memory:
        """
        Generate all tiers (abstract, overview) for a memory.

        Args:
            memory_id: Memory ID
            workspace_id: Workspace ID for authorization
            force: Regenerate even if tiers already exist

        Returns:
            Updated memory with generated tiers

        Raises:
            ValueError: If memory not found
        """
        # Get memory (internal read, don't track access)
        memory = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found in workspace {workspace_id}")

        # Skip if tiers already exist and force=False
        if not force and memory.abstract and memory.overview:
            self.logger.debug("Tiers already exist for memory %s, skipping", memory_id)
            return memory

        tier_metadata = dict(memory.metadata or {})
        use_extractive = self.extractive_enabled and not self._generation_allowed()
        if use_extractive:
            abstract_tier, overview_tier = extractive_tiers(
                memory.content,
                abstract_chars=max(1, self.abstract_skip_chars),
                overview_chars=max(1, self.overview_skip_chars),
            )
            abstract = memory.abstract if memory.abstract and not force else abstract_tier.content
            overview = memory.overview if memory.overview and not force else overview_tier.content
            tier_metadata["tier_method"] = "extractive"
            tier_metadata["tier_source_spans"] = {
                "abstract": [list(span) for span in abstract_tier.source_spans],
                "overview": [list(span) for span in overview_tier.source_spans],
            }
        else:
            # Generate overview first (abstract is derived from overview).
            overview = memory.overview
            if not overview or force:
                overview = await self.generate_overview(memory.content)
                self.logger.debug("Generated overview for memory %s: %s chars", memory_id, len(overview))
            abstract = memory.abstract
            if not abstract or force:
                abstract = await self.generate_abstract(overview)
                self.logger.debug("Generated abstract for memory %s: %s chars", memory_id, len(abstract))
            tier_metadata["tier_method"] = "generative"

        # Update memory in storage
        updated_memory = await self.storage.update_memory(
            workspace_id=workspace_id,
            memory_id=memory_id,
            abstract=abstract,
            overview=overview,
            metadata=tier_metadata,
        )

        self.logger.info("Generated tiers for memory %s", memory_id)
        return updated_memory

    async def generate_tiers_for_content(self, content: str) -> tuple[str, str]:
        """
        Generate tiers for content without persisting.

        Generates overview first, then derives abstract from the overview
        (shorter input produces better short summaries from LLMs).

        Args:
            content: Memory content

        Returns:
            Tuple of (abstract, overview)
        """
        overview = await self.generate_overview(content)
        abstract = await self.generate_abstract(overview)
        return abstract, overview

    async def request_tier_generation(self, memory_id: str, workspace_id: str) -> str | None:
        """
        Request tier generation, scheduling as background task if possible.

        Args:
            memory_id: Memory to generate tiers for
            workspace_id: Workspace the memory belongs to

        Returns:
            Task ID if scheduled as background task, None otherwise
        """
        if not self.enabled:
            self.logger.debug("Tier generation disabled, skipping for memory %s", memory_id)
            return None

        if self.task_service and self._generation_allowed():
            task_id = await self.task_service.schedule_task(
                task_type="generate_tiers",
                payload={"memory_id": memory_id, "workspace_id": workspace_id},
            )
            self.logger.debug("Scheduled background tier generation for memory %s (task=%s)", memory_id, task_id)
            return task_id

        # Inline fallback when no task service is available
        self.logger.debug("No task service available, generating tiers inline for memory %s", memory_id)
        await self.generate_tiers(memory_id, workspace_id)
        return None


class DefaultSemanticTieringServicePlugin(SemanticTieringServicePluginBase):
    """Default tier generation service plugin."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> DefaultSemanticTieringService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)

        enabled: bool = v.environ(
            MEMORYLAYER_SEMANTIC_TIERING_ENABLED,
            default=DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_ENABLED,
            type_fn=ext_parse_bool,
        )

        # TaskService is optional — tier generation works inline without it
        task_service: TaskService | None = None
        try:
            task_service = self.get_extension(EXT_TASK_SERVICE, v)
        except Exception:
            logger.debug("TaskService not available, tier generation will run inline")

        return DefaultSemanticTieringService(
            llm_service=llm_service,
            storage=storage,
            v=v,
            enabled=enabled,
            abstract_skip_chars=v.environ(
                MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS,
                default=DEFAULT_MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS,
                type_fn=int,
            ),
            overview_skip_chars=v.environ(
                MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS,
                default=DEFAULT_MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS,
                type_fn=int,
            ),
            extractive_enabled=v.environ(
                MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED,
                default=DEFAULT_MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED,
                type_fn=ext_parse_bool,
            ),
            task_service=task_service,
        )
