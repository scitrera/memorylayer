"""
Tier Generation Service - Base interface and plugin.

Provides the ABC interface and plugin base for tier generation services.
"""

from abc import ABC, abstractmethod

from ...config import DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE, MEMORYLAYER_SEMANTIC_TIERING_SERVICE
from ...models.memory import Memory
from .._constants import EXT_LLM_SERVICE, EXT_SEMANTIC_TIERING_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base


class SemanticTieringService(ABC):
    """Interface for tier generation service."""

    @abstractmethod
    async def generate_abstract(self, content: str, max_tokens: int = 30) -> str:
        """Generate brief abstract (tier 1) from memory content."""
        pass

    @abstractmethod
    async def generate_overview(self, content: str, max_tokens: int = 100) -> str:
        """Generate overview (tier 2) from memory content."""
        pass

    @abstractmethod
    async def generate_tiers(self, memory_id: str, workspace_id: str, force: bool = False) -> Memory:
        """Generate all tiers (abstract, overview) for a memory."""
        pass

    @abstractmethod
    async def generate_tiers_for_content(self, content: str) -> tuple[str, str]:
        """Generate tiers for content without persisting."""
        pass

    async def request_tier_generation(self, memory_id: str, workspace_id: str) -> str | None:
        """
        Request tier generation for a memory, potentially as a background task.

        Default implementation is a no-op. Subclasses wire in TaskService
        and/or inline fallback.

        Args:
            memory_id: Memory to generate tiers for
            workspace_id: Workspace the memory belongs to

        Returns:
            Task ID if scheduled as background task, None otherwise
        """
        return None


# noinspection PyAbstractClass
SemanticTieringServicePluginBase = make_service_plugin_base(
    ext_name=EXT_SEMANTIC_TIERING_SERVICE,
    config_key=MEMORYLAYER_SEMANTIC_TIERING_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_LLM_SERVICE),
)
