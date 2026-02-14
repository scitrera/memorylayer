"""
Tier Generation Service - Base interface and plugin.

Provides the ABC interface and plugin base for tier generation services.
"""
from abc import ABC, abstractmethod
from typing import Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_SEMANTIC_TIERING_SERVICE, DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE
from ...models.memory import Memory

from ..storage import EXT_STORAGE_BACKEND
from ..llm import EXT_LLM_SERVICE

# Extension point constant
EXT_SEMANTIC_TIERING_SERVICE = 'memorylayer-tier-generation-service'


class SemanticTieringService(ABC):
    """Interface for tier generation service."""

    @abstractmethod
    async def generate_abstract(
            self,
            content: str,
            max_tokens: int = 30
    ) -> str:
        """Generate brief abstract (tier 1) from memory content."""
        pass

    @abstractmethod
    async def generate_overview(
            self,
            content: str,
            max_tokens: int = 100
    ) -> str:
        """Generate overview (tier 2) from memory content."""
        pass

    @abstractmethod
    async def generate_tiers(
            self,
            memory_id: str,
            workspace_id: str,
            force: bool = False
    ) -> Memory:
        """Generate all tiers (abstract, overview) for a memory."""
        pass

    @abstractmethod
    async def generate_tiers_for_content(
            self,
            content: str
    ) -> tuple[str, str]:
        """Generate tiers for content without persisting."""
        pass

    async def request_tier_generation(self, memory_id: str, workspace_id: str) -> Optional[str]:
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
class SemanticTieringServicePluginBase(Plugin):
    """Base plugin for tier generation service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_SEMANTIC_TIERING_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_SEMANTIC_TIERING_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_SEMANTIC_TIERING_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_SEMANTIC_TIERING_SERVICE, DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_LLM_SERVICE,)
