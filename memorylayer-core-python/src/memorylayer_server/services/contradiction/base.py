"""Contradiction Service - Base interface and plugin."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_CONTRADICTION_PROVIDER, DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER
from .._constants import EXT_STORAGE_BACKEND, EXT_CONTRADICTION_SERVICE
from ...utils import generate_id


@dataclass
class ContradictionRecord:
    """A detected contradiction between two memories."""
    id: str = field(default_factory=lambda: generate_id("contra"))
    workspace_id: str = ''
    memory_a_id: str = ''
    memory_b_id: str = ''
    contradiction_type: Optional[str] = None  # e.g., "negation", "value_conflict"
    confidence: float = 0.0  # 0.0-1.0
    detection_method: str = ''  # e.g., "negation_pattern", "embedding_similarity"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None  # e.g., "keep_a", "keep_b", "keep_both", "merge"
    merged_content: Optional[str] = None


class ContradictionService(ABC):
    """Interface for contradiction detection and resolution."""

    @abstractmethod
    async def check_new_memory(self, workspace_id: str, memory_id: str) -> list[ContradictionRecord]:
        """Find contradictions between a new memory and existing memories.

        Args:
            workspace_id: Workspace boundary
            memory_id: ID of the newly created memory to check

        Returns:
            List of detected contradiction records
        """
        pass

    @abstractmethod
    async def get_unresolved(self, workspace_id: str, limit: int = 10) -> list[ContradictionRecord]:
        """Get unresolved contradictions for a workspace.

        Args:
            workspace_id: Workspace boundary
            limit: Maximum number of contradictions to return

        Returns:
            List of unresolved contradiction records
        """
        pass

    @abstractmethod
    async def resolve(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: Optional[str] = None,
    ) -> Optional[ContradictionRecord]:
        """Resolve a contradiction.

        Args:
            workspace_id: Workspace boundary
            contradiction_id: ID of the contradiction to resolve
            resolution: Resolution strategy ("keep_a", "keep_b", "keep_both", "merge")
            merged_content: Merged content if resolution is "merge"

        Returns:
            Updated contradiction record, or None if not found
        """
        pass


# noinspection PyAbstractClass
class ContradictionServicePluginBase(Plugin):
    """Base plugin for contradiction service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CONTRADICTION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CONTRADICTION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_CONTRADICTION_PROVIDER, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_CONTRADICTION_PROVIDER, DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)
