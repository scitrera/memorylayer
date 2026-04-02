"""Contradiction Service - Base interface and plugin."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ...config import MEMORYLAYER_CONTRADICTION_PROVIDER, DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER
from .._constants import EXT_STORAGE_BACKEND, EXT_CONTRADICTION_SERVICE
from .._plugin_factory import make_service_plugin_base
from ...utils import generate_id

# Valid contradiction types
CONTRADICTION_TYPE_NEGATION = 'negation'
CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT = 'semantic_value_conflict'
CONTRADICTION_TYPE_TEMPORAL_SUPERSESSION = 'temporal_supersession'
CONTRADICTION_TYPE_SCOPE_CONFLICT = 'scope_conflict'


@dataclass
class ContradictionRecord:
    """A detected contradiction between two memories."""
    id: str = field(default_factory=lambda: generate_id("contra"))
    workspace_id: str = ''
    memory_a_id: str = ''
    memory_b_id: str = ''
    contradiction_type: Optional[str] = None  # e.g., "negation", "semantic_value_conflict", "temporal_supersession", "scope_conflict"
    confidence: float = 0.0  # 0.0-1.0
    detection_method: str = ''  # e.g., "negation_pattern", "embedding_similarity", "entity_value_extraction"
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None  # e.g., "keep_a", "keep_b", "keep_both", "merge"
    merged_content: Optional[str] = None
    newer_memory_id: Optional[str] = None  # Temporal ordering: which memory is more recent


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

    @abstractmethod
    async def scan_workspace(
        self,
        workspace_id: str,
        batch_size: int = 50,
    ) -> list[ContradictionRecord]:
        """Scan all memories in a workspace for contradictions.

        Compares memory pairs with high embedding similarity to find conflicts
        that may not have been caught during individual memory creation.

        Args:
            workspace_id: Workspace boundary
            batch_size: Number of memories to process per batch

        Returns:
            List of newly detected contradiction records
        """
        pass

    @abstractmethod
    async def check_semantic_conflict(
        self,
        memory_a,
        memory_b,
    ) -> Optional[ContradictionRecord]:
        """Check if two memories have a semantic value conflict.

        Uses entity-value extraction (regex patterns) and embedding similarity
        to identify conflicts where the same subject has contradicting values.

        Args:
            memory_a: First memory object to compare
            memory_b: Second memory object to compare

        Returns:
            ContradictionRecord if a conflict is detected, None otherwise
        """
        pass


# noinspection PyAbstractClass
ContradictionServicePluginBase = make_service_plugin_base(
    ext_name=EXT_CONTRADICTION_SERVICE,
    config_key=MEMORYLAYER_CONTRADICTION_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)
