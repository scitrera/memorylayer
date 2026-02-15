"""
Deduplication Service - Base classes and protocols.

Prevents duplicate memories during session extraction and manual remember operations.
Uses content hashing for exact matches and embedding similarity for semantic matches.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_DEDUPLICATION_SERVICE, DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE

from .._constants import EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE, EXT_DEDUPLICATION_SERVICE

# ============================================
# Deduplication Configuration
# ============================================
# Threshold for considering content a semantic duplicate (triggers UPDATE action)
MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD = 'MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD'
DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD = 0.95

# Threshold for considering content similar enough to merge (triggers MERGE action)
MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD = 'MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD'
DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD = 0.85


class DeduplicationAction(str, Enum):
    """Action to take for a candidate memory."""

    SKIP = "skip"       # Exact duplicate, don't create
    CREATE = "create"   # New unique memory
    UPDATE = "update"   # Update existing with new info
    MERGE = "merge"     # Merge with existing memory


@dataclass
class DeduplicationResult:
    """Result of deduplication check for a single memory."""

    action: DeduplicationAction
    existing_memory_id: Optional[str] = None
    similarity_score: Optional[float] = None
    reason: str = ""


class DeduplicationService(ABC):
    """Interface for deduplication service."""

    @abstractmethod
    async def check_duplicate(
        self,
        content: str,
        content_hash: str,
        embedding: list[float],
        workspace_id: str
    ) -> DeduplicationResult:
        """Check if a memory is a duplicate."""
        pass

    @abstractmethod
    async def deduplicate_batch(
        self,
        candidates: list[tuple[str, str, list[float]]],
        workspace_id: str
    ) -> list[DeduplicationResult]:
        """Check multiple memories for duplicates."""
        pass


# noinspection PyAbstractClass
class DeduplicationServicePluginBase(Plugin):
    """Base plugin for deduplication service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_DEDUPLICATION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_DEDUPLICATION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_DEDUPLICATION_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_DEDUPLICATION_SERVICE, DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE)
        v.set_default_value(MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD, DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD)
        v.set_default_value(MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD, DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE)
