"""
Deduplication Service - Base classes and protocols.

Prevents duplicate memories during session extraction and manual remember operations.
Uses content hashing for exact matches and embedding similarity for semantic matches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ...config import DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE, MEMORYLAYER_DEDUPLICATION_SERVICE
from .._constants import EXT_DEDUPLICATION_SERVICE, EXT_EMBEDDING_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base

# ============================================
# Deduplication Configuration
# ============================================
# Threshold for considering content a semantic duplicate (triggers UPDATE action)
MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD = "MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD"
DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD = 0.95

# Threshold for considering content similar enough to merge (triggers MERGE action)
MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD = "MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD"
DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD = 0.85


class DeduplicationAction(str, Enum):
    """Action to take for a candidate memory."""

    SKIP = "skip"  # Exact duplicate, don't create
    CREATE = "create"  # New unique memory
    UPDATE = "update"  # Update existing with new info
    MERGE = "merge"  # Merge with existing memory


@dataclass
class DeduplicationResult:
    """Result of deduplication check for a single memory."""

    action: DeduplicationAction
    existing_memory_id: str | None = None
    similarity_score: float | None = None
    reason: str = ""


class DeduplicationService(ABC):
    """Interface for deduplication service."""

    @abstractmethod
    async def check_duplicate(self, content: str, content_hash: str, embedding: list[float], workspace_id: str) -> DeduplicationResult:
        """Check if a memory is a duplicate."""
        pass

    @abstractmethod
    async def deduplicate_batch(self, candidates: list[tuple[str, str, list[float]]], workspace_id: str) -> list[DeduplicationResult]:
        """Check multiple memories for duplicates."""
        pass


# noinspection PyAbstractClass
DeduplicationServicePluginBase = make_service_plugin_base(
    ext_name=EXT_DEDUPLICATION_SERVICE,
    config_key=MEMORYLAYER_DEDUPLICATION_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_EMBEDDING_SERVICE),
    extra_defaults={
        MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD: DEFAULT_MEMORYLAYER_DEDUPLICATION_DUPLICATE_THRESHOLD,
        MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD: DEFAULT_MEMORYLAYER_DEDUPLICATION_MERGE_THRESHOLD,
    },
)
