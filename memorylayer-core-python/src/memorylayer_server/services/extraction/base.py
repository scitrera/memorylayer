"""
Extraction Service - Base classes and interfaces.

Extracts memories from session content using LLM-based classification.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

from ...config import MEMORYLAYER_EXTRACTION_SERVICE, DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE
from ...models.memory import Memory, MemoryType, MemorySubtype
from .._constants import (
    EXT_DEDUPLICATION_SERVICE,
    EXT_EMBEDDING_SERVICE,
    EXT_EXTRACTION_SERVICE,
    EXT_LLM_SERVICE,
    EXT_STORAGE_BACKEND,
)
from .._plugin_factory import make_service_plugin_base


class ExtractionCategory(str, Enum):
    """Categories for extracted memories."""

    PROFILE = "profile"  # User identity, background
    PREFERENCES = "preferences"  # Choices, settings
    ENTITIES = "entities"  # Projects, people, concepts
    EVENTS = "events"  # Decisions, milestones
    CASES = "cases"  # Problems with solutions
    PATTERNS = "patterns"  # Reusable processes


# Mapping from extraction categories to memory types/subtypes
CATEGORY_MAPPING = {
    ExtractionCategory.PROFILE: (MemoryType.SEMANTIC, MemorySubtype.PROFILE),
    ExtractionCategory.PREFERENCES: (MemoryType.SEMANTIC, MemorySubtype.PREFERENCE),
    ExtractionCategory.ENTITIES: (MemoryType.SEMANTIC, MemorySubtype.ENTITY),
    ExtractionCategory.EVENTS: (MemoryType.EPISODIC, MemorySubtype.EVENT),
    ExtractionCategory.CASES: (MemoryType.EPISODIC, MemorySubtype.SOLUTION),
    ExtractionCategory.PATTERNS: (MemoryType.PROCEDURAL, MemorySubtype.WORKFLOW),
}


@dataclass
class ExtractionOptions:
    """Options for memory extraction."""

    min_importance: float = 0.5
    deduplicate: bool = True
    categories: Optional[list[ExtractionCategory]] = None  # None = all categories
    max_memories: int = 50


@dataclass
class ExtractedMemory:
    """A memory extracted from session content."""

    content: str
    category: ExtractionCategory
    importance: float
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of session extraction."""

    session_id: str
    memories_extracted: int
    memories_deduplicated: int
    memories_created: list[Memory]
    breakdown: dict[str, int]  # category -> count
    extraction_time_ms: int


class ExtractionService(ABC):
    """Interface for extraction service."""

    @abstractmethod
    async def extract_from_session(
            self,
            session_id: str,
            workspace_id: str,
            context_id: str,
            session_content: str,
            working_memory: dict,
            options: ExtractionOptions
    ) -> ExtractionResult:
        """Extract memories from a session."""
        pass

    @abstractmethod
    async def decompose_to_facts(self, content: str) -> list[dict]:
        """Decompose composite content into atomic facts.

        Returns list of dicts with keys: 'content', 'type' (optional), 'subtype' (optional).
        """
        pass

    @abstractmethod
    async def classify_content(self, content: str) -> tuple['MemoryType', 'Optional[MemorySubtype]']:
        """Classify a single memory's content into a type and subtype.

        Uses LLM to determine the extraction category, then maps through
        CATEGORY_MAPPING to get (MemoryType, MemorySubtype).

        Returns (MemoryType.SEMANTIC, None) as fallback.
        """
        pass


# noinspection PyAbstractClass
ExtractionServicePluginBase = make_service_plugin_base(
    ext_name=EXT_EXTRACTION_SERVICE,
    config_key=MEMORYLAYER_EXTRACTION_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_LLM_SERVICE, EXT_DEDUPLICATION_SERVICE, EXT_EMBEDDING_SERVICE),
)
