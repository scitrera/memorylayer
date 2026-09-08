"""
Extraction Service - Base classes and interfaces.

Extracts memories from session content using LLM-based classification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

from ...config import DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE, MEMORYLAYER_EXTRACTION_SERVICE
from ...models.memory import Memory, MemorySubtype, MemoryType
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
    categories: list[ExtractionCategory] | None = None  # None = all categories
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
        self, session_id: str, workspace_id: str, context_id: str, session_content: str, working_memory: dict, options: ExtractionOptions
    ) -> ExtractionResult:
        """Extract memories from a session."""
        pass

    @abstractmethod
    async def decompose_to_facts(self, content: str, reference_time: "datetime | None" = None) -> list[dict]:
        """Decompose composite content into atomic facts.

        Args:
            content: Composite memory content to decompose.
            reference_time: Optional reference timestamp for the source content.
                When provided, relative temporal references in ``content``
                (e.g. "last Tuesday", "yesterday", "two weeks ago") are resolved
                to absolute calendar dates against this anchor (write-time
                temporal normalization).

        Returns a list of dicts with keys: 'content', 'type' (optional),
        'subtype' (optional). Each dict MAY ALSO include an optional
        'event_time' key: an ISO-8601 string (or None) giving the absolute
        date the fact is ABOUT, resolved from relative references using
        ``reference_time``. Providers may return a parsed ``datetime`` (or None)
        for this key. Callers must treat a missing 'event_time' key as absent
        (the no-LLM / failure fallback returns dicts without it).
        """
        pass

    @abstractmethod
    async def classify_content(self, content: str) -> tuple["MemoryType", "MemorySubtype | None"]:
        """Classify a single memory's content into a type and subtype.

        Uses LLM to determine the extraction category, then maps through
        CATEGORY_MAPPING to get (MemoryType, MemorySubtype).

        Returns (MemoryType.SEMANTIC, None) as fallback.
        """
        pass

    # ------------------------------------------------------------------
    # Knowledge-graph extraction (entities + typed facts)
    #
    # These are the canonical home for the entity/fact extraction that feeds
    # the entity-anchored recall channel (entity metadata) and the typed
    # association graph (facts -> ontology relationships). They are concrete
    # (not abstract) so existing providers inherit them unchanged:
    #   * extract_entities — cheap, deterministic, no-LLM (regex). Safe to call
    #     synchronously at ingest.
    #   * extract_facts — LLM-backed (subject, relation, object) triples for the
    #     multi-hop knowledge graph. The base returns [] so it is a no-op until a
    #     provider overrides it; it is intended to run inside the async
    #     ``auto_enrich`` enrichment task (LLM latency rules out inline ingest),
    #     with each triple's relation validated against the ontology and persisted
    #     as a typed association edge.
    # ------------------------------------------------------------------

    def extract_entities(self, content: str) -> dict:
        """Extract entities from a single memory's content (no LLM).

        Returns ``{"speaker": str | None, "entities": list[str], "entity_types":
        dict[str, str]}``:

        * ``entities`` — flat list of mention strings (back-compat; the
          entity-anchor recall channel consumes this unchanged).
        * ``entity_types`` — maps each entity name to an ``EntityType`` value
          (``models/entity_registry.py``: person/org/project/place/concept/event)
          when the provider can type it. Registry accretion reads this to assign
          a proper ``EntityType`` per entity; names absent from the map fall back
          to the speaker-is-PERSON / else-CONCEPT heuristic.

        The default implementation uses the shared regex extractor (dialogue
        speaker + proper-noun spans) which cannot infer types, so it returns
        ``entity_types={}``; providers may override with typed NER/LLM extraction
        and populate it.
        """
        from ..memory.entities import extract_entities as _extract

        return _extract(content)

    async def extract_facts(self, content: str) -> list[dict]:
        """Extract typed (subject, relation, object) facts via LLM.

        Returns a list of ``{"subject", "relation", "object", "confidence"?}``
        dicts. The base implementation returns ``[]`` (no-op); LLM-backed
        providers override it. Callers map ``relation`` onto an ontology
        relationship type and persist the fact as a typed association edge so the
        graph supports multi-hop traversal.
        """
        return []

    async def generate_cue_anchors(self, content: str) -> list[dict]:
        """Generate structured ``[entity] + [aspect]`` cue anchors for a memory.

        A cue anchor is a compact semantic key (e.g. "Jane hiking trip",
        "Project Orion timeline") that names the main entity plus one salient
        facet of the memory. Cue anchors are embedded and indexed separately so
        recall can search them and dereference back to their primary memories,
        then RRF-fuse that as an additional retrieval arm (adopted from Microsoft
        Memora's abstraction+cue indexing).

        Returns 1-3 cue dicts, each carrying its parsed components::

            {"cue": str, "entity": str, "aspect": str}

        where ``cue`` is the full ``"[entity] [aspect]"`` string. Exposing the
        parsed ``entity`` lets the ingest path link the cue to a canonical
        Entity in the registry (cue arm + entity-anchored arm share one entity
        vocabulary). The base implementation returns ``[]`` (no-op) so existing
        providers inherit safely; LLM-backed providers override it. The cue
        channel ships DARK (default OFF), so this is never called unless the
        channel is explicitly enabled.
        """
        return []


# noinspection PyAbstractClass
ExtractionServicePluginBase = make_service_plugin_base(
    ext_name=EXT_EXTRACTION_SERVICE,
    config_key=MEMORYLAYER_EXTRACTION_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_LLM_SERVICE, EXT_DEDUPLICATION_SERVICE, EXT_EMBEDDING_SERVICE),
)
