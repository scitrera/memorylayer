"""Type definitions and enums for MemoryLayer.ai SDK."""

from enum import Enum


class MemoryType(str, Enum):
    """Cognitive memory types - how memory is structured."""

    EPISODIC = "episodic"  # Specific events/interactions
    SEMANTIC = "semantic"  # Facts, concepts, relationships
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Current task context


class MemorySubtype(str, Enum):
    """Domain subtypes - what the memory is about."""

    SOLUTION = "solution"  # Working fixes to problems
    PROBLEM = "problem"  # Issues encountered
    CODE_PATTERN = "code_pattern"  # Reusable patterns
    FIX = "fix"  # Bug fixes with context
    ERROR = "error"  # Error patterns and resolutions
    WORKFLOW = "workflow"  # Process knowledge
    PREFERENCE = "preference"  # User/project preferences
    DECISION = "decision"  # Architectural decisions
    DIRECTIVE = "directive"  # User instructions/constraints


class RecallMode(str, Enum):
    """Retrieval strategy for recall queries."""

    RAG = "rag"  # Fast vector similarity search
    LLM = "llm"  # Deep semantic LLM-powered retrieval
    HYBRID = "hybrid"  # Combine both strategies


class SearchTolerance(str, Enum):
    """Search precision level."""

    LOOSE = "loose"  # Fuzzy matching, broader results
    MODERATE = "moderate"  # Balanced precision/recall (default)
    STRICT = "strict"  # Exact matching, high relevance


class RelationshipCategory(str, Enum):
    """High-level relationship categories.

    These match the server's category names from the unified ontology.
    """

    HIERARCHICAL = "hierarchical"  # Parent-child, part-whole
    CAUSAL = "causal"  # Cause-effect relationships
    TEMPORAL = "temporal"  # Time-based ordering
    SIMILARITY = "similarity"  # Semantic similarity
    LEARNING = "learning"  # Knowledge evolution
    REFERENCE = "reference"  # Citations and references
    SOLUTION = "solution"  # Problem-solution relationships
    CONTEXT = "context"  # Contextual relationships
    WORKFLOW = "workflow"  # Process dependencies
    QUALITY = "quality"  # Quality assessments


class RelationshipType(str, Enum):
    """Specific relationship types between memories.

    These match the server's relationship type strings (snake_case).
    See: memorylayer_server.services.ontology.base.BASE_ONTOLOGY
    """

    # Causal
    CAUSES = "causes"
    TRIGGERS = "triggers"
    LEADS_TO = "leads_to"
    PREVENTS = "prevents"

    # Solution
    SOLVES = "solves"
    ADDRESSES = "addresses"
    ALTERNATIVE_TO = "alternative_to"
    IMPROVES = "improves"

    # Context
    OCCURS_IN = "occurs_in"
    APPLIES_TO = "applies_to"
    WORKS_WITH = "works_with"
    REQUIRES = "requires"

    # Learning
    BUILDS_ON = "builds_on"
    CONTRADICTS = "contradicts"
    CONFIRMS = "confirms"
    SUPERSEDES = "supersedes"

    # Similarity
    SIMILAR_TO = "similar_to"
    VARIANT_OF = "variant_of"
    RELATED_TO = "related_to"

    # Workflow
    FOLLOWS = "follows"
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    BLOCKS = "blocks"

    # Quality
    EFFECTIVE_FOR = "effective_for"
    PREFERRED_OVER = "preferred_over"
    DEPRECATED_BY = "deprecated_by"
