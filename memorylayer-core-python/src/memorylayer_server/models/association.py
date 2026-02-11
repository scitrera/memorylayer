"""
Association and relationship models for MemoryLayer.ai semantic graph.

Defines 65 relationship types organized by 11 categories for rich knowledge graphs.
Relationship types are plain strings validated against the unified ontology in
``memorylayer_server.services.ontology.base.BASE_ONTOLOGY``.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RelationshipCategory(str, Enum):
    """High-level categories for relationship types."""

    HIERARCHICAL = "hierarchical"  # Parent-child, part-whole
    CAUSAL = "causal"  # Cause and effect relationships
    TEMPORAL = "temporal"  # Time-based ordering
    SIMILARITY = "similarity"  # Similarity and relatedness
    LEARNING = "learning"  # Knowledge evolution
    REFINEMENT = "refinement"  # Refinement and replacement
    REFERENCE = "reference"  # Citations and references
    SOLUTION = "solution"  # Problem-solving relationships
    CONTEXT = "context"  # Contextual and applicability
    WORKFLOW = "workflow"  # Process and dependencies
    QUALITY = "quality"  # Quality and preference


# All known relationship type strings, kept as a convenience constant.
# The authoritative source of truth is BASE_ONTOLOGY in
# memorylayer_server.services.ontology.base
KNOWN_RELATIONSHIP_TYPES: frozenset[str] = frozenset({
    # Hierarchical
    "parent_of", "child_of", "part_of", "has_part", "instance_of", "type_of",
    # Causal
    "causes", "caused_by", "enables", "enabled_by",
    "triggers", "triggered_by", "leads_to", "led_to_by",
    "prevents", "prevented_by",
    # Temporal
    "before", "after", "during",
    # Similarity
    "similar_to", "duplicate_of", "related_to", "variant_of",
    # Learning
    "contradicts", "supports", "supported_by",
    "builds_on", "built_upon_by", "confirms",
    "supersedes", "superseded_by",
    # Refinement
    "refines", "refined_by", "replaces", "replaced_by",
    # Reference
    "references", "referenced_by",
    # Solution
    "solves", "solved_by", "addresses", "addressed_by",
    "alternative_to", "improves", "improved_by",
    # Context
    "occurs_in", "contains_occurrence", "applies_to", "has_applicable",
    "works_with", "requires", "required_by",
    # Workflow
    "follows", "followed_by", "depends_on", "depended_on_by",
    "blocks", "blocked_by",
    # Quality
    "effective_for", "has_effective", "preferred_over", "less_preferred_than",
    "deprecated_by", "deprecates",
})


def get_relationship_category(relationship: str) -> Optional[str]:
    """Get the category for a relationship type from the ontology.

    Returns None if the relationship is not in the known set.
    """
    from ..services.ontology.base import BASE_ONTOLOGY
    info = BASE_ONTOLOGY.get(relationship)
    return info.get("category") if info else None


class Association(BaseModel):
    """Typed edge in the semantic memory graph."""

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Unique association identifier")
    workspace_id: str = Field(..., description="Workspace boundary")

    # Graph structure
    source_id: str = Field(..., description="Source memory ID")
    target_id: str = Field(..., description="Target memory ID")
    relationship: str = Field(
        ...,
        description="Relationship type (e.g., similar_to, causes, solves)",
    )

    # Edge metadata
    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Relationship strength (0.0-1.0)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

    @property
    def category(self) -> Optional[str]:
        """Get the category of this relationship."""
        return get_relationship_category(self.relationship)


class AssociateInput(BaseModel):
    """Request model for creating an association between memories."""

    source_id: str = Field(..., description="Source memory ID")
    target_id: str = Field(..., description="Target memory ID")
    relationship: str = Field(
        ...,
        description="Relationship type (e.g., similar_to, causes, solves)",
    )
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Relationship strength")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class GraphQueryInput(BaseModel):
    """Request model for multi-hop graph traversal."""

    start_memory_id: str = Field(..., description="Starting memory for traversal")

    # Filters
    relationship_types: list[str] = Field(
        default_factory=list,
        description="Filter by specific relationship types (empty = all)"
    )
    relationship_categories: list[RelationshipCategory] = Field(
        default_factory=list,
        description="Filter by relationship categories (empty = all)"
    )

    # Traversal settings
    max_depth: int = Field(3, ge=1, le=5, description="Maximum traversal depth")
    direction: str = Field(
        "both",
        pattern="^(outgoing|incoming|both)$",
        description="Traversal direction: outgoing, incoming, both"
    )
    min_strength: float = Field(0.0, ge=0.0, le=1.0, description="Minimum edge strength")

    # Result limits
    max_paths: int = Field(100, ge=1, le=1000, description="Maximum paths to return")
    max_nodes: int = Field(50, ge=1, le=500, description="Maximum nodes in result")


class GraphPath(BaseModel):
    """A path through the memory graph."""

    nodes: list[str] = Field(..., description="Memory IDs in path order")
    edges: list[Association] = Field(..., description="Associations connecting nodes")
    total_strength: float = Field(..., description="Product of all edge strengths")
    depth: int = Field(..., description="Path length")


class GraphQueryResult(BaseModel):
    """Response model for graph queries."""

    paths: list[GraphPath] = Field(..., description="All paths found")
    total_paths: int = Field(..., description="Total paths (may exceed returned count)")
    unique_nodes: list[str] = Field(..., description="All unique memory IDs visited")
    query_latency_ms: int = Field(0, description="Query latency in milliseconds")
