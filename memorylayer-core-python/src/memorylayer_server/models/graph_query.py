"""Graph query domain models for MemoryLayer (P2 Track A).

These are the backend-agnostic DTOs returned by ``GraphQueryService`` — the
recall/RAG-facing read seam over the workspace association graph. The SHAPES
here are the contract: the OSS relational ``default`` backend and the
enterprise Apache-AGE ``age`` backend MUST return identical structures so a
caller (e.g. recall) can swap backends transparently.

Unlike ``graph_analysis`` (whole-graph community/centrality analytics), these
models describe localized read queries: neighborhoods, bounded k-hop
subgraphs, shortest paths, typed-edge pattern matches, and relationship
rollups. ``GraphNode`` / ``GraphEdge`` are deliberately minimal (id + type +
edge metadata) so both backends can populate them cheaply.
"""

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A memory node in a graph-query result.

    Carries only identity + type metadata — enough for a recall caller to
    hydrate or rank. Backend-agnostic: populated identically by the relational
    and AGE backends.
    """

    memory_id: str
    memory_type: str | None = None
    memory_subtype: str | None = None


class GraphEdge(BaseModel):
    """A directed association edge in a graph-query result.

    ``source_id``/``target_id`` preserve the stored direction of the
    association; callers that want undirected semantics collapse on the
    unordered endpoint pair. ``relationship`` + ``strength`` mirror the
    association columns.
    """

    source_id: str
    target_id: str
    relationship: str
    strength: float = Field(0.0, description="Edge strength (0.0-1.0)")


class NeighborResult(BaseModel):
    """Neighborhood (depth-bounded ego graph) around a single root memory."""

    root_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = Field(False, description="True if the node limit clipped the result")


class SubgraphResult(BaseModel):
    """A bounded k-hop subgraph induced by one or more root memories."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    root_ids: list[str] = Field(default_factory=list)
    truncated: bool = Field(False, description="True if the node limit clipped the result")


class PathResult(BaseModel):
    """Result of a shortest-path query between two memories."""

    found: bool = Field(False, description="True if a path within max_hops exists")
    nodes: list[GraphNode] = Field(default_factory=list, description="Path nodes in order, src..dst")
    edges: list[GraphEdge] = Field(default_factory=list, description="Path edges in order")
    hops: int = Field(0, description="Number of edges in the path (0 if not found)")


class PatternMatch(BaseModel):
    """A single edge matching a typed-relationship pattern query."""

    source_id: str
    target_id: str
    relationship: str
    strength: float = Field(0.0, description="Edge strength (0.0-1.0)")


class RelationshipRollup(BaseModel):
    """Count of edges per relationship type within a workspace."""

    relationship: str
    count: int = Field(0, description="Number of edges with this relationship type")


# ---------------------------------------------------------------------------
# Entity-neighborhood DTOs (the consumer of the Entity layer)
# ---------------------------------------------------------------------------
#
# These describe the cross-source / relationship-traversal primitive over the
# canonical-entity layer: "what memories does this entity mention, and which
# entities are co-mentioned with it?" This is P4 channel-G groundwork + a
# perspective/relationship lookup — NOT a chat-recall expansion. The SHAPES are
# the contract: the enterprise AGE backend (rich, reads the C1 Entity/MENTIONS
# vertices+edges) and the OSS relational backend (reads the registry tables +
# associations) MUST return identical structures (parity relaxed only under
# truncation, per the OSS/enterprise parity bar).


class EntityMentionedMemory(BaseModel):
    """A memory that an entity MENTIONS (Entity -> Memory edge), with its role."""

    memory_id: str
    role: str = Field("mention", description="Mention role, e.g. 'mention' or 'self'")
    memory_type: str | None = None
    memory_subtype: str | None = None


class CoMentionedEntity(BaseModel):
    """An entity co-mentioned with the root (shares >=1 mentioned memory)."""

    entity_id: str
    label: str | None = Field(None, description="Canonical name of the co-mentioned entity")
    entity_type: str | None = None
    shared_memory_count: int = Field(
        0, description="Number of distinct memories shared with the root entity"
    )


class EntityNeighborhood(BaseModel):
    """The neighborhood of a single canonical entity.

    ``memories_for_entity`` are the memories the entity MENTIONS (1 hop);
    ``entities_co_mentioned`` are the entities reachable via
    ``Entity -> MENTIONS -> Memory <- MENTIONS <- Entity`` (a shared memory),
    ranked by ``shared_memory_count`` desc then entity_id asc. ``truncated`` is
    set when either list is clipped by the limit.
    """

    entity_id: str
    memories_for_entity: list[EntityMentionedMemory] = Field(default_factory=list)
    entities_co_mentioned: list[CoMentionedEntity] = Field(default_factory=list)
    truncated: bool = Field(False, description="True if a limit clipped the result")


# ---------------------------------------------------------------------------
# Fragment-traversal DTOs (the consumer of the Fragment layer)
# ---------------------------------------------------------------------------
#
# These describe the fragment-traversal primitive over the decomposed-fact layer:
# "what facts (fragments) were derived from this source memory?" The fact channel
# stores decomposed facts as subtype="fact" memories carrying
# metadata["source_id"]=<parent>; the enterprise AGE tier materializes each as a
# Fragment vertex + DERIVED_FROM edge to the source Memory (P4.5). The SHAPES are
# the contract: the enterprise AGE backend (rich, reads the Fragment/DERIVED_FROM
# vertices+edges) and the OSS relational backend (reads the fact memories + their
# source_id) MUST return identical structures (parity relaxed only under
# truncation, per the OSS/enterprise parity bar).
#
# DARK + MEASURABLE: this is NOT wired into recall. Whether fragment-traversal
# beats the existing fact channel (E) on LoCoMo is a later measurement.


class DerivedFragment(BaseModel):
    """A fact fragment DERIVED_FROM a source memory (Fragment -> Memory edge)."""

    fragment_id: str
    content: str | None = Field(None, description="The fact fragment's text content")
    source_id: str = Field(..., description="The source memory the fragment was derived from")


class FragmentResult(BaseModel):
    """The fragments derived from a single source memory.

    ``fragments`` are the ``subtype="fact"`` memories whose
    ``metadata["source_id"]`` is the queried memory, ordered by fragment_id asc.
    ``truncated`` is set when the list is clipped by the limit.
    """

    source_id: str
    fragments: list[DerivedFragment] = Field(default_factory=list)
    truncated: bool = Field(False, description="True if a limit clipped the result")
