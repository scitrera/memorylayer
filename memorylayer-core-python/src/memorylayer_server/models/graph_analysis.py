"""Graph analysis domain models for MemoryLayer OSS.

Models for community detection, centrality analysis, and graph statistics
over the workspace association graph.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class GraphSnapshot(BaseModel):
    """Snapshot of a workspace's association graph.

    The actual NetworkX graph object is stored as a transient attribute
    on the service, not serialized here. This model carries metadata
    about the snapshot for API responses and caching.
    """

    workspace_id: str
    context_id: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    includes_rpg: bool = False


class Community(BaseModel):
    """A detected community (cluster) within the association graph.

    Communities are groups of memories that are more densely connected
    to each other than to the rest of the graph.
    """

    id: int = Field(..., description="Community ID (0-indexed, sorted by size descending)")
    memory_ids: list[str] = Field(default_factory=list, description="Member memory IDs")
    size: int = Field(0, description="Number of members")
    cohesion_score: float = Field(
        0.0, description="Intra-community edge density (0.0-1.0)"
    )
    central_node_ids: list[str] = Field(
        default_factory=list, description="Top-3 nodes by degree within community"
    )
    label: Optional[str] = Field(
        None, description="LLM-generated topic label (set by KnowledgebaseService)"
    )


class CentralNode(BaseModel):
    """A node with high centrality in the association graph ('god node')."""

    memory_id: str
    degree: int = Field(0, description="Number of direct connections")
    betweenness: float = Field(0.0, description="Betweenness centrality score")
    community_id: int = Field(-1, description="Community this node belongs to")


class Bridge(BaseModel):
    """An edge connecting two different communities.

    Bridges represent knowledge flow between topic clusters.
    """

    source_community_id: int
    target_community_id: int
    memory_id_source: str
    memory_id_target: str
    relationship_type: str
    strength: float = Field(0.0, ge=0.0, le=1.0)


class GraphStats(BaseModel):
    """Aggregate statistics about the workspace association graph."""

    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    density: float = Field(0.0, description="Graph density (0.0-1.0)")
    avg_degree: float = 0.0
    max_degree: int = 0
    god_node_count: int = Field(0, description="Nodes with degree > 2x median")


class GraphAnalysis(BaseModel):
    """Complete graph analysis result for a workspace.

    Returned by GraphAnalysisService.analyze() as a convenience
    bundling all analysis outputs.
    """

    snapshot: GraphSnapshot
    communities: list[Community] = Field(default_factory=list)
    central_nodes: list[CentralNode] = Field(default_factory=list)
    bridges: list[Bridge] = Field(default_factory=list)
    stats: GraphStats = Field(default_factory=GraphStats)
