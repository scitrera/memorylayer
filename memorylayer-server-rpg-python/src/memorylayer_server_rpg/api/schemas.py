# SPDX-License-Identifier: Apache-2.0
"""RPG (Repository Planning Graph) API request/response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class RpgNodeInput(BaseModel):
    """A single RPG node to upsert during sync."""

    node_id: str = Field(..., description="Unique node ID (path for files/dirs, path:QualifiedName for symbols)")
    node_type: str = Field(
        ...,
        description="RPG node type: rpg_directory, rpg_file, rpg_class, rpg_function, rpg_method, rpg_component, rpg_module, rpg_package, rpg_interface",
    )
    path: str = Field(..., description="File system path relative to repo root")
    name: str = Field(..., description="Display name (e.g. class name, function name, filename)")
    description: str = Field("", description="Human-readable description of this node")
    language: str | None = Field(None, description="Programming language (python, typescript, go, etc.)")
    parent_id: str | None = Field(None, description="Parent node ID in the hierarchy")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (line numbers, signatures, etc.)")


class RpgEdgeInput(BaseModel):
    """A single RPG edge to upsert during sync."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship: str = Field(
        ...,
        description="Edge type: contains, inherits, invokes, imports, composes, data_flow",
    )
    strength: float = Field(1.0, ge=0.0, le=1.0, description="Edge strength/confidence")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Edge metadata (for data_flow: data_id, data_type, transformation)",
    )


class RpgSyncRequest(BaseModel):
    """Request to sync an RPG graph snapshot into MemoryLayer."""

    nodes: list[RpgNodeInput] = Field(default_factory=list, description="Nodes to upsert")
    edges: list[RpgEdgeInput] = Field(default_factory=list, description="Edges to upsert")
    full_sync: bool = Field(
        False,
        description="If True, delete existing RPG nodes/edges not present in this payload (full replacement)",
    )
    source_commit: str | None = Field(None, description="Git commit SHA this snapshot was generated from")
    context_id: str = Field("rpg", description="Context partition for this sync (default 'rpg' for canonical graph)")
    task_id: str | None = Field(None, description="Task ID for provenance tracking")


class RpgSyncResponse(BaseModel):
    """Response from RPG sync operation."""

    nodes_created: int = Field(0, description="Number of new nodes created")
    nodes_updated: int = Field(0, description="Number of existing nodes updated")
    nodes_deleted: int = Field(0, description="Number of nodes deleted (full_sync only)")
    edges_created: int = Field(0, description="Number of new edges created")
    edges_updated: int = Field(0, description="Number of existing edges updated")
    edges_deleted: int = Field(0, description="Number of edges deleted (full_sync only)")
    sync_time_ms: int = Field(0, description="Total sync duration in milliseconds")
    source_commit: str | None = Field(None, description="Commit SHA that was synced")


class RpgNodeResponse(BaseModel):
    """A node in an RPG subgraph response."""

    node_id: str = Field(..., description="Unique node ID")
    node_type: str = Field(..., description="RPG node type")
    path: str = Field(..., description="File system path")
    name: str = Field(..., description="Display name")
    description: str = Field("", description="Node description")
    language: str | None = Field(None, description="Programming language")
    parent_id: str | None = Field(None, description="Parent node ID")
    depth: int = Field(0, description="Depth from query root (0 = root node)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    memory_id: str | None = Field(None, description="Underlying MemoryLayer memory ID")
    context_id: str | None = Field(None, description="Context partition this node belongs to")


class RpgEdgeResponse(BaseModel):
    """An edge in an RPG subgraph response."""

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Edge type")
    strength: float = Field(1.0, description="Edge strength")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Edge metadata")
    association_id: str | None = Field(None, description="Underlying MemoryLayer association ID")


class RpgSubgraphResponse(BaseModel):
    """Response containing an RPG subgraph."""

    nodes: list[RpgNodeResponse] = Field(default_factory=list, description="Nodes in the subgraph")
    edges: list[RpgEdgeResponse] = Field(default_factory=list, description="Edges in the subgraph")
    root_id: str | None = Field(None, description="Root node ID of the query")
    depth: int = Field(0, description="Maximum depth traversed")
    total_nodes: int = Field(0, description="Total node count")
    total_edges: int = Field(0, description="Total edge count")
    truncated: bool = Field(False, description="Whether the node cap truncated the result")


class RpgSearchResponse(BaseModel):
    """Response from RPG node search."""

    nodes: list[RpgNodeResponse] = Field(default_factory=list, description="Matching nodes")
    total_count: int = Field(0, description="Total matches (may exceed returned count)")
    query: str = Field("", description="Original search query")


class RpgStatusResponse(BaseModel):
    """Response for RPG encoding status."""

    workspace_id: str = Field(..., description="Workspace ID")
    has_rpg: bool = Field(False, description="Whether this workspace has RPG data")
    node_count: int = Field(0, description="Total RPG nodes in workspace")
    edge_count: int = Field(0, description="Total RPG edges in workspace")
    last_sync_commit: str | None = Field(None, description="Last synced commit SHA")
    last_sync_at: str | None = Field(None, description="Last sync timestamp")


class RpgMaintenanceResponse(BaseModel):
    """Response from an RPG maintenance operation."""

    operation: str = Field(..., description="Operation performed: validate, cleanup, or statistics")
    workspace_id: str = Field(..., description="Workspace ID that was operated on")
    result: dict[str, Any] = Field(default_factory=dict, description="Operation-specific result data")


class RpgOverlayInfo(BaseModel):
    """Information about a single RPG overlay context."""

    context_id: str = Field(..., description="Overlay context ID (e.g. rpg-task-abc123)")
    node_count: int = Field(0, description="Number of nodes in this overlay")


class RpgOverlayListResponse(BaseModel):
    """Response listing active RPG overlay contexts."""

    overlays: list[RpgOverlayInfo] = Field(default_factory=list, description="Active overlay contexts")
    total: int = Field(0, description="Total number of overlays")


class RpgOverlayDeleteResponse(BaseModel):
    """Response from deleting an RPG overlay."""

    context_id: str = Field(..., description="Overlay context ID that was deleted")
    deleted: int = Field(0, description="Number of nodes deleted")


class RpgDeleteNodesRequest(BaseModel):
    """Request to delete specific RPG nodes by node ID."""

    node_ids: list[str] = Field(..., description="RPG node IDs (not Memory UUIDs) to delete")
    context_id: str = Field("rpg", description="Context partition to delete from (default 'rpg')")


class RpgDeleteNodesResponse(BaseModel):
    """Response from deleting specific RPG nodes."""

    deleted: int = Field(0, description="Number of nodes successfully deleted")
    not_found: int = Field(0, description="Number of node IDs not found")


class RpgDeleteEdgesRequest(BaseModel):
    """Request to bulk-delete RPG edges by association ID."""

    association_ids: list[str] = Field(..., description="Association UUIDs to delete")


class RpgDeleteEdgesResponse(BaseModel):
    """Response from bulk-deleting RPG edges."""

    deleted: int = Field(0, description="Number of edges successfully deleted")
    not_found: int = Field(0, description="Number of association IDs not found")


class RpgListNodesRequest(BaseModel):
    """Query parameters for listing RPG nodes by type."""

    node_type: str | None = Field(None, description="Filter by RPG node type (e.g. rpg_file)")
    context_id: str = Field("rpg", description="Context partition to query (default 'rpg')")
    limit: int = Field(5000, ge=1, le=10000, description="Maximum number of nodes to return")


class RpgListNodesResponse(BaseModel):
    """Response from listing RPG nodes."""

    nodes: list[RpgNodeResponse] = Field(default_factory=list, description="Matching nodes")
    total_count: int = Field(0, description="Total number of nodes returned")


class RpgMergedSubgraphResponse(BaseModel):
    """Response containing a merged RPG subgraph (base + overlay)."""

    nodes: list[RpgNodeResponse] = Field(default_factory=list, description="Merged nodes")
    edges: list[RpgEdgeResponse] = Field(default_factory=list, description="Merged edges")
    root_id: str | None = Field(None, description="Root node ID")
    depth: int = Field(0, description="Maximum depth traversed")
    total_nodes: int = Field(0, description="Total merged node count")
    total_edges: int = Field(0, description="Total merged edge count")
    truncated: bool = Field(False, description="Whether either source graph was truncated")


class RpgConflictItem(BaseModel):
    """A single file-level conflict between tasks."""

    file_path: str = Field(..., description="Path of the conflicting file")
    task_ids: list[str] = Field(default_factory=list, description="Task IDs that claim this file")
    severity: str = Field("medium", description="Conflict severity: low, medium, high")


class RpgConflictResponse(BaseModel):
    """Response containing file-level conflicts for a task."""

    task_id: str = Field(..., description="The task that was checked for conflicts")
    conflicts: list[RpgConflictItem] = Field(default_factory=list, description="Detected file conflicts")
    total_conflicts: int = Field(0, description="Total number of conflicts")


class RpgSymbolConflictItem(BaseModel):
    """A single symbol-level conflict between two tasks."""

    symbol_id: str = Field(..., description="Symbol node ID (or composite for same-file conflicts)")
    file_path: str = Field(..., description="File containing the conflicting symbol(s)")
    task_ids: list[str] = Field(default_factory=list, description="Task IDs involved in this conflict")
    severity: str = Field("medium", description="Conflict severity: low, medium, high")
    description: str = Field("", description="Human-readable conflict description")


class RpgSymbolConflictResponse(BaseModel):
    """Response containing symbol-level conflicts between two tasks."""

    task_ids: list[str] = Field(default_factory=list, description="The two tasks compared")
    conflicts: list[RpgSymbolConflictItem] = Field(default_factory=list, description="Detected symbol conflicts")
    total_conflicts: int = Field(0, description="Total number of conflicts")


class RpgEnrichRequest(BaseModel):
    """Request to trigger LLM enrichment for an RPG graph."""

    context_id: str = Field("rpg", description="Context partition to enrich (default 'rpg')")
    phases: list[str] | None = Field(
        None,
        description="Enrichment phases to run. Defaults to all: ['features', 'descriptions', 'data_flows']",
    )
    async_mode: bool = Field(True, alias="async", description="If true, schedule as background task")

    model_config = {"populate_by_name": True}


class RpgEnrichResponse(BaseModel):
    """Response from an RPG enrichment trigger."""

    workspace_id: str = Field(..., description="Workspace that was enriched")
    context_id: str = Field("rpg", description="Context partition enriched")
    async_mode: bool = Field(True, description="Whether enrichment ran asynchronously")
    task_id: str | None = Field(None, description="Task ID when async_mode is true")
    stats: dict | None = Field(None, description="Enrichment stats when async_mode is false")
