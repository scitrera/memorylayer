# SPDX-License-Identifier: Apache-2.0
"""Repository Planning Graph namespace for the MemoryLayer Python SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .client import MemoryLayerClient
    from .models import AuthorityContext
    from .sync_client import SyncMemoryLayerClient


class RpgNodeInput(BaseModel):
    """Repository node accepted by :meth:`RpgAPI.sync`."""

    node_id: str
    node_type: str
    path: str
    name: str
    description: str = ""
    language: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RpgEdgeInput(BaseModel):
    """Typed relationship accepted by :meth:`RpgAPI.sync`."""

    source_id: str
    target_id: str
    relationship: str
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RpgNode(BaseModel):
    """A node returned from an RPG query."""

    model_config = ConfigDict(extra="allow")

    node_id: str
    node_type: str
    path: str
    name: str
    description: str = ""
    language: str | None = None
    parent_id: str | None = None
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_id: str | None = None
    context_id: str | None = None


class RpgEdge(BaseModel):
    """An edge returned from an RPG query."""

    model_config = ConfigDict(extra="allow")

    source_id: str
    target_id: str
    relationship: str
    strength: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    association_id: str | None = None


class RpgSyncResult(BaseModel):
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_deleted: int = 0
    edges_created: int = 0
    edges_updated: int = 0
    edges_deleted: int = 0
    sync_time_ms: int = 0
    source_commit: str | None = None


class RpgSubgraph(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: list[RpgNode] = Field(default_factory=list)
    edges: list[RpgEdge] = Field(default_factory=list)
    root_id: str | None = None
    depth: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    truncated: bool = False


class RpgSearchResult(BaseModel):
    nodes: list[RpgNode] = Field(default_factory=list)
    total_count: int = 0
    query: str = ""


class RpgStatus(BaseModel):
    workspace_id: str
    has_rpg: bool = False
    node_count: int = 0
    edge_count: int = 0
    last_sync_commit: str | None = None
    last_sync_at: str | None = None


class RpgOverlay(BaseModel):
    context_id: str
    node_count: int = 0


class RpgOverlayList(BaseModel):
    overlays: list[RpgOverlay] = Field(default_factory=list)
    total: int = 0


class RpgDeleteResult(BaseModel):
    deleted: int = 0
    not_found: int = 0


class RpgOverlayDeleteResult(BaseModel):
    context_id: str
    deleted: int = 0


class RpgNodeList(BaseModel):
    nodes: list[RpgNode] = Field(default_factory=list)
    total_count: int = 0


class RpgConflict(BaseModel):
    file_path: str
    task_ids: list[str] = Field(default_factory=list)
    severity: str = "medium"


class RpgConflictResult(BaseModel):
    task_id: str
    conflicts: list[RpgConflict] = Field(default_factory=list)
    total_conflicts: int = 0


class RpgSymbolConflict(BaseModel):
    symbol_id: str
    file_path: str
    task_ids: list[str] = Field(default_factory=list)
    severity: str = "medium"
    description: str = ""


class RpgSymbolConflictResult(BaseModel):
    task_ids: list[str] = Field(default_factory=list)
    conflicts: list[RpgSymbolConflict] = Field(default_factory=list)
    total_conflicts: int = 0


class RpgMaintenanceResult(BaseModel):
    operation: str
    workspace_id: str
    result: dict[str, Any] = Field(default_factory=dict)


class RpgEnrichmentResult(BaseModel):
    workspace_id: str
    context_id: str = "rpg"
    async_mode: bool = True
    task_id: str | None = None
    stats: dict[str, Any] | None = None


def _dump_items(items: Sequence[RpgNodeInput | RpgEdgeInput | dict[str, Any]]) -> list[dict[str, Any]]:
    return [item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else dict(item) for item in items]


def _csv(values: list[str] | None) -> str | None:
    return ",".join(values) if values else None


class RpgAPI:
    """Async RPG namespace, available as ``client.rpg``."""

    def __init__(self, client: MemoryLayerClient) -> None:
        self._client = client

    async def sync(
        self,
        nodes: list[RpgNodeInput | dict[str, Any]],
        edges: list[RpgEdgeInput | dict[str, Any]],
        *,
        full_sync: bool = False,
        source_commit: str | None = None,
        context_id: str = "rpg",
        task_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> RpgSyncResult:
        payload = {
            "nodes": _dump_items(nodes),
            "edges": _dump_items(edges),
            "full_sync": full_sync,
            "source_commit": source_commit,
            "context_id": context_id,
            "task_id": task_id,
        }
        data = await self._client._request("POST", "/rpg/sync", json=payload, authority=authority)
        return RpgSyncResult(**data)

    async def get_subgraph(
        self,
        *,
        path: str | None = None,
        node_id: str | None = None,
        depth: int = 3,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        context_id: str = "rpg",
        max_nodes: int = 2000,
        authority: AuthorityContext | None = None,
    ) -> RpgSubgraph:
        if path is None and node_id is None:
            raise ValueError("path or node_id is required")
        params = {
            "path": path,
            "node_id": node_id,
            "depth": depth,
            "node_types": _csv(node_types),
            "relationship_types": _csv(relationship_types),
            "context_id": context_id,
            "max_nodes": max_nodes,
        }
        data = await self._client._request("GET", "/rpg/subgraph", params=params, authority=authority)
        return RpgSubgraph(**data)

    async def search(
        self,
        query: str,
        *,
        node_types: list[str] | None = None,
        limit: int = 20,
        context_id: str = "rpg",
        authority: AuthorityContext | None = None,
    ) -> RpgSearchResult:
        params = {"query": query, "node_types": _csv(node_types), "limit": limit, "context_id": context_id}
        data = await self._client._request("GET", "/rpg/search", params=params, authority=authority)
        return RpgSearchResult(**data)

    async def get_status(
        self,
        context_id: str | None = None,
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgStatus:
        data = await self._client._request("GET", "/rpg/status", params={"context_id": context_id}, authority=authority)
        return RpgStatus(**data)

    async def list_overlays(self, *, authority: AuthorityContext | None = None) -> RpgOverlayList:
        data = await self._client._request("GET", "/rpg/overlays", authority=authority)
        return RpgOverlayList(**data)

    async def delete_overlay(
        self,
        context_id: str,
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgOverlayDeleteResult:
        data = await self._client._request(
            "DELETE",
            f"/rpg/overlays/{quote(context_id, safe='')}",
            authority=authority,
        )
        return RpgOverlayDeleteResult(**data)

    async def delete_nodes(
        self,
        node_ids: list[str],
        *,
        context_id: str = "rpg",
        authority: AuthorityContext | None = None,
    ) -> RpgDeleteResult:
        data = await self._client._request(
            "DELETE", "/rpg/nodes", json={"node_ids": node_ids, "context_id": context_id}, authority=authority
        )
        return RpgDeleteResult(**data)

    async def delete_edges(
        self,
        association_ids: list[str],
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgDeleteResult:
        data = await self._client._request("DELETE", "/rpg/edges", json={"association_ids": association_ids}, authority=authority)
        return RpgDeleteResult(**data)

    async def list_nodes(
        self,
        *,
        node_type: str | None = None,
        context_id: str = "rpg",
        limit: int = 5000,
        authority: AuthorityContext | None = None,
    ) -> RpgNodeList:
        data = await self._client._request(
            "GET",
            "/rpg/nodes",
            params={"node_type": node_type, "context_id": context_id, "limit": limit},
            authority=authority,
        )
        return RpgNodeList(**data)

    async def get_merged_subgraph(
        self,
        overlay_context: str,
        *,
        base_context: str = "rpg",
        path: str | None = None,
        depth: int = 3,
        authority: AuthorityContext | None = None,
    ) -> RpgSubgraph:
        data = await self._client._request(
            "GET",
            "/rpg/merged-subgraph",
            params={"base_context": base_context, "overlay_context": overlay_context, "path": path, "depth": depth},
            authority=authority,
        )
        return RpgSubgraph(**data)

    async def get_conflicts(
        self,
        task_id: str,
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgConflictResult:
        data = await self._client._request("GET", "/rpg/conflicts", params={"task_id": task_id}, authority=authority)
        return RpgConflictResult(**data)

    async def get_symbol_conflicts(
        self,
        task_id_a: str,
        task_id_b: str,
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgSymbolConflictResult:
        data = await self._client._request(
            "GET", "/rpg/conflicts/symbols", params={"task_id_a": task_id_a, "task_id_b": task_id_b}, authority=authority
        )
        return RpgSymbolConflictResult(**data)

    async def maintenance(
        self,
        operation: Literal["validate", "cleanup", "statistics", "cleanup_intents", "recompute_counts"],
        *,
        authority: AuthorityContext | None = None,
    ) -> RpgMaintenanceResult:
        data = await self._client._request("POST", f"/rpg/maintenance/{operation}", authority=authority)
        return RpgMaintenanceResult(**data)

    async def enrich(
        self,
        *,
        context_id: str = "rpg",
        phases: list[str] | None = None,
        async_mode: bool = True,
        authority: AuthorityContext | None = None,
    ) -> RpgEnrichmentResult:
        data = await self._client._request(
            "POST", "/rpg/enrich", json={"context_id": context_id, "phases": phases, "async": async_mode}, authority=authority
        )
        return RpgEnrichmentResult(**data)


class SyncRpgAPI:
    """Synchronous RPG namespace, available as ``sync_client.rpg``."""

    def __init__(self, client: SyncMemoryLayerClient) -> None:
        self._client = client

    def sync(
        self,
        nodes: list[RpgNodeInput | dict[str, Any]],
        edges: list[RpgEdgeInput | dict[str, Any]],
        *,
        full_sync: bool = False,
        source_commit: str | None = None,
        context_id: str = "rpg",
        task_id: str | None = None,
    ) -> RpgSyncResult:
        data = self._client._request(
            "POST",
            "/rpg/sync",
            json={
                "nodes": _dump_items(nodes),
                "edges": _dump_items(edges),
                "full_sync": full_sync,
                "source_commit": source_commit,
                "context_id": context_id,
                "task_id": task_id,
            },
        )
        return RpgSyncResult(**data)

    def get_subgraph(
        self,
        *,
        path: str | None = None,
        node_id: str | None = None,
        depth: int = 3,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        context_id: str = "rpg",
        max_nodes: int = 2000,
    ) -> RpgSubgraph:
        if path is None and node_id is None:
            raise ValueError("path or node_id is required")
        data = self._client._request(
            "GET",
            "/rpg/subgraph",
            params={
                "path": path,
                "node_id": node_id,
                "depth": depth,
                "node_types": _csv(node_types),
                "relationship_types": _csv(relationship_types),
                "context_id": context_id,
                "max_nodes": max_nodes,
            },
        )
        return RpgSubgraph(**data)

    def search(
        self,
        query: str,
        *,
        node_types: list[str] | None = None,
        limit: int = 20,
        context_id: str = "rpg",
    ) -> RpgSearchResult:
        data = self._client._request(
            "GET", "/rpg/search", params={"query": query, "node_types": _csv(node_types), "limit": limit, "context_id": context_id}
        )
        return RpgSearchResult(**data)

    def get_status(self, context_id: str | None = None) -> RpgStatus:
        return RpgStatus(**self._client._request("GET", "/rpg/status", params={"context_id": context_id}))

    def list_overlays(self) -> RpgOverlayList:
        return RpgOverlayList(**self._client._request("GET", "/rpg/overlays"))

    def delete_overlay(self, context_id: str) -> RpgOverlayDeleteResult:
        return RpgOverlayDeleteResult(**self._client._request("DELETE", f"/rpg/overlays/{quote(context_id, safe='')}"))

    def delete_nodes(self, node_ids: list[str], *, context_id: str = "rpg") -> RpgDeleteResult:
        return RpgDeleteResult(**self._client._request("DELETE", "/rpg/nodes", json={"node_ids": node_ids, "context_id": context_id}))

    def delete_edges(self, association_ids: list[str]) -> RpgDeleteResult:
        return RpgDeleteResult(**self._client._request("DELETE", "/rpg/edges", json={"association_ids": association_ids}))

    def list_nodes(self, *, node_type: str | None = None, context_id: str = "rpg", limit: int = 5000) -> RpgNodeList:
        data = self._client._request("GET", "/rpg/nodes", params={"node_type": node_type, "context_id": context_id, "limit": limit})
        return RpgNodeList(**data)

    def get_merged_subgraph(
        self,
        overlay_context: str,
        *,
        base_context: str = "rpg",
        path: str | None = None,
        depth: int = 3,
    ) -> RpgSubgraph:
        data = self._client._request(
            "GET",
            "/rpg/merged-subgraph",
            params={"base_context": base_context, "overlay_context": overlay_context, "path": path, "depth": depth},
        )
        return RpgSubgraph(**data)

    def get_conflicts(self, task_id: str) -> RpgConflictResult:
        return RpgConflictResult(**self._client._request("GET", "/rpg/conflicts", params={"task_id": task_id}))

    def get_symbol_conflicts(self, task_id_a: str, task_id_b: str) -> RpgSymbolConflictResult:
        data = self._client._request("GET", "/rpg/conflicts/symbols", params={"task_id_a": task_id_a, "task_id_b": task_id_b})
        return RpgSymbolConflictResult(**data)

    def maintenance(
        self, operation: Literal["validate", "cleanup", "statistics", "cleanup_intents", "recompute_counts"]
    ) -> RpgMaintenanceResult:
        return RpgMaintenanceResult(**self._client._request("POST", f"/rpg/maintenance/{operation}"))

    def enrich(
        self,
        *,
        context_id: str = "rpg",
        phases: list[str] | None = None,
        async_mode: bool = True,
    ) -> RpgEnrichmentResult:
        data = self._client._request("POST", "/rpg/enrich", json={"context_id": context_id, "phases": phases, "async": async_mode})
        return RpgEnrichmentResult(**data)
