"""Knowledgebase namespace for MemoryLayer.ai Python SDK.

Wraps the ``/v1/knowledgebase/*`` endpoints exposed by ``memorylayer-core-python``:
metadata + Wikipedia-style articles auto-generated from the workspace's
association graph (index, community, entity), an Obsidian vault export,
and the underlying graph analysis (communities, central nodes, bridges).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .client import MemoryLayerClient
    from .models import AuthorityContext
    from .sync_client import SyncMemoryLayerClient


# ---------------------------------------------------------------------------
# Models — mirror server-side Pydantic models so callers get typed responses.
# ---------------------------------------------------------------------------


class KbArticle(BaseModel):
    """A single knowledgebase article (index, community, or entity)."""

    model_config = ConfigDict(extra="allow")

    id: str
    article_type: str
    title: str
    content_md: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime


class KbGraphStats(BaseModel):
    """Aggregate statistics about the workspace association graph."""

    model_config = ConfigDict(extra="allow")

    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    god_node_count: int = 0


class KbMetadata(BaseModel):
    """Metadata about a generated knowledgebase for a workspace."""

    model_config = ConfigDict(extra="allow")

    workspace_id: str
    article_count: int = 0
    community_count: int = 0
    generated_at: datetime
    stats: Optional[KbGraphStats] = None


class KbCommunity(BaseModel):
    """A detected community (cluster) within the association graph."""

    model_config = ConfigDict(extra="allow")

    id: int
    memory_ids: list[str] = Field(default_factory=list)
    size: int = 0
    cohesion_score: float = 0.0
    central_node_ids: list[str] = Field(default_factory=list)
    label: Optional[str] = None


class KbCentralNode(BaseModel):
    """A node with high centrality in the association graph ('god node')."""

    model_config = ConfigDict(extra="allow")

    memory_id: str
    degree: int = 0
    betweenness: float = 0.0
    community_id: int = -1


class KbBridge(BaseModel):
    """An edge connecting two different communities."""

    model_config = ConfigDict(extra="allow")

    source_community_id: int
    target_community_id: int
    memory_id_source: str
    memory_id_target: str
    relationship_type: str
    strength: float = 0.0


class KbGraphSnapshot(BaseModel):
    """Snapshot metadata for the association graph."""

    model_config = ConfigDict(extra="allow")

    workspace_id: str
    context_id: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    includes_rpg: bool = False


class KbGraphAnalysis(BaseModel):
    """Complete graph analysis result for a workspace."""

    model_config = ConfigDict(extra="allow")

    snapshot: KbGraphSnapshot
    communities: list[KbCommunity] = Field(default_factory=list)
    central_nodes: list[KbCentralNode] = Field(default_factory=list)
    bridges: list[KbBridge] = Field(default_factory=list)
    stats: KbGraphStats = Field(default_factory=KbGraphStats)


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


class KnowledgebaseAPI:
    """Knowledgebase namespace — access via ``client.kb.<method>``."""

    def __init__(self, client: "MemoryLayerClient") -> None:
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    async def get(
        self,
        workspace_id: str | None = None,
        *,
        context_id: str | None = None,
        authority: "AuthorityContext | None" = None,
    ) -> KbMetadata:
        """Get the latest knowledgebase metadata for a workspace."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if context_id:
            params["context_id"] = context_id
        data = await self._client._request(
            "GET",
            "/knowledgebase",
            params=params,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        return KbMetadata(**data)

    async def list_articles(
        self,
        workspace_id: str | None = None,
        *,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        authority: "AuthorityContext | None" = None,
    ) -> list[KbArticle]:
        """List knowledgebase articles, optionally filtered by type."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if article_type:
            params["article_type"] = article_type
        data = await self._client._request(
            "GET",
            "/knowledgebase/articles",
            params=params,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        return [KbArticle(**a) for a in data.get("articles", [])]

    async def get_article(
        self,
        article_id: str,
        workspace_id: str | None = None,
        *,
        authority: "AuthorityContext | None" = None,
    ) -> KbArticle:
        """Fetch a single article by ID (e.g. ``index``, ``community-3``, ``entity-foo``)."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        data = await self._client._request(
            "GET",
            f"/knowledgebase/articles/{article_id}",
            params=params,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        return KbArticle(**data)

    async def generate(
        self,
        workspace_id: str | None = None,
        *,
        regenerate: bool = False,
        max_communities: int | None = None,
        max_god_nodes: int | None = None,
        include_rpg: bool = False,
        context_id: str | None = None,
        authority: "AuthorityContext | None" = None,
    ) -> KbMetadata:
        """Trigger (re)generation of the workspace knowledgebase."""
        payload: dict[str, Any] = {"regenerate": regenerate, "include_rpg": include_rpg}
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if context_id:
            payload["context_id"] = context_id
        if max_communities is not None:
            payload["max_communities"] = max_communities
        if max_god_nodes is not None:
            payload["max_god_nodes"] = max_god_nodes
        data = await self._client._request(
            "POST",
            "/knowledgebase/generate",
            json=payload,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        return KbMetadata(**data)

    async def export_vault(
        self,
        workspace_id: str | None = None,
        *,
        authority: "AuthorityContext | None" = None,
    ) -> bytes:
        """Download the knowledgebase as an Obsidian-compatible vault zip."""
        transport = self._client._ensure_transport()
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        obo_headers = self._client._obo_headers(authority)
        response = await transport.request(
            "GET", "/knowledgebase/export", params=params, headers=obo_headers,
        )
        response.raise_for_status()
        return response.content

    async def get_graph_analysis(
        self,
        workspace_id: str | None = None,
        *,
        context_id: str | None = None,
        authority: "AuthorityContext | None" = None,
    ) -> KbGraphAnalysis | None:
        """Run a fresh graph analysis on the workspace association graph."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if context_id:
            params["context_id"] = context_id
        data = await self._client._request(
            "GET",
            "/knowledgebase/graph",
            params=params,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        analysis = data.get("analysis")
        if analysis is None:
            return None
        return KbGraphAnalysis(**analysis)

    async def get_community(
        self,
        community_id: int,
        workspace_id: str | None = None,
        *,
        authority: "AuthorityContext | None" = None,
    ) -> KbCommunity:
        """Fetch a single community by ID with cached label + live members."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        data = await self._client._request(
            "GET",
            f"/knowledgebase/graph/communities/{community_id}",
            params=params,
            authority=authority,
            enterprise_feature="Knowledgebase",
        )
        return KbCommunity(**data)


# ---------------------------------------------------------------------------
# Sync API (mirrors KnowledgebaseAPI shape)
# ---------------------------------------------------------------------------


class SyncKnowledgebaseAPI:
    """Synchronous Knowledgebase namespace — access via ``sync_client.kb.<method>``."""

    def __init__(self, client: "SyncMemoryLayerClient") -> None:
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    def get(
        self,
        workspace_id: str | None = None,
        *,
        context_id: str | None = None,
    ) -> KbMetadata:
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if context_id:
            params["context_id"] = context_id
        data = self._client._request(
            "GET", "/knowledgebase", params=params, enterprise_feature="Knowledgebase",
        )
        return KbMetadata(**data)

    def list_articles(
        self,
        workspace_id: str | None = None,
        *,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KbArticle]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if article_type:
            params["article_type"] = article_type
        data = self._client._request(
            "GET", "/knowledgebase/articles", params=params,
            enterprise_feature="Knowledgebase",
        )
        return [KbArticle(**a) for a in data.get("articles", [])]

    def get_article(
        self,
        article_id: str,
        workspace_id: str | None = None,
    ) -> KbArticle:
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        data = self._client._request(
            "GET", f"/knowledgebase/articles/{article_id}", params=params,
            enterprise_feature="Knowledgebase",
        )
        return KbArticle(**data)

    def generate(
        self,
        workspace_id: str | None = None,
        *,
        regenerate: bool = False,
        max_communities: int | None = None,
        max_god_nodes: int | None = None,
        include_rpg: bool = False,
        context_id: str | None = None,
    ) -> KbMetadata:
        payload: dict[str, Any] = {"regenerate": regenerate, "include_rpg": include_rpg}
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if context_id:
            payload["context_id"] = context_id
        if max_communities is not None:
            payload["max_communities"] = max_communities
        if max_god_nodes is not None:
            payload["max_god_nodes"] = max_god_nodes
        data = self._client._request(
            "POST", "/knowledgebase/generate", json=payload,
            enterprise_feature="Knowledgebase",
        )
        return KbMetadata(**data)

    def get_graph_analysis(
        self,
        workspace_id: str | None = None,
        *,
        context_id: str | None = None,
    ) -> KbGraphAnalysis | None:
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if context_id:
            params["context_id"] = context_id
        data = self._client._request(
            "GET", "/knowledgebase/graph", params=params,
            enterprise_feature="Knowledgebase",
        )
        analysis = data.get("analysis")
        if analysis is None:
            return None
        return KbGraphAnalysis(**analysis)

    def get_community(
        self,
        community_id: int,
        workspace_id: str | None = None,
    ) -> KbCommunity:
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        data = self._client._request(
            "GET", f"/knowledgebase/graph/communities/{community_id}", params=params,
            enterprise_feature="Knowledgebase",
        )
        return KbCommunity(**data)
