"""Default (NetworkX) Graph Analysis Service implementation."""

import logging
import statistics
from typing import Optional

import networkx as nx
from networkx.algorithms import community as nx_community
from scitrera_app_framework import Variables, get_extension, get_logger

from ...models.graph_analysis import (
    Bridge,
    CentralNode,
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)
from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import GraphAnalysisServicePluginBase
from .base import GraphAnalysisService

# RPG memory subtypes to include when include_rpg=True
_RPG_SUBTYPES = [
    "rpg_file",
    "rpg_class",
    "rpg_function",
    "rpg_method",
    "rpg_module",
    "rpg_variable",
    "rpg_import",
]


class NetworkXGraphAnalysisService(GraphAnalysisService):
    """Graph analysis service using NetworkX.

    Builds the graph fresh on each call — no service-level caching.
    Caching should be handled at the storage or API layer if needed.
    """

    def __init__(self, storage: StorageBackend, v: Variables):
        self._storage = storage
        self.logger = get_logger(v, name="GraphAnalysisService")

    async def _build_graph(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
        include_rpg: bool = False,
    ) -> nx.Graph:
        """Build a NetworkX undirected graph from workspace associations.

        Nodes carry ``memory_type`` and ``memory_subtype`` metadata.
        Edges carry ``relationship_type`` and ``strength`` metadata.
        """
        g = nx.Graph()

        # Load all memories to populate node metadata
        memories = await self._storage.search_memories_by_filter(
            workspace_id,
            status="active",
            context_id=context_id,
            limit=10000,
        )
        for mem in memories:
            g.add_node(
                mem.id,
                memory_type=getattr(mem, "memory_type", None),
                memory_subtype=getattr(mem, "subtype", None),
            )

        # Optionally merge RPG nodes
        if include_rpg:
            try:
                rpg_memories = await self._storage.search_memories_by_filter(
                    workspace_id,
                    subtypes=_RPG_SUBTYPES,
                    status="active",
                    context_id=context_id,
                    limit=10000,
                )
                for mem in rpg_memories:
                    if mem.id not in g:
                        g.add_node(
                            mem.id,
                            memory_type=getattr(mem, "memory_type", None),
                            memory_subtype=getattr(mem, "subtype", None),
                        )
                        self.logger.debug("Added RPG node %s to graph", mem.id)
            except Exception as e:
                self.logger.debug("RPG node loading skipped: %s", e)

        # Load associations for all nodes and add edges
        node_ids = list(g.nodes())
        if not node_ids:
            return g

        seen_assoc_ids: set[str] = set()
        for node_id in node_ids:
            try:
                associations = await self._storage.get_associations(
                    workspace_id=workspace_id,
                    memory_id=node_id,
                    direction="outgoing",
                )
                for assoc in associations:
                    if assoc.id in seen_assoc_ids:
                        continue
                    seen_assoc_ids.add(assoc.id)
                    # Only add edge if both endpoints are in the graph
                    if assoc.source_id in g and assoc.target_id in g:
                        g.add_edge(
                            assoc.source_id,
                            assoc.target_id,
                            relationship_type=assoc.relationship,
                            strength=assoc.strength,
                        )
            except Exception as e:
                self.logger.debug("Failed to load associations for node %s: %s", node_id, e)

        self.logger.debug(
            "Built graph for workspace %s: %d nodes, %d edges",
            workspace_id,
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return g

    async def build_workspace_graph(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
        include_rpg: bool = False,
    ) -> GraphSnapshot:
        g = await self._build_graph(workspace_id, context_id=context_id, include_rpg=include_rpg)
        return GraphSnapshot(
            workspace_id=workspace_id,
            context_id=context_id,
            node_count=g.number_of_nodes(),
            edge_count=g.number_of_edges(),
            includes_rpg=include_rpg,
        )

    async def detect_communities(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
    ) -> list[Community]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        if g.number_of_nodes() == 0:
            return []

        try:
            raw_communities = list(nx_community.louvain_communities(g, seed=42))
        except Exception as e:
            self.logger.warning("Louvain community detection failed for workspace %s: %s", workspace_id, e)
            return []

        # Sort communities by size descending so the largest is community 0
        raw_communities.sort(key=lambda c: len(c), reverse=True)

        result = []
        for idx, members in enumerate(raw_communities):
            members_list = list(members)
            subgraph = g.subgraph(members_list)

            # Cohesion = intra-community edge density
            cohesion = nx.density(subgraph) if len(members_list) > 1 else 0.0

            # Top-3 central nodes by degree within the community
            degree_in_community = dict(subgraph.degree())
            top_central = sorted(degree_in_community, key=lambda n: degree_in_community[n], reverse=True)[:3]

            result.append(
                Community(
                    id=idx,
                    memory_ids=members_list,
                    size=len(members_list),
                    cohesion_score=round(cohesion, 4),
                    central_node_ids=top_central,
                )
            )

        self.logger.debug(
            "Detected %d communities in workspace %s", len(result), workspace_id
        )
        return result

    async def compute_centrality(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
    ) -> list[CentralNode]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        if g.number_of_nodes() == 0:
            return []

        degree_centrality = nx.degree_centrality(g)
        degree_counts = dict(g.degree())

        try:
            betweenness = nx.betweenness_centrality(g)
        except Exception as e:
            self.logger.warning("Betweenness centrality failed for workspace %s: %s", workspace_id, e)
            betweenness = {n: 0.0 for n in g.nodes()}

        # Determine community membership
        community_map: dict[str, int] = {}
        try:
            raw_communities = list(nx_community.louvain_communities(g, seed=42))
            raw_communities.sort(key=lambda c: len(c), reverse=True)
            for comm_idx, members in enumerate(raw_communities):
                for node_id in members:
                    community_map[node_id] = comm_idx
        except Exception as e:
            self.logger.debug("Community detection skipped during centrality computation: %s", e)

        # Identify "god nodes": degree > 2x median degree
        all_degrees = list(degree_counts.values())
        median_degree = statistics.median(all_degrees) if all_degrees else 0

        result = []
        for node_id in g.nodes():
            result.append(
                CentralNode(
                    memory_id=node_id,
                    degree=degree_counts.get(node_id, 0),
                    betweenness=round(betweenness.get(node_id, 0.0), 6),
                    community_id=community_map.get(node_id, -1),
                )
            )

        # Sort by betweenness descending
        result.sort(key=lambda n: n.betweenness, reverse=True)
        self.logger.debug(
            "Computed centrality for %d nodes (median degree=%.1f) in workspace %s",
            len(result),
            median_degree,
            workspace_id,
        )
        return result

    async def get_bridges(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
    ) -> list[Bridge]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        if g.number_of_edges() == 0:
            return []

        # Need community membership to identify cross-community edges
        community_map: dict[str, int] = {}
        try:
            raw_communities = list(nx_community.louvain_communities(g, seed=42))
            raw_communities.sort(key=lambda c: len(c), reverse=True)
            for comm_idx, members in enumerate(raw_communities):
                for node_id in members:
                    community_map[node_id] = comm_idx
        except Exception as e:
            self.logger.warning("Community detection failed during bridge analysis for workspace %s: %s", workspace_id, e)
            return []

        bridges = []
        for src, tgt, edge_data in g.edges(data=True):
            src_comm = community_map.get(src, -1)
            tgt_comm = community_map.get(tgt, -1)
            if src_comm == -1 or tgt_comm == -1:
                continue
            if src_comm != tgt_comm:
                bridges.append(
                    Bridge(
                        source_community_id=src_comm,
                        target_community_id=tgt_comm,
                        memory_id_source=src,
                        memory_id_target=tgt,
                        relationship_type=edge_data.get("relationship_type", "related_to"),
                        strength=edge_data.get("strength", 0.0),
                    )
                )

        # Sort by strength descending
        bridges.sort(key=lambda b: b.strength, reverse=True)
        self.logger.debug(
            "Found %d bridges in workspace %s", len(bridges), workspace_id
        )
        return bridges

    async def get_statistics(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
    ) -> GraphStats:
        g = await self._build_graph(workspace_id, context_id=context_id)

        node_count = g.number_of_nodes()
        edge_count = g.number_of_edges()

        if node_count == 0:
            return GraphStats(
                node_count=0,
                edge_count=0,
                community_count=0,
                density=0.0,
                avg_degree=0.0,
                max_degree=0,
                god_node_count=0,
            )

        density = nx.density(g)
        degrees = [d for _, d in g.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        median_degree = statistics.median(degrees) if degrees else 0
        god_node_count = sum(1 for d in degrees if d > 2 * median_degree)

        community_count = 0
        try:
            raw_communities = list(nx_community.louvain_communities(g, seed=42))
            community_count = len(raw_communities)
        except Exception as e:
            self.logger.debug("Community count unavailable for stats: %s", e)

        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            community_count=community_count,
            density=round(density, 6),
            avg_degree=round(avg_degree, 2),
            max_degree=max_degree,
            god_node_count=god_node_count,
        )

    async def analyze(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
        include_rpg: bool = False,
    ) -> GraphAnalysis:
        """Run complete graph analysis and return a bundled GraphAnalysis."""
        self.logger.info(
            "Running full graph analysis for workspace %s (context=%s, include_rpg=%s)",
            workspace_id,
            context_id,
            include_rpg,
        )

        snapshot = await self.build_workspace_graph(workspace_id, context_id=context_id, include_rpg=include_rpg)
        communities = await self.detect_communities(workspace_id, context_id=context_id)
        central_nodes = await self.compute_centrality(workspace_id, context_id=context_id)
        bridges = await self.get_bridges(workspace_id, context_id=context_id)
        stats = await self.get_statistics(workspace_id, context_id=context_id)

        return GraphAnalysis(
            snapshot=snapshot,
            communities=communities,
            central_nodes=central_nodes,
            bridges=bridges,
            stats=stats,
        )


class DefaultGraphAnalysisServicePlugin(GraphAnalysisServicePluginBase):
    """Plugin for the default NetworkX-backed graph analysis service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> NetworkXGraphAnalysisService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return NetworkXGraphAnalysisService(storage=storage, v=v)
