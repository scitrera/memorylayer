"""Default (NetworkX) Graph Analysis Service implementation."""

import asyncio
import logging
import statistics

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
from ._densify import DensifyConfig, densify_graph
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


def _node_label(mem, *, max_len: int = 80) -> str | None:
    """Short, human-readable label for a memory graph node.

    Prefers the abstract (a one-line summary) and falls back to the first line
    of content, truncated for compact display in the graph. Returns None when
    there is nothing usable, so callers can fall back to the memory id.
    """
    text = (getattr(mem, "abstract", None) or getattr(mem, "content", None) or "").strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    if len(first_line) > max_len:
        return first_line[: max_len - 1].rstrip() + "…"
    return first_line


class NetworkXGraphAnalysisService(GraphAnalysisService):
    """Graph analysis service using NetworkX.

    Builds the graph fresh on each call — no service-level caching.
    Caching should be handled at the storage or API layer if needed.
    """

    def __init__(self, storage: StorageBackend, v: Variables):
        self._storage = storage
        self.logger = get_logger(v, name="GraphAnalysisService")
        # Experimental, config-gated graph densification (default OFF).
        self._densify_cfg = DensifyConfig.from_variables(v)

    async def _build_graph(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
        densify_override: "DensifyConfig | None" = None,
    ) -> nx.Graph:
        """Build a NetworkX undirected graph from workspace associations.

        Nodes carry ``memory_type`` and ``memory_subtype`` metadata.
        Edges carry ``relationship_type`` and ``strength`` metadata.

        Uses a single bulk call to ``get_associations_batch`` instead of
        N per-node ``get_associations`` calls to avoid N+1 query behaviour.
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
                label=_node_label(mem),
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
                            label=_node_label(mem),
                        )
                        self.logger.debug("Added RPG node %s to graph", mem.id)
            except Exception as e:
                self.logger.debug("RPG node loading skipped: %s", e)

        # Load associations for all nodes in ONE bulk call and add edges
        node_ids = list(g.nodes())
        if not node_ids:
            return g

        try:
            associations = await self._storage.get_associations_batch(
                workspace_id=workspace_id,
                memory_ids=node_ids,
                direction="outgoing",
            )
            for assoc in associations:
                # Only add edge if both endpoints are in the graph
                if assoc.source_id in g and assoc.target_id in g:
                    g.add_edge(
                        assoc.source_id,
                        assoc.target_id,
                        relationship_type=assoc.relationship,
                        strength=assoc.strength,
                    )
        except Exception as e:
            self.logger.debug("Failed to load associations in bulk for workspace %s: %s", workspace_id, e)

        # Experimental: augment the sparse association graph with weighted edges
        # from cosine / MaxSim / entity-cooccurrence signals so Louvain produces
        # meaningful communities. Gated (default OFF); best-effort — a failure
        # leaves the base association graph intact.
        cfg = densify_override if densify_override is not None else self._densify_cfg
        if cfg.enabled:
            try:
                await densify_graph(g, workspace_id, self._storage, cfg, self.logger)
            except Exception as e:
                self.logger.warning("Graph densification failed for workspace %s: %s", workspace_id, e)

        self.logger.debug(
            "Built graph for workspace %s: %d nodes, %d edges",
            workspace_id,
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return g

    async def densify_preview(
        self,
        workspace_id: str,
        override_cfg: "DensifyConfig",
        context_id: str | None = None,
    ) -> dict:
        """Experimental tuning tool. Build the base association graph, run
        community detection (baseline), apply densification with ``override_cfg``
        to the SAME graph, and run detection again — returning a before/after
        comparison plus the per-signal edge summary. Read-only (no persistence);
        lets operators sweep densify parameters via the API without restarting.
        """
        # Base association graph only (force densify OFF for the baseline build).
        g = await self._build_graph(
            workspace_id, context_id, densify_override=DensifyConfig(enabled=False),
        )
        baseline = self._louvain_summary(g)

        if override_cfg.enabled:
            try:
                await densify_graph(g, workspace_id, self._storage, override_cfg, self.logger)
            except Exception as e:
                self.logger.warning("densify_preview: densification failed for %s: %s", workspace_id, e)
        densified = self._louvain_summary(g)

        return {
            "workspace_id": workspace_id,
            "baseline": baseline,
            "densified": densified,
            "densify": g.graph.get("densify"),
            "config": {
                "cosine": {"enabled": override_cfg.cosine_enabled, "threshold": override_cfg.cosine_threshold,
                           "k": override_cfg.cosine_k, "weight": override_cfg.cosine_weight},
                "maxsim": {"enabled": override_cfg.maxsim_enabled, "threshold": override_cfg.maxsim_threshold,
                           "k": override_cfg.maxsim_k, "weight": override_cfg.maxsim_weight},
                "entity": {"enabled": override_cfg.entity_enabled, "weight": override_cfg.entity_weight,
                           "max_group": override_cfg.entity_max_group},
            },
        }

    @staticmethod
    def _louvain_summary(g) -> dict:
        """Community-structure summary of a graph: counts, size histogram, top
        sizes and weighted modularity. Used by densify_preview for tuning."""
        comms = list(nx_community.louvain_communities(g, seed=42, weight="weight"))
        sizes = sorted((len(c) for c in comms), reverse=True)
        hist = {"size_1": 0, "size_2_5": 0, "size_6_20": 0, "size_21_plus": 0}
        for s in sizes:
            if s == 1:
                hist["size_1"] += 1
            elif s <= 5:
                hist["size_2_5"] += 1
            elif s <= 20:
                hist["size_6_20"] += 1
            else:
                hist["size_21_plus"] += 1
        try:
            modularity = round(nx_community.modularity(g, comms, weight="weight"), 4)
        except Exception:
            modularity = None
        return {
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
            "community_count": len(comms),
            "size_histogram": hist,
            "top_sizes": sizes[:15],
            "modularity": modularity,
        }

    async def _analyze_core(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
        include_central_nodes: bool = True,
    ) -> GraphAnalysis:
        """Build the graph once, run Louvain once, derive all analysis outputs.

        This is the single-pass core used by ``analyze()``.  The six public
        ABC methods each call ``_build_graph`` independently (fine for direct
        invocation); only ``analyze()`` goes through this fast path.

        ``include_central_nodes=False`` skips betweenness centrality and the
        per-memory ``central_nodes`` payload — the heavy part of the analysis —
        for the lighter communities-only view (communities + bridges + stats).
        """
        g = await self._build_graph(workspace_id, context_id=context_id, include_rpg=include_rpg)

        # Louvain + betweenness-centrality over the in-memory graph is pure-CPU (no
        # I/O) and O(V*E) on betweenness — heavy enough to starve the asyncio event
        # loop and trip the liveness probe (/livez can't be answered, kubelet SIGTERMs
        # the pod). Run it in a worker thread so the loop keeps servicing probes and
        # other requests while it computes.
        return await asyncio.to_thread(
            self._compute_analysis_from_graph,
            g,
            workspace_id,
            context_id,
            include_rpg,
            include_central_nodes,
        )

    def _compute_analysis_from_graph(
        self,
        g,
        workspace_id: str,
        context_id: str | None,
        include_rpg: bool,
        include_central_nodes: bool,
    ) -> GraphAnalysis:
        """Synchronous, CPU-bound analysis over an already-built graph.

        Runs OFF the event loop via ``asyncio.to_thread`` (see ``_analyze_core``);
        it must not perform any async or storage I/O — it only reads the in-memory
        networkx graph, which the loop no longer touches once built.
        """
        node_count = g.number_of_nodes()
        edge_count = g.number_of_edges()

        # --- snapshot ---
        snapshot = GraphSnapshot(
            workspace_id=workspace_id,
            context_id=context_id,
            node_count=node_count,
            edge_count=edge_count,
            includes_rpg=include_rpg,
        )

        # --- early-exit for empty graph ---
        if node_count == 0:
            return GraphAnalysis(
                snapshot=snapshot,
                communities=[],
                central_nodes=[],
                bridges=[],
                stats=GraphStats(
                    node_count=0,
                    edge_count=0,
                    community_count=0,
                    density=0.0,
                    avg_degree=0.0,
                    max_degree=0,
                    god_node_count=0,
                ),
            )

        # --- single Louvain run (seed=42, same as all per-method calls) ---
        try:
            raw_communities = list(nx_community.louvain_communities(g, seed=42))
        except Exception as e:
            self.logger.warning("Louvain community detection failed for workspace %s: %s", workspace_id, e)
            raw_communities = []

        # Sort communities by size descending — same ordering as detect_communities()
        raw_communities.sort(key=lambda c: len(c), reverse=True)

        # Build community membership map: node_id -> community_index
        community_map: dict[str, int] = {}
        for comm_idx, members in enumerate(raw_communities):
            for node_id in members:
                community_map[node_id] = comm_idx

        # --- communities (mirrors detect_communities logic exactly) ---
        communities: list[Community] = []
        for idx, members in enumerate(raw_communities):
            members_list = list(members)
            subgraph = g.subgraph(members_list)
            cohesion = nx.density(subgraph) if len(members_list) > 1 else 0.0
            degree_in_community = dict(subgraph.degree())
            top_central = sorted(degree_in_community, key=lambda n: degree_in_community[n], reverse=True)[:3]
            communities.append(
                Community(
                    id=idx,
                    memory_ids=members_list,
                    size=len(members_list),
                    cohesion_score=round(cohesion, 4),
                    central_node_ids=top_central,
                )
            )

        # --- centrality (mirrors compute_centrality logic exactly) ---
        degree_centrality = nx.degree_centrality(g)  # noqa: F841 — kept for parity
        degree_counts = dict(g.degree())
        all_degrees = list(degree_counts.values())
        median_degree = statistics.median(all_degrees) if all_degrees else 0

        # Betweenness + the per-memory central_nodes payload are the heavy part of
        # the analysis; skip them entirely for the communities-only view.
        central_nodes: list[CentralNode] = []
        if include_central_nodes:
            try:
                betweenness = nx.betweenness_centrality(g)
            except Exception as e:
                self.logger.warning("Betweenness centrality failed for workspace %s: %s", workspace_id, e)
                betweenness = {n: 0.0 for n in g.nodes()}

            for node_id in g.nodes():
                central_nodes.append(
                    CentralNode(
                        memory_id=node_id,
                        label=g.nodes[node_id].get("label"),
                        degree=degree_counts.get(node_id, 0),
                        betweenness=round(betweenness.get(node_id, 0.0), 6),
                        community_id=community_map.get(node_id, -1),
                    )
                )
            central_nodes.sort(key=lambda n: n.betweenness, reverse=True)

        # --- bridges (mirrors get_bridges logic exactly) ---
        bridges: list[Bridge] = []
        if edge_count > 0 and community_map:
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
            bridges.sort(key=lambda b: b.strength, reverse=True)

        # --- statistics (mirrors get_statistics logic exactly) ---
        density = nx.density(g)
        degrees = [d for _, d in g.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        god_node_count = sum(1 for d in degrees if d > 2 * median_degree)

        stats = GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            community_count=len(raw_communities),
            density=round(density, 6),
            avg_degree=round(avg_degree, 2),
            max_degree=max_degree,
            god_node_count=god_node_count,
        )

        self.logger.debug(
            "Single-pass analysis complete for workspace %s: %d nodes, %d edges, %d communities",
            workspace_id,
            node_count,
            edge_count,
            len(raw_communities),
        )

        return GraphAnalysis(
            snapshot=snapshot,
            communities=communities,
            central_nodes=central_nodes,
            bridges=bridges,
            stats=stats,
        )

    async def build_workspace_graph(
        self,
        workspace_id: str,
        context_id: str | None = None,
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
        context_id: str | None = None,
    ) -> list[Community]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        return await asyncio.to_thread(self._communities_from_graph, g, workspace_id)

    def _communities_from_graph(self, g, workspace_id: str) -> list[Community]:
        """CPU-bound community detection; runs off the event loop (asyncio.to_thread)."""
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

        self.logger.debug("Detected %d communities in workspace %s", len(result), workspace_id)
        return result

    async def compute_centrality(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> list[CentralNode]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        return await asyncio.to_thread(self._centrality_from_graph, g, workspace_id)

    def _centrality_from_graph(self, g, workspace_id: str) -> list[CentralNode]:
        """CPU-bound centrality (betweenness + Louvain); runs off the event loop (asyncio.to_thread)."""
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
                    label=g.nodes[node_id].get("label"),
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
        context_id: str | None = None,
    ) -> list[Bridge]:
        g = await self._build_graph(workspace_id, context_id=context_id)
        return await asyncio.to_thread(self._bridges_from_graph, g, workspace_id)

    def _bridges_from_graph(self, g, workspace_id: str) -> list[Bridge]:
        """CPU-bound bridge detection (Louvain + edge scan); runs off the event loop (asyncio.to_thread)."""
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
        self.logger.debug("Found %d bridges in workspace %s", len(bridges), workspace_id)
        return bridges

    async def get_statistics(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> GraphStats:
        g = await self._build_graph(workspace_id, context_id=context_id)
        return await asyncio.to_thread(self._statistics_from_graph, g, workspace_id)

    def _statistics_from_graph(self, g, workspace_id: str) -> GraphStats:
        """CPU-bound statistics (Louvain community count + degree stats); off the event loop (asyncio.to_thread)."""
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
        context_id: str | None = None,
        include_rpg: bool = False,
        include_central_nodes: bool = True,
    ) -> GraphAnalysis:
        """Run complete graph analysis and return a bundled GraphAnalysis.

        Uses ``_analyze_core`` so the graph is built once and Louvain runs once,
        instead of the previous approach of calling each public method separately
        (which rebuilt the graph 5× and reran Louvain ~4×).

        ``include_central_nodes=False`` returns the lighter communities-only
        analysis (no betweenness, no per-memory central_nodes).
        """
        self.logger.info(
            "Running graph analysis for workspace %s (context=%s, include_rpg=%s, central_nodes=%s)",
            workspace_id,
            context_id,
            include_rpg,
            include_central_nodes,
        )
        return await self._analyze_core(
            workspace_id,
            context_id=context_id,
            include_rpg=include_rpg,
            include_central_nodes=include_central_nodes,
        )


class DefaultGraphAnalysisServicePlugin(GraphAnalysisServicePluginBase):
    """Plugin for the default NetworkX-backed graph analysis service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> NetworkXGraphAnalysisService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return NetworkXGraphAnalysisService(storage=storage, v=v)
