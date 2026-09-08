"""Default (relational) Graph Query Service implementation (P2 Track A).

``RelationalGraphQueryService`` answers the recall/RAG-facing graph reads
directly over the relational storage primitives — no graph database required.
It is the OSS default and the PARITY GOLDEN that the enterprise Apache-AGE
backend is conformance-tested against: both backends MUST return identical
normalized DTOs.

Design choices (parity-critical)
---------------------------------
* Traversal (``neighbors``, ``k_hop_subgraph``, ``shortest_path``) is a
  deterministic breadth-first expansion over the workspace edge set fetched in
  bulk via ``storage.get_associations_batch`` (the existing non-N+1 primitive).
  BFS is done in Python for exact cross-backend parity.

* **Truncation is applied AFTER the full bounded reachable set is computed**
  (MAJOR-1/2 fix). The node limit cap is NOT applied mid-expansion; this
  guarantees both backends retain the same deterministic node set regardless of
  which order edges happen to appear in the batch. The canonical cap ordering
  is ``[roots sorted] + [non-roots sorted][:remaining]`` — identical to the
  AGE backend.

* **Expansion frontier is sorted** at every hop for determinism; the batch
  itself is sorted by ``(source_id, target_id, relationship)`` before processing.

* ``relationship_rollup`` uses the dedicated
  ``storage.count_associations_by_relationship`` GROUP BY aggregate.

* Node attrs are hydrated from ``search_memories_by_filter`` (one bulk read of
  active workspace memories) with a ``get_memory`` fallback.

* ``validate_graph_hops`` (from ``base``) is called on every depth / max_hops
  parameter — both backends raise the same ``ValueError`` for out-of-range
  values. This is the MAJOR-3 fix.

Tie-break (``shortest_path``): when multiple equal-length paths exist, the path
whose node-id sequence is lexicographically smallest is selected. Enforced by
expanding BFS frontiers in sorted order; the AGE backend mirrors this with an
``ORDER BY`` on path node-id strings.
"""

import logging

from scitrera_app_framework import Variables, get_extension, get_logger

from ...models.association import Association
from ...models.graph_query import (
    CoMentionedEntity,
    DerivedFragment,
    EntityMentionedMemory,
    EntityNeighborhood,
    FragmentResult,
    GraphEdge,
    GraphNode,
    NeighborResult,
    PathResult,
    PatternMatch,
    RelationshipRollup,
    SubgraphResult,
)
from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import GraphQueryServicePluginBase
from .base import GraphQueryService, MAX_GRAPH_TRAVERSAL_HOPS, validate_graph_hops

# Same hard cap the graph-analysis backends use when bulk-loading memories.
_MEMORY_LIMIT = 10000


class RelationalGraphQueryService(GraphQueryService):
    """Graph query service over relational storage primitives (OSS default)."""

    PROVIDER_NAME = "default"

    def __init__(self, storage: StorageBackend, v: Variables):
        self._storage = storage
        self.logger = get_logger(v, name="GraphQueryService")

    # --- internal helpers -------------------------------------------------

    async def _node_attr_map(self, workspace_id: str) -> dict[str, dict]:
        """Bulk-load active workspace memories into ``{id: {type, subtype}}``.

        One ``search_memories_by_filter`` read (same source rows as the
        graph-analysis node set). Callers fall back to ``get_memory`` for any
        id not present (e.g. an edge endpoint outside the active set).
        """
        memories = await self._storage.search_memories_by_filter(
            workspace_id,
            status="active",
            limit=_MEMORY_LIMIT,
        )
        return {
            mem.id: {
                "memory_type": getattr(mem, "memory_type", None),
                "memory_subtype": getattr(mem, "subtype", None),
            }
            for mem in memories
        }

    async def _hydrate_node(
        self,
        workspace_id: str,
        memory_id: str,
        attr_map: dict[str, dict],
    ) -> GraphNode:
        """Build a ``GraphNode`` for ``memory_id`` from the attr map / fallback."""
        attrs = attr_map.get(memory_id)
        if attrs is None:
            mem = await self._storage.get_memory(workspace_id, memory_id, track_access=False)
            if mem is not None:
                attrs = {
                    "memory_type": getattr(mem, "memory_type", None),
                    "memory_subtype": getattr(mem, "subtype", None),
                }
                attr_map[memory_id] = attrs
            else:
                attrs = {"memory_type": None, "memory_subtype": None}
        return GraphNode(
            memory_id=memory_id,
            memory_type=attrs.get("memory_type"),
            memory_subtype=attrs.get("memory_subtype"),
        )

    @staticmethod
    def _matches_direction(assoc: Association, frontier_id: str, direction: str) -> str | None:
        """Return the OTHER endpoint if ``assoc`` extends from ``frontier_id``
        in ``direction``, else None.

        ``outgoing``: only source==frontier (go to target).
        ``incoming``: only target==frontier (go to source).
        ``both``: either endpoint == frontier (go to the other).
        """
        if direction == "outgoing":
            if assoc.source_id == frontier_id:
                return assoc.target_id
            return None
        if direction == "incoming":
            if assoc.target_id == frontier_id:
                return assoc.source_id
            return None
        # both
        if assoc.source_id == frontier_id:
            return assoc.target_id
        if assoc.target_id == frontier_id:
            return assoc.source_id
        return None

    async def _bfs(
        self,
        workspace_id: str,
        roots: list[str],
        *,
        depth: int,
        relationship_types: list[str] | None,
        direction: str,
        node_limit: int,
    ) -> tuple[set[str], list[Association], bool]:
        """Deterministic BFS reachable-set + induced edge set with post-expansion cap.

        Two phases:

        1. **Reachability** (MAJOR-1/2 fix): compute the FULL ``depth``-hop
           reachable set WITHOUT applying ``node_limit`` mid-expansion. The cap
           is applied AFTER the expansion is complete so the retained node set is
           independent of edge-batch ordering. Both backends use the same
           canonical truncation order: ``sorted(roots) + sorted(non-roots)``,
           taking the first ``node_limit`` entries. The frontier is sorted at
           every hop and the batch is sorted before processing — full determinism.

        2. **Induced edges**: return ALL (relationship-filtered) associations
           whose both endpoints are in the retained visited set.

        Returns ``(visited_node_ids, induced_edges, truncated)``.
        """
        visited: set[str] = set(roots)
        frontier: list[str] = sorted(set(roots))

        # Phase 1: full bounded reachable set (NO cap mid-expansion).
        for _ in range(max(0, depth)):
            if not frontier:
                break
            batch = await self._storage.get_associations_batch(
                workspace_id=workspace_id,
                memory_ids=frontier,
                direction="both",
                relationships=relationship_types,
            )
            # Sort batch for determinism before processing.
            batch_sorted = sorted(batch, key=lambda a: (a.source_id, a.target_id, a.relationship))
            frontier_set = set(frontier)
            next_frontier: set[str] = set()
            for assoc in batch_sorted:
                for anchor in (assoc.source_id, assoc.target_id):
                    if anchor not in frontier_set:
                        continue
                    other = self._matches_direction(assoc, anchor, direction)
                    if other is None or other in visited:
                        continue
                    visited.add(other)
                    next_frontier.add(other)
            frontier = sorted(next_frontier)

        # Apply the node cap AFTER full expansion with canonical ordering:
        # roots first (sorted), then non-roots (sorted). This matches the AGE
        # backend's ``[memory_id] + sorted(visited - {memory_id})[:limit]`` and
        # ``roots + sorted(visited - roots)[:remaining]`` orderings.
        truncated = False
        if len(visited) > node_limit:
            truncated = True
            sorted_roots = sorted(set(roots))
            sorted_non_roots = sorted(visited - set(roots))
            ordered = sorted_roots + sorted_non_roots
            visited = set(ordered[:node_limit])

        # Phase 2: induced edge set over the retained visited nodes.
        induced = await self._storage.get_associations_batch(
            workspace_id=workspace_id,
            memory_ids=sorted(visited),
            direction="both",
            relationships=relationship_types,
        )
        edges_by_id: dict[str, Association] = {}
        for e in induced:
            if e.source_id in visited and e.target_id in visited:
                edges_by_id[e.id] = e
        edges = list(edges_by_id.values())
        edges.sort(key=lambda e: (e.source_id, e.target_id, e.relationship))
        return visited, edges, truncated

    async def _build_subgraph_dtos(
        self,
        workspace_id: str,
        visited: set[str],
        edges: list[Association],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Hydrate nodes + map edges into DTOs with deterministic ordering."""
        attr_map = await self._node_attr_map(workspace_id)
        nodes = [
            await self._hydrate_node(workspace_id, nid, attr_map)
            for nid in sorted(visited)
        ]
        edge_dtos = [
            GraphEdge(
                source_id=e.source_id,
                target_id=e.target_id,
                relationship=e.relationship,
                strength=e.strength,
            )
            for e in edges
        ]
        return nodes, edge_dtos

    # --- public API -------------------------------------------------------

    async def neighbors(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> NeighborResult:
        validate_graph_hops(depth, kind="depth")
        visited, edges, truncated = await self._bfs(
            workspace_id,
            [memory_id],
            depth=depth,
            relationship_types=relationship_types,
            direction=direction,
            node_limit=limit,
        )
        nodes, edge_dtos = await self._build_subgraph_dtos(workspace_id, visited, edges)
        return NeighborResult(
            root_id=memory_id,
            nodes=nodes,
            edges=edge_dtos,
            truncated=truncated,
        )

    async def k_hop_subgraph(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        depth: int = 2,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        node_limit: int = 200,
    ) -> SubgraphResult:
        validate_graph_hops(depth, kind="depth")
        visited, edges, truncated = await self._bfs(
            workspace_id,
            list(memory_ids),
            depth=depth,
            relationship_types=relationship_types,
            direction=direction,
            node_limit=node_limit,
        )
        nodes, edge_dtos = await self._build_subgraph_dtos(workspace_id, visited, edges)
        return SubgraphResult(
            nodes=nodes,
            edges=edge_dtos,
            root_ids=sorted(set(memory_ids)),
            truncated=truncated,
        )

    async def shortest_path(
        self,
        workspace_id: str,
        src_id: str,
        dst_id: str,
        *,
        max_hops: int = 5,
        relationship_types: list[str] | None = None,
    ) -> PathResult:
        validate_graph_hops(max_hops, kind="max_hops")
        # Trivial path: src == dst.
        attr_map = await self._node_attr_map(workspace_id)
        if src_id == dst_id:
            node = await self._hydrate_node(workspace_id, src_id, attr_map)
            return PathResult(found=True, nodes=[node], edges=[], hops=0)

        # BFS tracking predecessor + the edge used to reach each node.
        # Tie-break: frontiers sorted at every hop + batch sorted by
        # (source_id, target_id, relationship) → among equal-length paths the
        # lexicographically smallest node-id sequence is always selected.
        # The AGE backend mirrors this with ORDER BY on path node sequences.
        predecessor: dict[str, tuple[str, Association]] = {}
        visited: set[str] = {src_id}
        frontier: list[str] = [src_id]

        found = False
        for _ in range(max(0, max_hops)):
            if not frontier or found:
                break
            batch = await self._storage.get_associations_batch(
                workspace_id=workspace_id,
                memory_ids=frontier,
                direction="both",
                relationships=relationship_types,
            )
            frontier_set = set(frontier)
            # Deterministic edge ordering: lexicographic tie-break for equal paths.
            batch_sorted = sorted(batch, key=lambda e: (e.source_id, e.target_id, e.relationship))
            next_frontier: list[str] = []
            for assoc in batch_sorted:
                for anchor in (assoc.source_id, assoc.target_id):
                    if anchor not in frontier_set:
                        continue
                    other = assoc.target_id if anchor == assoc.source_id else assoc.source_id
                    if other in visited:
                        continue
                    visited.add(other)
                    predecessor[other] = (anchor, assoc)
                    next_frontier.append(other)
                    if other == dst_id:
                        found = True
            frontier = sorted(set(next_frontier))

        if dst_id not in predecessor:
            return PathResult(found=False, nodes=[], edges=[], hops=0)

        # Reconstruct path src..dst.
        rev_nodes: list[str] = [dst_id]
        rev_edges: list[Association] = []
        cur = dst_id
        while cur != src_id:
            prev, edge = predecessor[cur]
            rev_edges.append(edge)
            rev_nodes.append(prev)
            cur = prev
        path_node_ids = list(reversed(rev_nodes))
        path_edges = list(reversed(rev_edges))

        nodes = [await self._hydrate_node(workspace_id, nid, attr_map) for nid in path_node_ids]
        edge_dtos = [
            GraphEdge(
                source_id=e.source_id,
                target_id=e.target_id,
                relationship=e.relationship,
                strength=e.strength,
            )
            for e in path_edges
        ]
        return PathResult(found=True, nodes=nodes, edges=edge_dtos, hops=len(path_edges))

    async def typed_pattern(
        self,
        workspace_id: str,
        *,
        relationship: str,
        limit: int = 100,
    ) -> list[PatternMatch]:
        attr_map = await self._node_attr_map(workspace_id)
        node_ids = sorted(attr_map.keys())
        if not node_ids:
            return []
        edges = await self._storage.get_associations_batch(
            workspace_id=workspace_id,
            memory_ids=node_ids,
            direction="both",
            relationships=[relationship],
        )
        edges_sorted = sorted(edges, key=lambda e: (e.source_id, e.target_id, e.relationship))
        matches = [
            PatternMatch(
                source_id=e.source_id,
                target_id=e.target_id,
                relationship=e.relationship,
                strength=e.strength,
            )
            for e in edges_sorted
            if e.relationship == relationship
        ]
        return matches[:limit]

    async def relationship_rollup(
        self,
        workspace_id: str,
    ) -> list[RelationshipRollup]:
        counts = await self._storage.count_associations_by_relationship(workspace_id)
        rollups = [
            RelationshipRollup(relationship=rel, count=cnt)
            for rel, cnt in counts.items()
        ]
        # Stable ordering: count desc, relationship asc.
        rollups.sort(key=lambda r: (-r.count, r.relationship))
        return rollups

    async def entity_neighborhood(
        self,
        workspace_id: str,
        entity_id: str,
        *,
        hops: int = 1,
        memory_limit: int = 100,
        entity_limit: int = 50,
    ) -> EntityNeighborhood:
        """OSS relational fallback for the C2 entity-neighborhood primitive.

        Reads the registry MEMBER edges directly (no graph database): the
        entity's own members give ``memories_for_entity``; the OTHER entities
        whose members include any of those memories give
        ``entities_co_mentioned`` (ranked by shared-memory count). Contract-
        equivalent to the enterprise AGE backend; parity relaxed only under
        truncation (the cap orders deterministically so the retained set is
        stable regardless of row order).

        Empty/disabled registry: ``list_workspace_entity_members`` returns ``[]``
        (or raises ``NotImplementedError`` on a backend without the registry),
        either of which yields an empty neighborhood — never an error.
        """
        _members_limit = 100000
        try:
            members = await self._storage.list_workspace_entity_members(
                workspace_id, limit=_members_limit
            )
        except NotImplementedError:
            return EntityNeighborhood(entity_id=entity_id)

        # Signal the silent correctness ceiling: if the cap was hit, the
        # co-mention counts and the reachable neighborhood may be incomplete.
        members_truncated = len(members) == _members_limit

        # Memories this entity mentions (memory_id -> role; first role wins on a
        # deterministic (entity_id, memory_id, role)-ordered list).
        own_roles: dict[str, str] = {}
        for m in members:
            if m["entity_id"] != entity_id:
                continue
            own_roles.setdefault(m["memory_id"], m["role"])

        own_memory_ids = set(own_roles)

        # Co-mention: for every OTHER entity, count how many of THIS entity's
        # mentioned memories it also mentions (distinct memories).
        co_shared: dict[str, set[str]] = {}
        for m in members:
            other = m["entity_id"]
            if other == entity_id:
                continue
            if m["memory_id"] in own_memory_ids:
                co_shared.setdefault(other, set()).add(m["memory_id"])

        # Hydrate memory type/subtype for the entity's mentioned memories.
        attr_map = await self._node_attr_map(workspace_id)
        mentioned = []
        for mid in sorted(own_memory_ids):
            attrs = attr_map.get(mid, {})
            mentioned.append(
                EntityMentionedMemory(
                    memory_id=mid,
                    role=own_roles[mid],
                    memory_type=attrs.get("memory_type"),
                    memory_subtype=attrs.get("memory_subtype"),
                )
            )

        # Resolve co-mentioned entity labels/types (registry-wide entity list).
        label_by_id: dict[str, dict] = {}
        if co_shared:
            try:
                entities = await self._storage.list_workspace_entities(workspace_id)
                label_by_id = {e["id"]: e for e in entities}
            except NotImplementedError:
                label_by_id = {}

        co_entities = [
            CoMentionedEntity(
                entity_id=eid,
                label=(label_by_id.get(eid) or {}).get("canonical_name"),
                entity_type=(label_by_id.get(eid) or {}).get("entity_type"),
                shared_memory_count=len(shared),
            )
            for eid, shared in co_shared.items()
        ]
        # Rank: shared-memory count desc, then entity_id asc (deterministic).
        co_entities.sort(key=lambda c: (-c.shared_memory_count, c.entity_id))

        truncated = members_truncated
        if len(mentioned) > memory_limit:
            truncated = True
            mentioned = mentioned[:memory_limit]
        if len(co_entities) > entity_limit:
            truncated = True
            co_entities = co_entities[:entity_limit]

        return EntityNeighborhood(
            entity_id=entity_id,
            memories_for_entity=mentioned,
            entities_co_mentioned=co_entities,
            truncated=truncated,
        )


    async def fragments_for_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int = 100,
    ) -> FragmentResult:
        """OSS relational fallback for the P4.5 fragment-traversal primitive.

        Reads the workspace's ``subtype="fact"`` memories directly (no graph
        database) and keeps those whose ``metadata["source_id"]`` is
        ``memory_id`` — the facts decomposed from that memory. Contract-
        equivalent to the enterprise AGE backend (which reads the
        Fragment/DERIVED_FROM vertices+edges); parity relaxed only under
        truncation (the cap orders by fragment_id asc so the retained set is
        stable regardless of row order).

        DARK + MEASURABLE — NOT wired into recall. No flag here: the OSS fallback
        reads the fact memories that the fact channel already produces; an empty/
        disabled fact layer simply yields an empty result, never an error.
        """
        _FACT_LIMIT = 100000
        facts = await self._storage.search_memories_by_filter(
            workspace_id,
            subtypes=["fact"],
            status="active",
            limit=_FACT_LIMIT,
        )
        # Signal the silent correctness ceiling: a clipped fact scan may miss
        # fragments of this memory.
        facts_truncated = len(facts) == _FACT_LIMIT

        derived = [
            DerivedFragment(
                fragment_id=mem.id,
                content=getattr(mem, "content", None),
                source_id=memory_id,
            )
            for mem in facts
            if (getattr(mem, "metadata", None) or {}).get("source_id") == memory_id
        ]
        # Deterministic ordering by fragment_id asc — matches the AGE backend's
        # ``ORDER BY f.id`` so both backends agree on the truncation boundary.
        derived.sort(key=lambda d: d.fragment_id)

        truncated = facts_truncated
        if len(derived) > limit:
            truncated = True
            derived = derived[:limit]

        return FragmentResult(
            source_id=memory_id,
            fragments=derived,
            truncated=truncated,
        )


class DefaultGraphQueryServicePlugin(GraphQueryServicePluginBase):
    """Plugin for the relational (OSS default) graph query service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> RelationalGraphQueryService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return RelationalGraphQueryService(storage=storage, v=v)
