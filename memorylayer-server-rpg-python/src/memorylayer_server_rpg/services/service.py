# SPDX-License-Identifier: Apache-2.0
"""RPG service — bulk sync, subgraph extraction, and search for code structure graphs.

RPG nodes are stored as Memory entities (type=SEMANTIC, subtype=rpg_*).
RPG edges are stored as Association entities with code_structure relationships.

The service operates directly on the StorageBackend for efficiency — RPG nodes
are structural data that don't need the full remember/recall pipeline (embeddings,
dedup, decomposition, etc.).
"""

import hashlib
import logging
import time as _time
from datetime import UTC, datetime

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import Memory, MemoryType, RememberInput
from memorylayer_server.models.workspace import Context
from memorylayer_server.services.storage.base import StorageBackend
from memorylayer_server.utils.id_generation import generate_id  # noqa: F401

from memorylayer_server_rpg.services.ontology_contributor import RPG_RELATIONSHIP_TYPES as _RPG_REL_TYPES_DICT
from memorylayer_server_rpg.services.ontology_contributor import RPG_SUBTYPES as _RPG_SUBTYPES_DICT

logger = logging.getLogger(__name__)

# RPG subtypes that map to node types — derived from the contributor's
# single source of truth so the two stay in sync automatically.
RPG_SUBTYPES: frozenset[str] = frozenset({s for subtypes in _RPG_SUBTYPES_DICT.values() for s in subtypes})

# Code structure relationship types — derived from the single source of truth
# in ontology_contributor.py so the two stay in sync automatically.
RPG_RELATIONSHIP_TYPES: frozenset[str] = frozenset(_RPG_REL_TYPES_DICT.keys())

# Metadata key used to store the RPG node_id on the Memory entity
_NODE_ID_KEY = "rpg_node_id"
_RPG_TAG = "rpg"


def _content_hash(text: str) -> str:
    """Compute SHA-256 content hash."""
    return hashlib.sha256(text.encode()).hexdigest()


class RpgService:
    """Service for managing RPG (Repository Planning Graph) data in MemoryLayer.

    RPG nodes are stored as Memory entities with:
      - type = SEMANTIC
      - subtype = rpg_* (rpg_file, rpg_class, etc.)
      - metadata["rpg_node_id"] = the RPG node ID (path or path:QualifiedName)
      - tags = ["rpg"]
      - content = description (or name if no description)

    RPG edges are stored as Associations with code_structure relationship types.
    """

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    async def sync(
        self,
        workspace_id: str,
        nodes: list[dict],
        edges: list[dict],
        full_sync: bool = False,
        source_commit: str | None = None,
        context_id: str = "rpg",
        task_id: str | None = None,
    ) -> dict:
        """Sync an RPG graph snapshot into MemoryLayer.

        Args:
            workspace_id: Target workspace.
            nodes: List of node dicts (from RpgNodeInput).
            edges: List of edge dicts (from RpgEdgeInput).
            full_sync: If True, remove RPG nodes/edges not in this payload.
            source_commit: Git commit SHA for provenance.
            context_id: Context partition (default "rpg" for canonical graph).
            task_id: Optional task ID for provenance tracking.

        Returns:
            Dict with counts: nodes_created, nodes_updated, nodes_deleted,
            edges_created, edges_updated, edges_deleted, sync_time_ms.
        """
        # Validate context_id to prevent pollution of non-RPG partitions
        if context_id != "rpg" and not context_id.startswith("rpg-"):
            raise ValueError(f"context_id must be 'rpg' or start with 'rpg-'. Got: {context_id}")

        t0 = _time.monotonic()

        # Persist the globally unique context row id while keeping the friendly
        # logical context name in RPG metadata and the public API.
        effective_context_row_id = await self._resolve_context_row_id(workspace_id, context_id, create=True)
        assert effective_context_row_id is not None

        stats = {
            "nodes_created": 0,
            "nodes_updated": 0,
            "nodes_deleted": 0,
            "nodes_existing": 0,
            "edges_created": 0,
            "edges_updated": 0,
            "edges_deleted": 0,
            "edges_existing": 0,
        }

        # Build lookup of existing RPG nodes in this workspace (scoped to
        # context). Filter by the resolved ROW id so reads line up with
        # writes (memories.context_id stores the row id, not the name).
        existing_nodes = await self._get_rpg_nodes(workspace_id, context_id=effective_context_row_id)
        existing_by_node_id: dict[str, Memory] = {m.metadata.get(_NODE_ID_KEY): m for m in existing_nodes if m.metadata.get(_NODE_ID_KEY)}
        # Seed nodes_existing with current pre-sync count; will be adjusted after upsert loop
        _pre_sync_existing_count = len(existing_by_node_id)

        # Track which node_ids are in the payload (for full_sync cleanup)
        payload_node_ids: set[str] = set()
        # Map node_id -> memory_id for edge creation (seed with all existing nodes)
        node_id_to_memory_id: dict[str, str] = {nid: mem.id for nid, mem in existing_by_node_id.items()}

        # --- Upsert nodes ---
        for node in nodes:
            node_id = node["node_id"]
            payload_node_ids.add(node_id)

            desc = node.get("description") or node.get("name", node_id)
            content = desc
            metadata = dict(node.get("metadata", {}))
            metadata[_NODE_ID_KEY] = node_id
            metadata["rpg_path"] = node.get("path", "")
            metadata["rpg_name"] = node.get("name", "")
            if node.get("language"):
                metadata["rpg_language"] = node["language"]
            if node.get("parent_id"):
                metadata["rpg_parent_id"] = node["parent_id"]
            if source_commit:
                metadata["rpg_source_commit"] = source_commit
            if context_id:
                metadata["rpg_context_id"] = context_id
            if task_id:
                metadata["rpg_task_id"] = task_id

            node_type = node.get("node_type", "rpg_file")

            existing = existing_by_node_id.get(node_id)
            if existing:
                # Update existing node
                await self._storage.update_memory(
                    workspace_id=workspace_id,
                    memory_id=existing.id,
                    content=content,
                    content_hash=_content_hash(content),
                    subtype=node_type,
                    metadata=metadata,
                )
                node_id_to_memory_id[node_id] = existing.id
                stats["nodes_updated"] += 1
            else:
                # Create new node
                remember_input = RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subtype=node_type,
                    importance=0.5,
                    tags=[_RPG_TAG],
                    metadata=metadata,
                    # Use the resolved context ROW id (not the user-friendly
                    # name) so the `memories.context_id REFERENCES contexts(id)`
                    # FK is satisfied. The user-facing context name remains in
                    # `metadata["rpg_context_id"]` for downstream filtering.
                    context_id=effective_context_row_id,
                    observer_id=task_id,
                )
                memory = await self._storage.create_memory(
                    workspace_id=workspace_id,
                    input=remember_input,
                )
                node_id_to_memory_id[node_id] = memory.id
                stats["nodes_created"] += 1

        # --- Full sync cleanup: delete nodes not in payload ---
        if full_sync:
            for node_id, mem in existing_by_node_id.items():
                if node_id not in payload_node_ids:
                    await self._storage.delete_memory(
                        workspace_id=workspace_id,
                        memory_id=mem.id,
                        hard=True,
                    )
                    stats["nodes_deleted"] += 1

        # Nodes that existed before this sync and were not touched by the payload
        stats["nodes_existing"] = _pre_sync_existing_count - stats["nodes_updated"]

        # --- Upsert edges ---
        # Build lookup of existing RPG edges: key -> (assoc_id, existing_metadata)
        existing_edges = await self._get_rpg_edges(workspace_id, node_id_to_memory_id)
        existing_edge_keys: dict[tuple[str, str, str], str] = {}
        existing_edge_metadata: dict[tuple[str, str, str], dict] = {}
        for assoc in existing_edges:
            key = (assoc.source_id, assoc.target_id, assoc.relationship)
            existing_edge_keys[key] = assoc.id
            existing_edge_metadata[key] = assoc.metadata or {}

        payload_edge_keys: set[tuple[str, str, str]] = set()

        for edge in edges:
            source_node_id = edge["source_id"]
            target_node_id = edge["target_id"]
            relationship = edge["relationship"]

            source_mem_id = node_id_to_memory_id.get(source_node_id)
            target_mem_id = node_id_to_memory_id.get(target_node_id)

            if not source_mem_id or not target_mem_id:
                logger.warning(
                    "Skipping edge %s -> %s (%s): node not found",
                    source_node_id,
                    target_node_id,
                    relationship,
                )
                continue

            edge_key = (source_mem_id, target_mem_id, relationship)
            payload_edge_keys.add(edge_key)

            edge_metadata = dict(edge.get("metadata", {}))
            edge_metadata["rpg_source_node_id"] = source_node_id
            edge_metadata["rpg_target_node_id"] = target_node_id

            if edge_key in existing_edge_keys:
                # Edge already exists — update metadata if it has changed
                if edge_metadata != existing_edge_metadata.get(edge_key):
                    await self._storage.update_association(
                        workspace_id=workspace_id,
                        association_id=existing_edge_keys[edge_key],
                        metadata=edge_metadata,
                    )
                    logger.debug(
                        "Updated edge metadata %s -> %s (%s)",
                        source_node_id,
                        target_node_id,
                        relationship,
                    )
                    stats["edges_updated"] += 1
                else:
                    # Edge exists and metadata is unchanged
                    stats["edges_existing"] += 1
            else:
                # Create new edge
                assoc_input = AssociateInput(
                    source_id=source_mem_id,
                    target_id=target_mem_id,
                    relationship=relationship,
                    strength=edge.get("strength", 1.0),
                    metadata=edge_metadata,
                )
                await self._storage.create_association(
                    workspace_id=workspace_id,
                    input=assoc_input,
                )
                stats["edges_created"] += 1

        # --- Full sync cleanup: delete edges not in payload ---
        if full_sync:
            for edge_key, assoc_id in existing_edge_keys.items():
                if edge_key not in payload_edge_keys:
                    deleted = await self._storage.delete_association(
                        workspace_id=workspace_id,
                        association_id=assoc_id,
                    )
                    if deleted:
                        stats["edges_deleted"] += 1

        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        stats["sync_time_ms"] = elapsed_ms
        stats["source_commit"] = source_commit

        # Store sync metadata on workspace.
        # On partial sync, the stats only reflect the payload-scoped changes,
        # not the full graph totals — so we increment instead of overwrite.
        await self._update_sync_metadata(
            workspace_id,
            source_commit,
            stats,
            context_id=context_id,
            context_row_id=effective_context_row_id,
            full_sync=full_sync,
        )

        logger.info(
            "RPG sync complete for workspace %s: %d created, %d updated, %d deleted nodes; %d created, %d updated edges in %dms",
            workspace_id,
            stats["nodes_created"],
            stats["nodes_updated"],
            stats["nodes_deleted"],
            stats["edges_created"],
            stats["edges_updated"],
            elapsed_ms,
        )

        return stats

    # ------------------------------------------------------------------
    # Subgraph
    # ------------------------------------------------------------------

    async def get_subgraph(
        self,
        workspace_id: str,
        root_path: str | None = None,
        root_node_id: str | None = None,
        depth: int = 3,
        node_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        context_id: str = "rpg",
        max_nodes: int = 2000,
    ) -> dict:
        """Extract an RPG subgraph rooted at a path or node_id.

        Args:
            workspace_id: Target workspace.
            root_path: File path to find the root node (e.g. "src/main.py").
            root_node_id: Direct RPG node ID (takes precedence over root_path).
            depth: Maximum traversal depth from root.
            node_types: Filter results to these node types.
            relationship_types: Filter edges to these types.
            context_id: Context partition (default "rpg" for canonical graph).

        Returns:
            Dict with nodes, edges, root_id, depth, total_nodes, total_edges.
        """
        if relationship_types is None:
            relationship_types = list(RPG_RELATIONSHIP_TYPES)

        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_row_id is None:
            return {
                "nodes": [],
                "edges": [],
                "root_id": root_node_id or root_path,
                "depth": depth,
                "total_nodes": 0,
                "total_edges": 0,
            }

        # Resolve root memory
        root_memory = None
        if root_node_id:
            root_memory = await self._find_node_by_id(workspace_id, root_node_id, context_id=context_id)
        elif root_path:
            root_memory = await self._find_node_by_path(workspace_id, root_path, context_id=context_id)

        if not root_memory:
            return {
                "nodes": [],
                "edges": [],
                "root_id": root_node_id or root_path,
                "depth": depth,
                "total_nodes": 0,
                "total_edges": 0,
            }

        # Use graph traversal to find connected nodes
        graph_result = await self._storage.traverse_graph(
            workspace_id=workspace_id,
            start_id=root_memory.id,
            max_depth=depth,
            direction="both",
            relationships=relationship_types,
        )

        # Collect unique memory IDs from paths (with node count cap)
        visited_ids: set[str] = {root_memory.id}
        edges_out = []
        truncated = False

        for path in graph_result.paths:
            for node_id in path.nodes:
                if len(visited_ids) >= max_nodes:
                    truncated = True
                    break
                visited_ids.add(node_id)
            if truncated:
                break
            for edge in path.edges:
                edges_out.append(
                    {
                        "source_id": edge.metadata.get("rpg_source_node_id", edge.source_id),
                        "target_id": edge.metadata.get("rpg_target_node_id", edge.target_id),
                        "relationship": edge.relationship,
                        "strength": edge.strength,
                        "metadata": edge.metadata,
                        "association_id": edge.id,
                    }
                )

        # Fetch full memory objects for each node
        nodes_out = []
        for mem_id in visited_ids:
            mem = await self._storage.get_memory(workspace_id, mem_id, track_access=False)
            if not mem:
                continue
            if not mem.subtype or not mem.subtype.startswith("rpg_"):
                continue
            if mem.context_id != context_row_id or self._is_sync_sentinel(mem):
                continue
            if node_types and mem.subtype not in node_types:
                continue

            nodes_out.append(self._memory_to_node_response(mem, root_memory.id))

        # Deduplicate edges
        seen_edges: set[tuple[str, str, str]] = set()
        unique_edges = []
        included_node_ids = {node["node_id"] for node in nodes_out}
        for e in edges_out:
            if e["source_id"] not in included_node_ids or e["target_id"] not in included_node_ids:
                continue
            key = (e["source_id"], e["target_id"], e["relationship"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {
            "nodes": nodes_out,
            "edges": unique_edges,
            "root_id": root_memory.metadata.get(_NODE_ID_KEY, root_memory.id),
            "depth": depth,
            "total_nodes": len(nodes_out),
            "total_edges": len(unique_edges),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_nodes(
        self,
        workspace_id: str,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 20,
        context_id: str = "rpg",
    ) -> dict:
        """Search RPG nodes by text (name/description/path).

        Uses full-text search on the Memory content field.

        Args:
            workspace_id: Target workspace.
            query: Search text.
            node_types: Filter to specific RPG node types.
            limit: Max results.
            context_id: Context partition (default "rpg" for canonical graph).

        Returns:
            Dict with nodes list and total_count.
        """
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_row_id is None:
            return {"nodes": [], "total_count": 0, "query": query}

        # Use FTS with the persisted context row id pushed down to storage.
        results = await self._storage.full_text_search(
            workspace_id=workspace_id,
            query=query,
            limit=limit * 3,  # Overfetch to allow filtering by subtype/tags
            context_id=context_row_id,
        )

        nodes = []
        for mem in results:
            if not mem.subtype or not mem.subtype.startswith("rpg_"):
                continue
            if node_types and mem.subtype not in node_types:
                continue
            if _RPG_TAG not in (mem.tags or []):
                continue
            nodes.append(self._memory_to_node_response(mem))
            if len(nodes) >= limit:
                break

        return {
            "nodes": nodes,
            "total_count": len(nodes),
            "query": query,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self, workspace_id: str, context_id: str | None = None) -> dict:
        """Get RPG status for a workspace.

        Prefers sentinel-cached node/edge counts from the last sync to avoid
        materializing all nodes. Falls back to a full node fetch only when no
        sentinel exists.

        Args:
            workspace_id: Target workspace.
            context_id: Filter to this context. None = all contexts.

        Returns:
            Dict with has_rpg, node_count, edge_count, last_sync_commit, last_sync_at.
        """
        sync_meta = await self._get_sync_metadata(workspace_id, context_id=context_id)

        # Use sentinel-cached counts when available (avoids full node materialisation)
        if sync_meta.get("last_node_count") is not None:
            node_count = sync_meta["last_node_count"]
            edge_count = sync_meta.get("last_edge_count", 0)
        else:
            # No sentinel yet — fall back to full fetch
            nodes = await self._get_rpg_nodes(workspace_id, context_id=context_id)
            node_count = len(nodes)
            edge_count = sync_meta.get("last_edge_count", 0)

        return {
            "workspace_id": workspace_id,
            "has_rpg": node_count > 0,
            "node_count": node_count,
            "edge_count": edge_count,
            "last_sync_commit": sync_meta.get("last_sync_commit"),
            "last_sync_at": sync_meta.get("last_sync_at"),
        }

    # ------------------------------------------------------------------
    # Overlay management
    # ------------------------------------------------------------------

    async def get_merged_subgraph(
        self,
        workspace_id: str,
        base_context: str = "rpg",
        overlay_context: str = "",
        root_path: str | None = None,
        depth: int = 3,
    ) -> dict:
        """Get subgraph merging canonical + overlay. Overlay nodes override canonical."""
        base = await self.get_subgraph(workspace_id, root_path=root_path, depth=depth, context_id=base_context)
        if not overlay_context:
            return base
        overlay = await self.get_subgraph(workspace_id, root_path=root_path, depth=depth, context_id=overlay_context)
        # Merge: overlay nodes take precedence over base nodes with same node_id
        merged_nodes = {n["node_id"]: n for n in base.get("nodes", [])}
        for n in overlay.get("nodes", []):
            merged_nodes[n["node_id"]] = n  # overlay wins
        merged_edges = list(base.get("edges", [])) + list(overlay.get("edges", []))
        # Deduplicate edges
        seen: set[tuple[str, str, str]] = set()
        unique_edges: list[dict] = []
        for e in merged_edges:
            key = (e.get("source_id"), e.get("target_id"), e.get("relationship"))
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)
        return {
            "nodes": list(merged_nodes.values()),
            "edges": unique_edges,
            "root_id": base.get("root_id") or overlay.get("root_id"),
            "depth": depth,
            "total_nodes": len(merged_nodes),
            "total_edges": len(unique_edges),
            "truncated": bool(base.get("truncated") or overlay.get("truncated")),
        }

    async def list_overlays(self, workspace_id: str) -> list[dict]:
        """List all active RPG overlay contexts (rpg-task-* and rpg-intent-*).

        Queries sync metadata sentinels (one per context) instead of all nodes
        to avoid materialising the full node set.
        """
        # Sentinels are tagged with [_RPG_TAG, "rpg_sync_meta"] — one per context.
        # This is far cheaper than fetching all nodes just to find distinct context_ids.
        sentinels = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            tags=[_RPG_TAG, "rpg_sync_meta"],
            status="active",
            limit=10000,
        )
        overlays = []
        for sentinel in sentinels:
            ctx = (sentinel.metadata or {}).get("rpg_context_id") or getattr(sentinel, "context_id", None)
            if ctx and ctx.startswith("rpg-"):
                node_count = (sentinel.metadata or {}).get("rpg_last_node_count", 0)
                overlays.append({"context_id": ctx, "node_count": node_count})
        # Fall back to full node scan if no sentinels found (e.g. pre-sentinel data)
        if not overlays:
            all_nodes = await self._get_rpg_nodes(workspace_id, context_id=None)
            contexts: dict[str, int] = {}
            for node in all_nodes:
                ctx = (node.metadata or {}).get("rpg_context_id") or node.context_id or "_default"
                if ctx.startswith("rpg-"):
                    contexts[ctx] = contexts.get(ctx, 0) + 1
            return [{"context_id": k, "node_count": v} for k, v in sorted(contexts.items())]
        return sorted(overlays, key=lambda o: o["context_id"])

    async def delete_overlay(self, workspace_id: str, context_id: str) -> dict:
        """Delete all RPG nodes in a specific overlay context."""
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_row_id is None:
            return {"context_id": context_id, "deleted": 0}
        nodes = await self._get_rpg_nodes(workspace_id, context_id=context_id)
        deleted = 0
        for mem in nodes:
            # Delete all associations (edges) for this node before deleting the node
            await self._delete_node_associations(workspace_id, mem.id)
            try:
                await self._storage.delete_memory(workspace_id, mem.id, hard=True)
                deleted += 1
            except Exception:
                logger.warning("Failed to delete RPG overlay node %s in context %s", mem.id, context_id)
        sentinel = await self._find_sync_sentinel(workspace_id, context_id)
        if sentinel:
            await self._storage.delete_memory(workspace_id, sentinel.id, hard=True)
        await self._storage.delete_context(workspace_id, context_row_id)
        return {"context_id": context_id, "deleted": deleted}

    async def delete_nodes(self, workspace_id: str, node_ids: list[str], context_id: str = "rpg") -> dict:
        """Delete specific RPG nodes by their rpg_node_id values.

        Fetches existing nodes for the given context, matches by rpg_node_id in
        metadata, hard-deletes matching Memory entities, and returns counts.
        """
        if not node_ids:
            return {"deleted": 0, "not_found": 0}

        nodes = await self._get_rpg_nodes(workspace_id, context_id=context_id)
        node_id_set = set(node_ids)
        node_id_to_memory: dict[str, Memory] = {}
        for mem in nodes:
            rpg_node_id = (mem.metadata or {}).get(_NODE_ID_KEY)
            if rpg_node_id and rpg_node_id in node_id_set:
                node_id_to_memory[rpg_node_id] = mem

        deleted = 0
        for rpg_node_id, mem in node_id_to_memory.items():
            # Delete all associations (edges) for this node before deleting the node
            await self._delete_node_associations(workspace_id, mem.id)
            try:
                await self._storage.delete_memory(workspace_id, mem.id, hard=True)
                deleted += 1
            except Exception:
                logger.warning("Failed to delete RPG node %s (memory %s) in context %s", rpg_node_id, mem.id, context_id)

        not_found = len(node_id_set) - deleted
        logger.info(
            "RPG delete_nodes: workspace=%s, context=%s, requested=%d, deleted=%d, not_found=%d",
            workspace_id,
            context_id,
            len(node_ids),
            deleted,
            not_found,
        )
        return {"deleted": deleted, "not_found": not_found}

    async def delete_edges_bulk(self, workspace_id: str, association_ids: list[str]) -> dict:
        """Bulk-delete RPG edges (associations) by their IDs.

        Args:
            workspace_id: Workspace boundary.
            association_ids: Association UUIDs to delete.

        Returns:
            Dict with 'deleted' and 'not_found' counts.
        """
        if not association_ids:
            return {"deleted": 0, "not_found": 0}

        deleted = 0
        for assoc_id in association_ids:
            try:
                result = await self._storage.delete_association(
                    workspace_id=workspace_id,
                    association_id=assoc_id,
                )
                if result:
                    deleted += 1
            except Exception:
                logger.warning("Failed to delete association %s in workspace %s", assoc_id, workspace_id)

        not_found = len(association_ids) - deleted
        logger.info(
            "RPG delete_edges_bulk: workspace=%s, requested=%d, deleted=%d, not_found=%d",
            workspace_id,
            len(association_ids),
            deleted,
            not_found,
        )
        return {"deleted": deleted, "not_found": not_found}

    async def recompute_counts(self, workspace_id: str, context_id: str = "rpg") -> dict:
        """Recompute the sentinel node/edge counts via a live query.

        Used to repair corrupted sentinel counts (e.g. from partial sync overwrites).
        Counts all RPG nodes for the workspace and all outgoing RPG edges, then
        updates the sentinel metadata.

        Args:
            workspace_id: Target workspace.
            context_id: Context partition to recompute (default 'rpg').

        Returns:
            Dict with computed node_count and edge_count.
        """
        nodes = await self._get_rpg_nodes(workspace_id, context_id=context_id)
        # Filter out the sync sentinel itself
        real_nodes = [n for n in nodes if (n.metadata or {}).get(_NODE_ID_KEY) != "_rpg_sync_sentinel"]
        node_count = len(real_nodes)

        # Count edges by fetching outgoing associations for all nodes in batch
        memory_ids = [n.id for n in real_nodes]
        edge_count = 0
        if memory_ids:
            all_edges = await self._storage.get_associations_batch(
                workspace_id=workspace_id,
                memory_ids=memory_ids,
                direction="outgoing",
                relationships=list(RPG_RELATIONSHIP_TYPES),
            )
            edge_count = len(all_edges)

        # Update the sentinel with the live counts
        sentinel = await self._find_sync_sentinel(workspace_id, context_id)
        if sentinel:
            prev_meta = sentinel.metadata or {}
            new_meta = dict(prev_meta)
            new_meta["rpg_last_node_count"] = node_count
            new_meta["rpg_last_edge_count"] = edge_count
            await self._storage.update_memory(
                workspace_id=workspace_id,
                memory_id=sentinel.id,
                metadata=new_meta,
            )

        logger.info(
            "RPG recompute_counts: workspace=%s, context=%s, nodes=%d, edges=%d",
            workspace_id,
            context_id,
            node_count,
            edge_count,
        )
        return {
            "workspace_id": workspace_id,
            "context_id": context_id,
            "node_count": node_count,
            "edge_count": edge_count,
        }

    async def list_nodes(
        self,
        workspace_id: str,
        node_type: str | None = None,
        context_id: str = "rpg",
        limit: int = 5000,
    ) -> dict:
        """List RPG nodes, optionally filtered by node type.

        Args:
            workspace_id: Target workspace.
            node_type: Filter to this RPG subtype (e.g. 'rpg_file'). None = all RPG types.
            context_id: Context partition to query.
            limit: Maximum number of nodes to return.

        Returns:
            Dict with 'nodes' list and 'total_count'.
        """
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_row_id is None:
            return {"nodes": [], "total_count": 0}
        subtypes = [node_type] if node_type else list(RPG_SUBTYPES)
        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            subtypes=subtypes,
            tags=[_RPG_TAG],
            status="active",
            context_id=context_row_id,
            limit=limit,
        )
        nodes = [self._memory_to_node_response(mem) for mem in results if not self._is_sync_sentinel(mem)]
        return {"nodes": nodes, "total_count": len(nodes)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_context_row_id(
        self,
        workspace_id: str,
        context_id: str | None,
        *,
        create: bool = False,
    ) -> str | None:
        """Resolve a public RPG context name to the persisted context row ID."""
        if context_id is None:
            return None

        existing = await self._storage.get_context(workspace_id, context_id)
        if existing:
            return existing.id

        contexts = await self._storage.list_contexts(workspace_id)
        for context in contexts:
            if context.name == context_id:
                return context.id

        if not create:
            return None

        row_id = f"{workspace_id}:{context_id}"
        context = Context(
            id=row_id,
            workspace_id=workspace_id,
            name=context_id,
            description=f"RPG graph context: {context_id}",
            settings={"rpg": True},
        )
        try:
            await self._storage.create_context(workspace_id, context)
        except Exception:
            # Concurrent syncs can both observe a missing context before one
            # wins the globally unique row-id insert. Recover only when that
            # deterministic row now exists; otherwise preserve the real error.
            existing = await self._storage.get_context(workspace_id, row_id)
            if existing:
                return existing.id
            raise
        return row_id

    @staticmethod
    def _is_sync_sentinel(memory: Memory) -> bool:
        return (memory.metadata or {}).get(_NODE_ID_KEY) == "_rpg_sync_sentinel"

    async def _delete_node_associations(self, workspace_id: str, memory_id: str) -> None:
        """Delete all outgoing and incoming associations for a memory before deletion.

        Must be called before deleting a node to avoid dangling edges.
        """
        for direction in ("outgoing", "incoming"):
            try:
                assocs = await self._storage.get_associations(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    direction=direction,
                    relationships=list(RPG_RELATIONSHIP_TYPES),
                )
                for assoc in assocs:
                    await self._storage.delete_association(
                        workspace_id=workspace_id,
                        association_id=assoc.id,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to delete %s associations for memory %s in workspace %s: %s",
                    direction,
                    memory_id,
                    workspace_id,
                    exc,
                )

    async def _get_rpg_nodes(self, workspace_id: str, context_id: str | None = None) -> list[Memory]:
        """Get all RPG nodes in a workspace via filtered query on subtype and tags.

        Args:
            workspace_id: Target workspace.
            context_id: Filter to this context. None = all contexts.
        """
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_id is not None and context_row_id is None:
            return []

        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            subtypes=list(RPG_SUBTYPES),
            tags=[_RPG_TAG],
            status="active",
            context_id=context_row_id,
            limit=10000,
        )
        if len(results) >= 10000:
            logger.warning(
                "RPG node query hit 10,000 limit for workspace %s context %s - results may be truncated",
                workspace_id,
                context_id,
            )
        return [memory for memory in results if not self._is_sync_sentinel(memory)]

    async def _get_rpg_edges(
        self,
        workspace_id: str,
        node_id_to_memory_id: dict[str, str],
    ) -> list:
        """Get all RPG edges for the given nodes.

        Uses batch association lookup (single query) instead of per-node queries.
        """
        memory_ids = list(node_id_to_memory_id.values())
        if not memory_ids:
            return []

        return await self._storage.get_associations_batch(
            workspace_id=workspace_id,
            memory_ids=memory_ids,
            direction="outgoing",
            relationships=list(RPG_RELATIONSHIP_TYPES),
        )

    async def _find_node_by_id(self, workspace_id: str, node_id: str, context_id: str | None = None) -> Memory | None:
        """Find an RPG node by its rpg_node_id metadata."""
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_id is not None and context_row_id is None:
            return None
        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            tags=[_RPG_TAG],
            metadata_filter={_NODE_ID_KEY: node_id},
            status="active",
            context_id=context_row_id,
            limit=1,
        )
        return results[0] if results else None

    async def _find_node_by_path(self, workspace_id: str, path: str, context_id: str | None = None) -> Memory | None:
        """Find an RPG node by file path."""
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_id is not None and context_row_id is None:
            return None
        # Try by rpg_path first
        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            tags=[_RPG_TAG],
            metadata_filter={"rpg_path": path},
            status="active",
            context_id=context_row_id,
            limit=1,
        )
        if results:
            return results[0]

        # Fall back to rpg_node_id (for path-as-node-id cases)
        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            tags=[_RPG_TAG],
            metadata_filter={_NODE_ID_KEY: path},
            status="active",
            context_id=context_row_id,
            limit=1,
        )
        return results[0] if results else None

    def _memory_to_node_response(self, mem: Memory, root_memory_id: str | None = None) -> dict:
        """Convert a Memory entity to an RPG node response dict."""
        metadata = dict(mem.metadata or {})
        return {
            "node_id": metadata.get(_NODE_ID_KEY, mem.id),
            "node_type": mem.subtype or "rpg_file",
            "path": metadata.get("rpg_path", ""),
            "name": metadata.get("rpg_name", ""),
            "description": mem.content or "",
            "language": metadata.get("rpg_language"),
            "parent_id": metadata.get("rpg_parent_id"),
            "depth": 0,  # Will be set by caller if needed
            "metadata": {k: v for k, v in metadata.items() if not k.startswith("rpg_") and k != _NODE_ID_KEY},
            "memory_id": mem.id,
            "context_id": metadata.get("rpg_context_id") or getattr(mem, "context_id", None),
        }

    async def _update_sync_metadata(
        self,
        workspace_id: str,
        commit: str | None,
        stats: dict,
        *,
        context_id: str,
        context_row_id: str,
        full_sync: bool = False,
    ) -> None:
        """Store sync metadata as a sentinel RPG memory.

        On full_sync, the stats reflect the entire graph, so we overwrite the
        sentinel counts. On partial sync (incremental updates, enrichment),
        the stats only describe the payload-scoped changes, so we increment
        the existing sentinel counts by the delta.
        """
        try:
            sentinel = await self._find_sync_sentinel(workspace_id, context_id)

            if full_sync:
                # Full sync: stats describe the entire graph payload
                total_edges = stats["edges_created"] + stats["edges_updated"] + stats.get("edges_existing", 0)
                total_nodes = stats.get("nodes_created", 0) + stats.get("nodes_updated", 0)
                if stats.get("nodes_existing") is not None:
                    total_nodes += stats["nodes_existing"]
            else:
                # Partial sync: increment existing counts by the delta
                prev_meta = (sentinel.metadata if sentinel else {}) or {}
                prev_nodes = int(prev_meta.get("rpg_last_node_count", 0) or 0)
                prev_edges = int(prev_meta.get("rpg_last_edge_count", 0) or 0)
                total_nodes = prev_nodes + stats.get("nodes_created", 0) - stats.get("nodes_deleted", 0)
                total_edges = prev_edges + stats.get("edges_created", 0) - stats.get("edges_deleted", 0)
                # Guard against negative counts from corruption
                total_nodes = max(0, total_nodes)
                total_edges = max(0, total_edges)
            meta = {
                _NODE_ID_KEY: "_rpg_sync_sentinel",
                "rpg_last_sync_commit": commit,
                "rpg_last_sync_at": datetime.now(UTC).isoformat(),
                "rpg_last_sync_stats": {
                    "nodes_created": stats["nodes_created"],
                    "nodes_updated": stats["nodes_updated"],
                    "edges_created": stats["edges_created"],
                },
                "rpg_last_edge_count": total_edges,
                "rpg_last_node_count": total_nodes,
                "rpg_context_id": context_id,
            }
            content = "RPG sync metadata — last commit: %s" % (commit or "unknown")
            if sentinel:
                await self._storage.update_memory(
                    workspace_id=workspace_id,
                    memory_id=sentinel.id,
                    content=content,
                    content_hash=_content_hash(content),
                    metadata=meta,
                )
            else:
                inp = RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subtype="rpg_component",
                    importance=0.1,
                    tags=[_RPG_TAG, "rpg_sync_meta"],
                    metadata=meta,
                    context_id=context_row_id,
                )
                await self._storage.create_memory(workspace_id=workspace_id, input=inp)
        except Exception:
            logger.warning("Failed to update RPG sync metadata for workspace %s", workspace_id)

    async def _find_sync_sentinel(self, workspace_id: str, context_id: str) -> Memory | None:
        """Find the sync metadata sentinel memory."""
        context_row_id = await self._resolve_context_row_id(workspace_id, context_id)
        if context_row_id is None:
            return None
        results = await self._storage.search_memories_by_filter(
            workspace_id=workspace_id,
            tags=[_RPG_TAG, "rpg_sync_meta"],
            metadata_filter={_NODE_ID_KEY: "_rpg_sync_sentinel"},
            status="active",
            context_id=context_row_id,
            limit=1,
        )
        return results[0] if results else None

    async def _get_sync_metadata(self, workspace_id: str, context_id: str | None = None) -> dict:
        """Get sync metadata from the sentinel memory."""
        try:
            if context_id is not None:
                sentinel = await self._find_sync_sentinel(workspace_id, context_id)
                sentinels = [sentinel] if sentinel else []
            else:
                sentinels = await self._storage.search_memories_by_filter(
                    workspace_id=workspace_id,
                    tags=[_RPG_TAG, "rpg_sync_meta"],
                    metadata_filter={_NODE_ID_KEY: "_rpg_sync_sentinel"},
                    status="active",
                    limit=10000,
                )

            if sentinels:
                latest = max(
                    sentinels,
                    key=lambda item: (item.metadata or {}).get("rpg_last_sync_at", ""),
                )
                latest_metadata = latest.metadata or {}
                result = {
                    "last_sync_commit": latest_metadata.get("rpg_last_sync_commit"),
                    "last_sync_at": latest_metadata.get("rpg_last_sync_at"),
                    "last_edge_count": sum(int((item.metadata or {}).get("rpg_last_edge_count", 0) or 0) for item in sentinels),
                    "last_node_count": sum(int((item.metadata or {}).get("rpg_last_node_count", 0) or 0) for item in sentinels),
                }
                return result
        except Exception:
            logger.warning("Failed to get RPG sync metadata for workspace %s", workspace_id)
        return {}
