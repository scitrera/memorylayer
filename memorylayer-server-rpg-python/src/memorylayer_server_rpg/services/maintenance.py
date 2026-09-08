# SPDX-License-Identifier: Apache-2.0
"""RPG maintenance service — periodic graph health operations."""

import logging
from datetime import UTC, datetime, timedelta

from memorylayer_server.services.storage.base import StorageBackend

from .service import RPG_RELATIONSHIP_TYPES, RpgService

logger = logging.getLogger(__name__)

DEFAULT_INTENT_MAX_AGE_HOURS = 24


class RpgMaintenanceService:
    """Periodic maintenance operations for RPG graphs."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage
        self._rpg = RpgService(storage)

    async def validate_graph(self, workspace_id: str) -> dict:
        """Check graph health: dangling edges, orphan nodes.

        Uses batch association lookup to avoid N per-node queries.

        Args:
            workspace_id: Target workspace.

        Returns:
            Report dict with dangling_edges, orphan_nodes, node_count, edge_count, healthy.
        """
        nodes = await self._rpg._get_rpg_nodes(workspace_id)
        node_memory_ids: set[str] = {m.id for m in nodes}

        # Batch fetch all outgoing edges in one call
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        all_outgoing = await self._rpg._get_rpg_edges(workspace_id, node_id_to_memory_id)

        # Build per-node association maps from the batch result
        nodes_with_outgoing: set[str] = set()
        dangling_edges = 0
        total_edges = 0
        for assoc in all_outgoing:
            nodes_with_outgoing.add(assoc.source_id)
            total_edges += 1
            if assoc.target_id not in node_memory_ids:
                dangling_edges += 1
                logger.debug(
                    "Dangling edge %s: target %s not found in workspace %s",
                    assoc.id,
                    assoc.target_id,
                    workspace_id,
                )

        # Batch fetch all incoming edges to detect nodes with no connections at all
        all_incoming = await self._storage.get_associations_batch(
            workspace_id=workspace_id,
            memory_ids=list(node_memory_ids),
            direction="incoming",
            relationships=list(RPG_RELATIONSHIP_TYPES),
        )
        nodes_with_incoming: set[str] = {assoc.target_id for assoc in all_incoming}

        orphan_node_ids: list[str] = []
        for mem in nodes:
            if mem.id not in nodes_with_outgoing and mem.id not in nodes_with_incoming:
                # Skip the sync metadata sentinel — it's expected to have no edges
                if (mem.metadata or {}).get("rpg_node_id") == "_rpg_sync_sentinel":
                    continue
                orphan_node_ids.append((mem.metadata or {}).get("rpg_node_id", mem.id))

        healthy = dangling_edges == 0 and len(orphan_node_ids) == 0
        logger.info(
            "RPG graph validate for workspace %s: %d nodes, %d edges, %d dangling, %d orphans",
            workspace_id,
            len(nodes),
            total_edges,
            dangling_edges,
            len(orphan_node_ids),
        )

        return {
            "workspace_id": workspace_id,
            "node_count": len(nodes),
            "edge_count": total_edges,
            "dangling_edges": dangling_edges,
            "orphan_nodes": len(orphan_node_ids),
            "orphan_node_ids": orphan_node_ids,
            "healthy": healthy,
        }

    async def cleanup_stale_nodes(self, workspace_id: str) -> dict:
        """Remove RPG nodes that have no edges at all (orphans).

        Uses batch association lookup to avoid N per-node queries. Deletes
        all associations for each orphan before deleting the node itself.

        Args:
            workspace_id: Target workspace.

        Returns:
            Cleanup stats dict with deleted_count, checked_count.
        """
        nodes = await self._rpg._get_rpg_nodes(workspace_id)
        node_memory_ids = [m.id for m in nodes]
        deleted_count = 0

        # Batch fetch all outgoing and incoming associations in two calls
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        all_outgoing = await self._rpg._get_rpg_edges(workspace_id, node_id_to_memory_id)
        all_incoming = await self._storage.get_associations_batch(
            workspace_id=workspace_id,
            memory_ids=node_memory_ids,
            direction="incoming",
            relationships=list(RPG_RELATIONSHIP_TYPES),
        )

        nodes_with_connections: set[str] = set()
        for assoc in all_outgoing:
            nodes_with_connections.add(assoc.source_id)
        for assoc in all_incoming:
            nodes_with_connections.add(assoc.target_id)

        for mem in nodes:
            if mem.id in nodes_with_connections:
                continue
            node_id = (mem.metadata or {}).get("rpg_node_id", mem.id)
            # Skip the sync metadata sentinel — it's expected to have no edges
            if node_id == "_rpg_sync_sentinel":
                continue
            # Delete associations (defensive — there should be none, but clean up anyway)
            await self._rpg._delete_node_associations(workspace_id, mem.id)
            try:
                await self._storage.delete_memory(
                    workspace_id=workspace_id,
                    memory_id=mem.id,
                    hard=True,
                )
                deleted_count += 1
                logger.debug(
                    "Deleted orphan RPG node %s (memory %s) from workspace %s",
                    node_id,
                    mem.id,
                    workspace_id,
                )
            except Exception as exc:
                logger.error("Failed to delete orphan RPG node %s: %s", node_id, exc)

        logger.info(
            "RPG cleanup for workspace %s: %d checked, %d orphan nodes deleted",
            workspace_id,
            len(nodes),
            deleted_count,
        )

        return {
            "workspace_id": workspace_id,
            "checked_count": len(nodes),
            "deleted_count": deleted_count,
        }

    async def cleanup_stale_intents(self, workspace_id: str, max_age_hours: int = DEFAULT_INTENT_MAX_AGE_HOURS) -> dict:
        """Remove intent overlays older than max_age_hours.

        Intent overlays (rpg-intent-*) are created when tasks start and should
        be cleaned up when tasks merge. If a task is abandoned, the intent
        overlay persists forever. This operation garbage-collects stale ones.

        Args:
            workspace_id: Target workspace.
            max_age_hours: Maximum age in hours before an intent is considered stale.

        Returns:
            Dict with cleaned_count and checked_count.
        """
        overlays = await self._rpg.list_overlays(workspace_id)
        intent_overlays = [o for o in overlays if o.get("context_id", "").startswith("rpg-intent-")]

        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        cleaned_count = 0

        for overlay in intent_overlays:
            context_id = overlay["context_id"]
            # Check the overlay's creation time from sync metadata
            try:
                status = await self._rpg.get_status(workspace_id, context_id=context_id)
                last_sync_at = status.get("last_sync_at")
                if last_sync_at:
                    sync_time = datetime.fromisoformat(last_sync_at)
                    # Ensure timezone-aware comparison
                    if sync_time.tzinfo is None:
                        sync_time = sync_time.replace(tzinfo=UTC)
                    if sync_time > cutoff:
                        continue  # Still fresh, skip

                # If no sync_at timestamp, or it's older than cutoff, delete it
                await self._rpg.delete_overlay(workspace_id, context_id)
                cleaned_count += 1
                logger.info(
                    "Cleaned stale intent overlay %s (workspace %s)",
                    context_id,
                    workspace_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to check/clean intent overlay %s: %s",
                    context_id,
                    exc,
                )

        logger.info(
            "RPG intent cleanup for workspace %s: %d checked, %d stale intents removed",
            workspace_id,
            len(intent_overlays),
            cleaned_count,
        )

        return {
            "workspace_id": workspace_id,
            "checked_count": len(intent_overlays),
            "cleaned_count": cleaned_count,
            "max_age_hours": max_age_hours,
        }

    async def compute_statistics(self, workspace_id: str) -> dict:
        """Compute and cache RPG statistics for a workspace.

        Counts nodes by type, edges by relationship type, and stores results
        in workspace RPG sync metadata.

        Args:
            workspace_id: Target workspace.

        Returns:
            Stats dict with node_type_counts, edge_type_counts, total_nodes, total_edges.
        """
        nodes = await self._rpg._get_rpg_nodes(workspace_id)

        node_type_counts: dict[str, int] = {}
        edge_type_counts: dict[str, int] = {}
        total_edges = 0

        for mem in nodes:
            node_type = mem.subtype or "rpg_unknown"
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        # Batch fetch all outgoing associations in one call
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        all_outgoing = await self._rpg._get_rpg_edges(workspace_id, node_id_to_memory_id)
        for assoc in all_outgoing:
            rel = assoc.relationship
            edge_type_counts[rel] = edge_type_counts.get(rel, 0) + 1
            total_edges += 1

        stats = {
            "workspace_id": workspace_id,
            "total_nodes": len(nodes),
            "total_edges": total_edges,
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        }

        logger.info(
            "RPG statistics for workspace %s: %d nodes, %d edges",
            workspace_id,
            len(nodes),
            total_edges,
        )

        return stats
