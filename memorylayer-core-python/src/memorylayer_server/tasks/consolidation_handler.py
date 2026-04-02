"""Memory consolidation task handler — daily scheduled merge of low-importance similar memories."""

from datetime import UTC
from logging import Logger
from typing import Any

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..config import (
    DEFAULT_MEMORYLAYER_CONSOLIDATION_ENABLED,
    DEFAULT_MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE,
    DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE,
    DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY,
    MEMORYLAYER_CONSOLIDATION_ENABLED,
    MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE,
    MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE,
    MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY,
)
from ..services.storage import EXT_STORAGE_BACKEND
from ..services.storage.base import StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..utils import dot_product as _dot_product


def _is_enabled(v: Variables) -> bool:
    """Return True if consolidation is enabled via config."""
    raw = v.get(MEMORYLAYER_CONSOLIDATION_ENABLED, DEFAULT_MEMORYLAYER_CONSOLIDATION_ENABLED)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("1", "true", "yes")


def _merge_memories_simplified(primary: Any, others: list[Any]) -> dict:
    """Simplified merge: keeps primary content, unions tags, deep-merges metadata,
    and sets importance to min(max(a, b) * 1.1, 1.0).

    Args:
        primary: The memory with highest importance (kept as base)
        others: Other memories in the cluster being merged into primary

    Returns:
        Dict of update kwargs for storage.update_memory()
    """
    merged_tags: set[str] = set(primary.tags or [])
    merged_metadata: dict = dict(primary.metadata or {})
    max_importance = primary.importance

    provenance_ids: list[str] = []

    for mem in others:
        # Union tags
        merged_tags.update(mem.tags or [])
        # Deep-merge metadata (other's keys fill in missing primary keys)
        for k, val in (mem.metadata or {}).items():
            if k not in merged_metadata:
                merged_metadata[k] = val
        # Track max importance across cluster
        max_importance = max(max_importance, mem.importance)
        provenance_ids.append(mem.id)

    # Boost importance slightly (capped at 1.0)
    new_importance = min(max_importance * 1.1, 1.0)

    # Record provenance in metadata
    existing_provenance = merged_metadata.get("consolidated_from", [])
    if isinstance(existing_provenance, list):
        merged_metadata["consolidated_from"] = existing_provenance + provenance_ids
    else:
        merged_metadata["consolidated_from"] = provenance_ids

    return {
        "tags": list(merged_tags),
        "metadata": merged_metadata,
        "importance": new_importance,
    }


def _find_clusters(
    memories: list[Any],
    min_similarity: float,
    min_cluster_size: int,
) -> list[list[Any]]:
    """Group memories into clusters where all pairs have similarity >= min_similarity.

    Uses a greedy union-find style approach: for each memory, find all others
    that are mutually similar above the threshold. Returns clusters of size
    >= min_cluster_size.

    Args:
        memories: List of memory objects with .embedding and .importance fields
        min_similarity: Minimum pairwise similarity to group two memories
        min_cluster_size: Minimum cluster size to be worth consolidating

    Returns:
        List of clusters (each cluster is a list of memories)
    """
    n = len(memories)
    if n < min_cluster_size:
        return []

    # Build adjacency: index i -> set of adjacent indices
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        emb_i = memories[i].embedding
        if not emb_i:
            continue
        for j in range(i + 1, n):
            emb_j = memories[j].embedding
            if not emb_j:
                continue
            sim = _dot_product(emb_i, emb_j)
            if sim >= min_similarity:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Union-find to group connected components
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in adjacency[i]:
            union(i, j)

    # Group by root
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for indices in groups.values():
        if len(indices) >= min_cluster_size:
            clusters.append([memories[idx] for idx in indices])

    return clusters


class ConsolidationTaskHandler(TaskHandlerPlugin):
    """
    Daily memory consolidation task handler.

    Finds clusters of low-importance memories with high mutual similarity and
    merges each cluster into the highest-importance member. Tracks provenance
    in the surviving memory's metadata.

    Disabled by default; enable with MEMORYLAYER_CONSOLIDATION_ENABLED=true.
    """

    def get_task_type(self) -> str:
        return "memory_consolidation"

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        if not _is_enabled(v):
            return None
        return TaskSchedule(
            interval_seconds=86400,  # Once per day
            default_payload={},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        if not _is_enabled(v):
            return

        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        logger: Logger = get_logger(v, name=self.get_task_type())

        min_cluster_size = int(
            v.get(
                MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE,
                DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE,
            )
        )
        max_importance = float(
            v.get(
                MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE,
                DEFAULT_MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE,
            )
        )
        min_similarity = float(
            v.get(
                MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY,
                DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY,
            )
        )

        workspace_id = payload.get("workspace_id")

        if workspace_id:
            workspaces_to_process = [workspace_id]
        else:
            workspaces = await storage.list_workspaces()
            workspaces_to_process = [ws.id for ws in workspaces]

        total_merged = 0
        total_deleted = 0

        for ws_id in workspaces_to_process:
            try:
                merged, deleted = await self._consolidate_workspace(
                    storage,
                    logger,
                    ws_id,
                    min_cluster_size,
                    max_importance,
                    min_similarity,
                )
                total_merged += merged
                total_deleted += deleted
            except Exception as exc:
                logger.error("Consolidation failed for workspace %s: %s", ws_id, exc)

        logger.info(
            "Consolidation complete: %d memories merged (primary updated), %d memories deleted",
            total_merged,
            total_deleted,
        )

    async def _consolidate_workspace(
        self,
        storage: StorageBackend,
        logger: Logger,
        workspace_id: str,
        min_cluster_size: int,
        max_importance: float,
        min_similarity: float,
    ) -> tuple[int, int]:
        """Run consolidation on a single workspace.

        Returns:
            (merged_count, deleted_count) tuple
        """
        from datetime import datetime

        epoch = datetime(2000, 1, 1, tzinfo=UTC)

        # Collect all low-importance memories with embeddings
        candidates = []
        offset = 0
        batch_size = 200

        while True:
            batch = await storage.get_recent_memories(
                workspace_id,
                created_after=epoch,
                limit=batch_size,
                detail_level="full",
                offset=offset,
            )
            if not batch:
                break

            for item in batch:
                if isinstance(item, dict):
                    mem_id = item.get("id")
                    importance = item.get("importance", 1.0)
                    if importance is not None and importance <= max_importance and mem_id:
                        mem = await storage.get_memory(workspace_id, mem_id, track_access=False)
                        if mem and mem.embedding and not getattr(mem, "pinned", False):
                            candidates.append(mem)
                else:
                    importance = getattr(item, "importance", 1.0)
                    if importance <= max_importance and getattr(item, "embedding", None):
                        if not getattr(item, "pinned", False):
                            candidates.append(item)

            offset += len(batch)
            if len(batch) < batch_size:
                break

        if len(candidates) < min_cluster_size:
            logger.debug(
                "Workspace %s: only %d candidate(s) below importance threshold, skipping",
                workspace_id,
                len(candidates),
            )
            return 0, 0

        logger.info(
            "Workspace %s: found %d candidate memories for consolidation",
            workspace_id,
            len(candidates),
        )

        clusters = _find_clusters(candidates, min_similarity, min_cluster_size)
        logger.info("Workspace %s: found %d cluster(s) to consolidate", workspace_id, len(clusters))

        merged_count = 0
        deleted_count = 0

        for cluster in clusters:
            # Primary = highest importance member
            primary = max(cluster, key=lambda m: m.importance)
            others = [m for m in cluster if m.id != primary.id]

            if not others:
                continue

            # Compute merged update fields
            updates = _merge_memories_simplified(primary, others)

            # Apply updates to primary
            try:
                await storage.update_memory(workspace_id, primary.id, **updates)
                merged_count += 1
            except Exception as exc:
                logger.error(
                    "Failed to update primary memory %s during consolidation: %s",
                    primary.id,
                    exc,
                )
                continue

            # Soft-delete the other cluster members
            for mem in others:
                try:
                    await storage.delete_memory(workspace_id, mem.id, hard=False)
                    deleted_count += 1
                    logger.debug(
                        "Consolidated memory %s into primary %s (workspace %s)",
                        mem.id,
                        primary.id,
                        workspace_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to delete memory %s during consolidation: %s",
                        mem.id,
                        exc,
                    )

        logger.info(
            "Workspace %s consolidation: %d cluster(s) processed, %d primaries updated, %d memories deleted",
            workspace_id,
            len(clusters),
            merged_count,
            deleted_count,
        )
        return merged_count, deleted_count
