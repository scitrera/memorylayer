# SPDX-License-Identifier: Apache-2.0
"""RPG conflict detection service — finds file and symbol conflicts between tasks.

Detects when multiple tasks declare intent to modify the same files or symbols,
enabling coordination between concurrent agents.
"""

import logging

from memorylayer_server.services.storage.base import StorageBackend

from .service import RpgService

logger = logging.getLogger(__name__)


class RpgConflictService:
    """Server-side conflict detection for efficient cross-overlay queries."""

    def __init__(self, storage: StorageBackend, rpg_service: RpgService):
        self._storage = storage
        self._rpg = rpg_service

    async def detect_file_conflicts(self, workspace_id: str, task_id: str) -> dict:
        """Find files claimed by multiple task intents.

        Scans all rpg-intent-* overlays to find file path intersections
        with the target task's intent.

        Args:
            workspace_id: Target workspace.
            task_id: The task to check conflicts for.

        Returns:
            Dict with task_id, conflicts list, and total_conflicts count.
        """
        target_context = f"rpg-intent-{task_id}"

        # Get all overlays to find rpg-intent-* contexts
        all_overlays = await self._rpg.list_overlays(workspace_id)
        intent_overlays = [o for o in all_overlays if o["context_id"].startswith("rpg-intent-")]

        # Get target task's intent nodes
        target_nodes = await self._rpg._get_rpg_nodes(workspace_id, context_id=target_context)
        target_paths: set[str] = set()
        for node in target_nodes:
            path = (node.metadata or {}).get("rpg_path", "")
            if path:
                target_paths.add(path)

        if not target_paths:
            return {"task_id": task_id, "conflicts": [], "total_conflicts": 0}

        # Check each other intent overlay for file path intersections
        # conflict_map: file_path -> list of conflicting task IDs
        conflict_map: dict[str, list[str]] = {}

        for overlay in intent_overlays:
            other_context = overlay["context_id"]
            if other_context == target_context:
                continue

            other_task_id = other_context.removeprefix("rpg-intent-")
            other_nodes = await self._rpg._get_rpg_nodes(workspace_id, context_id=other_context)

            for node in other_nodes:
                other_path = (node.metadata or {}).get("rpg_path", "")
                if other_path and other_path in target_paths:
                    if other_path not in conflict_map:
                        conflict_map[other_path] = []
                    conflict_map[other_path].append(other_task_id)

        # Build response
        conflicts = []
        for file_path, task_ids in sorted(conflict_map.items()):
            severity = "high" if len(task_ids) >= 2 else "medium"
            conflicts.append(
                {
                    "file_path": file_path,
                    "task_ids": task_ids,
                    "severity": severity,
                }
            )

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        conflicts.sort(key=lambda c: severity_order.get(c["severity"], 3))

        return {
            "task_id": task_id,
            "conflicts": conflicts,
            "total_conflicts": len(conflicts),
        }

    async def detect_symbol_conflicts(
        self,
        workspace_id: str,
        task_id_a: str,
        task_id_b: str,
    ) -> dict:
        """Find symbols modified by both tasks.

        Compares node sets from both task overlays (rpg-task-*) to detect
        symbol-level overlaps.

        Args:
            workspace_id: Target workspace.
            task_id_a: First task ID.
            task_id_b: Second task ID.

        Returns:
            Dict with task_ids, conflicts list, and total_conflicts count.
        """
        context_a = f"rpg-task-{task_id_a}"
        context_b = f"rpg-task-{task_id_b}"

        nodes_a = await self._rpg._get_rpg_nodes(workspace_id, context_id=context_a)
        nodes_b = await self._rpg._get_rpg_nodes(workspace_id, context_id=context_b)

        # Filter to symbol nodes (exclude files and directories)
        file_types = {"rpg_file", "rpg_directory"}

        def _extract_symbols(nodes):
            symbols = {}
            for node in nodes:
                subtype = node.subtype or ""
                if subtype in file_types:
                    continue
                node_id = (node.metadata or {}).get("rpg_node_id", "")
                path = (node.metadata or {}).get("rpg_path", "")
                name = (node.metadata or {}).get("rpg_name", "")
                if node_id:
                    symbols[node_id] = {"path": path, "name": name, "type": subtype}
            return symbols

        symbols_a = _extract_symbols(nodes_a)
        symbols_b = _extract_symbols(nodes_b)

        if not symbols_a or not symbols_b:
            return {
                "task_ids": [task_id_a, task_id_b],
                "conflicts": [],
                "total_conflicts": 0,
            }

        # Build file-path index for B
        file_index_b: dict[str, list[str]] = {}
        for sym_id, info in symbols_b.items():
            path = info["path"]
            if path:
                file_index_b.setdefault(path, []).append(sym_id)

        conflicts = []
        seen: set[str] = set()

        for sym_id_a, info_a in symbols_a.items():
            # Exact symbol match = high severity
            if sym_id_a in symbols_b and sym_id_a not in seen:
                seen.add(sym_id_a)
                sym_type = info_a["type"].removeprefix("rpg_")
                conflicts.append(
                    {
                        "symbol_id": sym_id_a,
                        "file_path": info_a["path"],
                        "task_ids": [task_id_a, task_id_b],
                        "severity": "high",
                        "description": ('Both tasks modify {} "{}" in {}'.format(sym_type, info_a["name"], info_a["path"])),
                    }
                )
                continue

            # Different symbols in same file = medium severity
            same_file_syms = file_index_b.get(info_a["path"], [])
            for sym_id_b in same_file_syms:
                pair_key = "::".join(sorted([sym_id_a, sym_id_b]))
                if pair_key not in seen and sym_id_a != sym_id_b:
                    seen.add(pair_key)
                    info_b = symbols_b[sym_id_b]
                    conflicts.append(
                        {
                            "symbol_id": f"{sym_id_a}+{sym_id_b}",
                            "file_path": info_a["path"],
                            "task_ids": [task_id_a, task_id_b],
                            "severity": "medium",
                            "description": (
                                'Tasks touch different symbols in {}: "{}" ({}) vs "{}" ({})'.format(
                                    info_a["path"], info_a["name"], task_id_a, info_b["name"], task_id_b
                                )
                            ),
                        }
                    )

        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        conflicts.sort(key=lambda c: severity_order.get(c["severity"], 3))

        return {
            "task_ids": [task_id_a, task_id_b],
            "conflicts": conflicts,
            "total_conflicts": len(conflicts),
        }
