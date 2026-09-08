# SPDX-License-Identifier: Apache-2.0
"""RPG maintenance task handler — periodic graph health and cleanup operations."""

import logging

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.storage.base import StorageBackend
from memorylayer_server.services.tasks import TaskHandlerPlugin, TaskSchedule
from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from memorylayer_server_rpg.services.maintenance import RpgMaintenanceService

RPG_MAINTENANCE_TASK = "rpg_maintenance"

_VALID_OPERATIONS = frozenset({"validate", "cleanup", "statistics"})


class RpgMaintenanceTaskHandler(TaskHandlerPlugin):
    """Periodic RPG graph maintenance task handler.

    Registered as a multi-extension plugin under EXT_MULTI_TASK_HANDLERS.
    Runs every 24 hours by default (validate + statistics across all workspaces).

    Payload fields:
        workspace_id (str, optional): Limit to a single workspace. Omit for all.
        operation (str, optional): One of "validate", "cleanup", "statistics".
            Defaults to "statistics".
    """

    def get_task_type(self) -> str:
        return RPG_MAINTENANCE_TASK

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        return TaskSchedule(
            interval_seconds=24 * 3600,  # Every 24 hours
            default_payload={"operation": "statistics"},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        logger: logging.Logger = get_logger(v, name=self.get_task_type())

        maintenance = RpgMaintenanceService(storage)
        operation = payload.get("operation", "statistics")
        workspace_id = payload.get("workspace_id")

        if operation not in _VALID_OPERATIONS:
            logger.error("Unknown RPG maintenance operation: %s", operation)
            return

        if workspace_id:
            workspaces_to_process = [workspace_id]
        else:
            workspaces = await storage.list_workspaces()
            workspaces_to_process = [ws.id for ws in workspaces]

        logger.info(
            "Running RPG maintenance operation '%s' for %d workspace(s)",
            operation,
            len(workspaces_to_process),
        )

        for ws_id in workspaces_to_process:
            try:
                await self._run_operation(maintenance, logger, operation, ws_id)
            except Exception as exc:
                logger.error("RPG maintenance '%s' failed for workspace %s: %s", operation, ws_id, exc)

    async def _run_operation(
        self,
        maintenance: RpgMaintenanceService,
        logger: logging.Logger,
        operation: str,
        workspace_id: str,
    ) -> None:
        """Execute a single maintenance operation for one workspace."""
        if operation == "validate":
            result = await maintenance.validate_graph(workspace_id)
            logger.info(
                "RPG validate workspace %s: %d nodes, %d edges, %d dangling, %d orphans, healthy=%s",
                workspace_id,
                result["node_count"],
                result["edge_count"],
                result["dangling_edges"],
                result["orphan_nodes"],
                result["healthy"],
            )
        elif operation == "cleanup":
            result = await maintenance.cleanup_stale_nodes(workspace_id)
            logger.info(
                "RPG cleanup workspace %s: %d checked, %d deleted",
                workspace_id,
                result["checked_count"],
                result["deleted_count"],
            )
        elif operation == "statistics":
            result = await maintenance.compute_statistics(workspace_id)
            logger.info(
                "RPG statistics workspace %s: %d nodes, %d edges",
                workspace_id,
                result["total_nodes"],
                result["total_edges"],
            )
