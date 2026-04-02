"""Workspace contradiction scan task handler — daily scheduled scan."""

from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..services.contradiction import EXT_CONTRADICTION_SERVICE, ContradictionService
from ..services.storage import EXT_STORAGE_BACKEND
from ..services.storage.base import StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule


class WorkspaceContradictionScanHandler(TaskHandlerPlugin):
    """
    Daily contradiction scan task handler.

    Iterates over all workspaces and calls scan_workspace() on each one to detect
    contradictions that were missed during incremental check_new_memory calls.
    Runs once per day by default.
    """

    def get_task_type(self) -> str:
        return "workspace_contradiction_scan"

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        return TaskSchedule(
            interval_seconds=86400,  # Once per day
            default_payload={},
        )

    async def handle(self, v: Variables, payload: dict) -> None:
        contradiction_service: ContradictionService = self.get_extension(EXT_CONTRADICTION_SERVICE, v)
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get("workspace_id")

        if workspace_id:
            # Single-workspace scan (e.g., triggered on-demand with a specific workspace)
            logger.info("Running contradiction scan for workspace %s", workspace_id)
            records = await contradiction_service.scan_workspace(workspace_id)
            logger.info(
                "Contradiction scan complete for workspace %s: %d new contradiction(s) found",
                workspace_id,
                len(records),
            )
        else:
            # Scan all workspaces
            logger.info("Running contradiction scan for all workspaces")
            workspaces = await storage.list_workspaces()
            total_found = 0
            for workspace in workspaces:
                try:
                    records = await contradiction_service.scan_workspace(workspace.id)
                    total_found += len(records)
                    if records:
                        logger.info(
                            "Workspace %s: %d new contradiction(s) found",
                            workspace.id,
                            len(records),
                        )
                    else:
                        logger.debug("Workspace %s: no new contradictions found", workspace.id)
                except Exception as exc:
                    logger.error(
                        "Contradiction scan failed for workspace %s: %s",
                        workspace.id,
                        exc,
                    )
            logger.info(
                "Contradiction scan complete: %d workspace(s) scanned, %d total contradiction(s) found",
                len(workspaces),
                total_found,
            )
