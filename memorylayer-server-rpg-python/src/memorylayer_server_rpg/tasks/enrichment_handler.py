# SPDX-License-Identifier: Apache-2.0
"""RPG enrichment task handler — on-demand LLM enrichment of code structure graphs."""

import logging

from memorylayer_server.services.llm import EXT_LLM_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.storage.base import StorageBackend
from memorylayer_server.services.tasks import TaskHandlerPlugin, TaskSchedule
from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from memorylayer_server_rpg.services.enrichment import RpgEnrichmentService

RPG_ENRICHMENT_TASK = "rpg_enrichment"


class RpgEnrichmentTaskHandler(TaskHandlerPlugin):
    """On-demand RPG graph enrichment task handler.

    Registered as a multi-extension plugin under EXT_MULTI_TASK_HANDLERS.
    No schedule — triggered on demand (e.g., after a full sync).

    Payload fields:
        workspace_id (str, required): Workspace to enrich.
        context_id (str, optional): Context partition. Defaults to "rpg".
        phases (list[str], optional): Phases to run. Defaults to all three:
            ["features", "descriptions", "data_flows"].
        source (str, optional): Origin of the trigger, for logging only
            (e.g. "post_sync", "api").
    """

    def get_task_type(self) -> str:
        return RPG_ENRICHMENT_TASK

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        # On-demand only — no periodic schedule
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        llm_service = self.get_extension(EXT_LLM_SERVICE, v)
        logger: logging.Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get("workspace_id")
        if not workspace_id:
            logger.error("RPG enrichment task missing required 'workspace_id' in payload")
            return

        context_id = payload.get("context_id", "rpg")
        phases = payload.get("phases")  # None = all phases
        source = payload.get("source", "unknown")

        logger.info(
            "Running RPG enrichment for workspace %s (context=%s, phases=%s, source=%s)",
            workspace_id,
            context_id,
            phases,
            source,
        )

        enrichment = RpgEnrichmentService(storage=storage, llm_service=llm_service)

        try:
            result = await enrichment.enrich(
                workspace_id=workspace_id,
                context_id=context_id,
                phases=phases,
            )
            logger.info(
                "RPG enrichment complete for workspace %s: %s",
                workspace_id,
                result.get("phases", {}),
            )
        except Exception as exc:
            logger.error(
                "RPG enrichment failed for workspace %s: %s",
                workspace_id,
                exc,
            )
