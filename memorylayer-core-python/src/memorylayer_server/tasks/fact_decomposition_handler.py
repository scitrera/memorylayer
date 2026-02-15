"""
Fact Decomposition Task Handler.

Background task handler for decomposing composite memories into atomic facts
via the TaskService infrastructure.

When triggered:
1. Fetches the composite memory from storage
2. Calls ExtractionService.decompose_to_facts(content)
3. For each atomic fact: routes through MemoryService.ingest_fact() for full
   pipeline treatment (dedup, store, associate, contradiction, tier gen)
4. Creates PART_OF associations from each fact to the parent
5. Archives the parent memory (status = ARCHIVED)
"""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger, Variables

from ..models.association import AssociateInput
from ..models.memory import MemoryType, MemorySubtype, MemoryStatus, RememberInput
from ..services.storage import StorageBackend, EXT_STORAGE_BACKEND
from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services.extraction import ExtractionService, EXT_EXTRACTION_SERVICE
from ..services.memory import EXT_MEMORY_SERVICE, MemoryService


class FactDecompositionTaskHandler(TaskHandlerPlugin):
    """
    On-demand fact decomposition task handler.

    Triggered after a composite memory is stored to decompose it into
    atomic facts. Each fact is routed through MemoryService.ingest_fact()
    so it receives the full per-fact pipeline (dedup, association,
    contradiction check, tier generation).

    No recurring schedule -- runs only when explicitly submitted via
    the task service.
    """

    def get_task_type(self) -> str:
        return 'decompose_facts'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None  # On-demand only, not recurring

    async def handle(self, v: Variables, payload: dict) -> None:
        """Execute fact decomposition for a composite memory.

        Args:
            v: Variables object
            payload: Must contain 'memory_id' and 'workspace_id'
        """
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        extraction_service: ExtractionService = self.get_extension(EXT_EXTRACTION_SERVICE, v)
        logger: Logger = get_logger(v, name=self.get_task_type())

        # Get memory service for per-fact pipeline
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)

        memory_id = payload.get('memory_id')
        workspace_id = payload.get('workspace_id')

        if not memory_id or not workspace_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, memory_id=%s",
                workspace_id, memory_id,
            )
            return

        # 1. Fetch composite memory
        memory = await storage.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            logger.warning("Memory %s not found in workspace %s, skipping decomposition", memory_id, workspace_id)
            return

        # Skip if already archived or deleted
        if memory.status != MemoryStatus.ACTIVE:
            logger.debug("Memory %s is not active (status=%s), skipping decomposition", memory_id, memory.status)
            return

        # 2. Decompose into atomic facts
        logger.info("Decomposing memory %s into atomic facts", memory_id)
        facts = await extraction_service.decompose_to_facts(memory.content)

        # If only one fact returned, it is already atomic -- skip decomposition
        if len(facts) <= 1:
            logger.debug("Memory %s is already atomic (1 fact returned), skipping", memory_id)
            return

        # 3. Process each fact through the full pipeline via MemoryService
        created_fact_ids = []
        for fact in facts:
            # Determine type/subtype: prefer fact-level overrides, fall back to parent
            fact_type = memory.type
            fact_subtype = memory.subtype
            try:
                if fact.get("type"):
                    fact_type = MemoryType(fact["type"])
            except ValueError:
                pass
            try:
                if fact.get("subtype"):
                    fact_subtype = MemorySubtype(fact["subtype"])
            except ValueError:
                pass

            fact_input = RememberInput(
                content=fact["content"],
                type=fact_type,
                subtype=fact_subtype,
                importance=memory.importance,
                tags=memory.tags,
                metadata={**(memory.metadata or {}), "decomposed_from": memory_id},
                context_id=memory.context_id,
                user_id=memory.user_id,
            )

            result = await memory_service.ingest_fact(
                workspace_id=workspace_id,
                input=fact_input,
                source_memory_id=memory_id,
                inline=False,  # Sub-tasks still go to background
            )
            if result:
                created_fact_ids.append(result.id)

        # 4. Create PART_OF associations from each fact to the parent
        for fact_id in created_fact_ids:
            try:
                assoc_input = AssociateInput(
                    source_id=fact_id,
                    target_id=memory_id,
                    relationship="part_of",
                    strength=1.0,
                    metadata={"auto_generated": True, "source": "fact_decomposition"},
                )
                await storage.create_association(workspace_id, assoc_input)
            except Exception as e:
                logger.warning(
                    "Failed to create PART_OF association from %s to %s: %s",
                    fact_id, memory_id, e,
                )

        # 5. Archive the parent memory
        try:
            await storage.update_memory(
                workspace_id,
                memory_id,
                status=MemoryStatus.ARCHIVED.value,
            )
            logger.info(
                "Decomposed memory %s into %d atomic facts and archived parent",
                memory_id, len(created_fact_ids),
            )
        except Exception as e:
            logger.warning("Failed to archive parent memory %s: %s", memory_id, e)
