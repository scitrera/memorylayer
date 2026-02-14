"""Auto-enrich task handler for association creation and type classification.

Delegates association work to AssociationService (which owns ontology-based
relationship classification).  Type classification uses ExtractionService.
"""
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services.storage import StorageBackend, EXT_STORAGE_BACKEND
from ..services.embedding import EmbeddingService, EXT_EMBEDDING_SERVICE
from ..services.association import EXT_ASSOCIATION_SERVICE
from ..services.extraction import EXT_EXTRACTION_SERVICE


class AutoEnrichTaskHandler(TaskHandlerPlugin):
    """
    On-demand auto-enrich task handler.

    Triggered after a new memory is stored to:
    1. Find similar memories and delegate association creation to the
       AssociationService (which handles LLM-classified relationship types
       via the OntologyService).
    2. Optionally reclassify the memory type using LLM-based extraction
       when the ``classify_type`` flag is set in the payload.

    No recurring schedule -- runs only when explicitly submitted via the
    task service.
    """

    # Vector search parameters
    CANDIDATE_LIMIT = 5
    SIMILARITY_THRESHOLD = 0.6

    def get_task_type(self) -> str:
        return 'auto_enrich'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        # No recurring schedule - triggered on-demand after remember
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        logger: Logger = get_logger(v, name=self.get_task_type())

        memory_id = payload.get('memory_id')
        workspace_id = payload.get('workspace_id')
        content = payload.get('content')
        embedding = payload.get('embedding')

        if not memory_id or not workspace_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, memory_id=%s",
                workspace_id, memory_id,
            )
            return

        if not content:
            logger.warning("Missing content in auto_enrich payload for memory %s", memory_id)
            return

        # Resolve services from the framework
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)

        # If no embedding was provided in the payload, generate one
        if embedding is None:
            try:
                embedding_service: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)
                embedding = await embedding_service.embed(content)
            except Exception as e:
                logger.warning("Failed to generate embedding for memory %s: %s", memory_id, e)
                return

        # Search for similar memories using vector search
        try:
            similar_memories = await storage.search_memories(
                workspace_id=workspace_id,
                query_embedding=embedding,
                limit=self.CANDIDATE_LIMIT,
                min_relevance=self.SIMILARITY_THRESHOLD,
            )
        except Exception as e:
            logger.warning("Failed to search similar memories for %s: %s", memory_id, e)
            return

        # Delegate association creation to AssociationService
        if similar_memories:
            candidates = [
                (mem.id, score) for mem, score in similar_memories
                if mem.id != memory_id
            ]
            if candidates:
                try:
                    association_service = self.get_extension(EXT_ASSOCIATION_SERVICE, v)
                    associations = await association_service.auto_associate(
                        workspace_id=workspace_id,
                        new_memory_id=memory_id,
                        similar_memories=candidates,
                        threshold=self.SIMILARITY_THRESHOLD,
                        new_memory_content=content,
                    )
                    logger.info(
                        "Created %d auto-association(s) for memory %s in workspace %s",
                        len(associations), memory_id, workspace_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Auto-association failed for memory %s: %s", memory_id, e,
                    )
        else:
            logger.debug("No similar memories found for %s in workspace %s", memory_id, workspace_id)

        # Type classification (when flag is set)
        if payload.get('classify_type', False):
            try:
                extraction_service = self.get_extension(EXT_EXTRACTION_SERVICE, v)
                classified_type, classified_subtype = await extraction_service.classify_content(content)

                # Fetch current memory to compare types
                current_memory = await storage.get_memory(workspace_id, memory_id)
                if current_memory and current_memory.type != classified_type:
                    update_kwargs = {'type': classified_type.value}
                    if classified_subtype is not None:
                        update_kwargs['subtype'] = classified_subtype.value
                    await storage.update_memory(
                        workspace_id=workspace_id,
                        memory_id=memory_id,
                        **update_kwargs,
                    )
                    logger.info(
                        "Reclassified memory %s from %s to %s",
                        memory_id, current_memory.type, classified_type,
                    )
            except Exception as e:
                logger.debug("Type classification skipped for %s: %s", memory_id, e)
