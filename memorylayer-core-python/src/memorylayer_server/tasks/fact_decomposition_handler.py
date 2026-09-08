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

from scitrera_app_framework import Variables, get_logger

from ..models.association import AssociateInput
from ..models.memory import MemoryStatus, MemoryType, RememberInput
from ..services.events import emit_event
from ..services.extraction import EXT_EXTRACTION_SERVICE, ExtractionService
from ..services.memory import EXT_MEMORY_SERVICE, MemoryService
from ..services.storage import EXT_STORAGE_BACKEND, StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule


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
        return "decompose_facts"

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
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

        memory_id = payload.get("memory_id")
        workspace_id = payload.get("workspace_id")
        # job_id is an optional correlation id threaded from the producer
        # (e.g. document ingestion). Read defensively so the handler tolerates
        # payloads that carry it; it is echoed back out on decompose_complete.
        job_id = payload.get("job_id")

        if not memory_id or not workspace_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, memory_id=%s",
                workspace_id,
                memory_id,
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

        # 2. Decompose into atomic facts. Anchor relative dates to the source
        # memory's effective time (event_time if known, else when it was
        # recorded) so decomposed facts resolve relative references to absolute
        # dates and surface their own event_time (write-time normalization).
        logger.info("Decomposing memory %s into atomic facts", memory_id)
        reference_time = memory.event_time or memory.created_at

        # OCR-free (visual) decomposition: a document-page memory ingested in
        # visual-only mode carries just a provenance placeholder as ``content``
        # (no transcript), flagged via ``metadata['visual_only']``. For these,
        # feed the page IMAGE to a vision model so real facts are read straight
        # from the page. The page image is resolved through a duck-typed storage
        # hook (``get_page_image_b64``) so this OSS handler stays backend-generic
        # (the in-memory/OSS backends simply lack the method). Text memories —
        # including transcript-backed page memories — are unaffected.
        images = None
        meta = memory.metadata or {}
        if meta.get("visual_only") and memory.source_page_id:
            image_resolver = getattr(storage, "get_page_image_b64", None)
            if image_resolver is not None:
                page_image_b64 = await image_resolver(memory.source_page_id)
                if page_image_b64:
                    images = [page_image_b64]
                    logger.info(
                        "Visual decomposition for memory %s from page %s image",
                        memory_id, memory.source_page_id,
                    )

        facts = await extraction_service.decompose_to_facts(
            memory.content, reference_time=reference_time, images=images,
        )

        # If only one fact returned, it is already atomic -- skip decomposition
        if len(facts) <= 1:
            logger.debug("Memory %s is already atomic (1 fact returned), skipping", memory_id)
            return

        # 3. Process each fact through the full pipeline via MemoryService
        created_fact_ids = []
        failed_facts = 0
        for fact in facts:
            # Determine type/subtype: prefer fact-level overrides, fall back to parent
            fact_type = memory.type
            fact_subtype = memory.subtype
            try:
                if fact.get("type"):
                    fact_type = MemoryType(fact["type"])
            except ValueError:
                pass
            if fact.get("subtype"):
                fact_subtype = fact["subtype"]

            fact_input = RememberInput(
                content=fact["content"],
                type=fact_type,
                subtype=fact_subtype,
                importance=memory.importance,
                tags=memory.tags,
                metadata={**(memory.metadata or {}), "decomposed_from": memory_id},
                context_id=memory.context_id,
                user_id=memory.user_id,
                # Carry the fact's resolved absolute date (write-time temporal
                # normalization). Absent/None leaves event_time unset.
                event_time=fact.get("event_time"),
            )

            # One fact must not take the batch down with it. Decomposition
            # produces dozens of facts, and several of them routinely converge
            # on the same merge target as facts from OTHER decompositions
            # running concurrently -- so a lost CAS race (ETag does not match
            # current revision) is expected under load, not exceptional.
            #
            # Letting it propagate abandoned every remaining fact AND skipped
            # the archive below, leaving the parent ACTIVE and factless, which
            # doc_verify then re-drove into the same collision. Failing one fact
            # loses one fact; failing the task lost the document's progress and
            # looped.
            try:
                result = await memory_service.ingest_fact(
                    workspace_id=workspace_id,
                    input=fact_input,
                    source_memory_id=memory_id,
                    inline=False,  # Sub-tasks still go to background
                )
            except Exception as e:  # noqa: BLE001 - per-fact isolation, see above
                failed_facts += 1
                logger.warning(
                    "Failed to ingest fact %d/%d of memory %s (continuing): %s",
                    len(created_fact_ids) + failed_facts, len(facts), memory_id, e,
                )
                continue
            if result:
                created_fact_ids.append(result.id)

        # 4. Create PART_OF associations from each fact to the parent.
        #
        # Deduplicate the RESULT ids, not the facts: ``ingest_fact`` merges a
        # fact into an existing memory when one is similar enough, so two
        # distinct facts from this same parent routinely come back as the SAME
        # id. Writing the association per fact then inserts an identical
        # (source_id, target_id, 'part_of') row twice and trips uq_association.
        # Merging working as designed is what produces the collision, so the
        # answer is to collapse ids here rather than to dedupe content harder.
        # ``dict.fromkeys`` keeps first-seen order, so writes stay deterministic.
        unique_fact_ids = [fid for fid in dict.fromkeys(created_fact_ids) if fid != memory_id]
        if len(unique_fact_ids) != len(created_fact_ids):
            logger.debug(
                "Memory %s: %d fact(s) merged into an existing memory or the "
                "parent itself; associating %d unique fact(s)",
                memory_id, len(created_fact_ids) - len(unique_fact_ids), len(unique_fact_ids),
            )
        for fact_id in unique_fact_ids:
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
                    fact_id,
                    memory_id,
                    e,
                )

        # 5. Archive the parent memory
        try:
            await storage.update_memory(
                workspace_id,
                memory_id,
                status=MemoryStatus.ARCHIVED.value,
            )
            logger.info(
                "Decomposed memory %s into %d atomic facts and archived parent%s",
                memory_id,
                len(unique_fact_ids),
                " (%d fact(s) failed)" % failed_facts if failed_facts else "",
            )
        except Exception as e:
            logger.warning("Failed to archive parent memory %s: %s", memory_id, e)

        # 6. Emit decompose_complete (feed A) so the kb-coalesce join refreshes
        #    the knowledgebase after async decomposition settles.  Best-effort;
        #    no-ops cleanly in OSS-standalone (no Aether send_event).  Emitted
        #    even if the archive above failed (intentional): the facts already
        #    exist, so the KB should still refresh; the coalesce join + the
        #    early-return ACTIVE guard on retry make any duplicate emit harmless.
        await emit_event(
            v,
            workspace_id,
            "memorylayer.decompose_complete",
            {
                "job_id": job_id,
                "memory_id": memory_id,
                "fact_count": len(created_fact_ids),
            },
            logger,
        )
