"""Knowledge-phase plumbing: the scheduling signal and its persistence.

``enqueue_post_store`` owns the decision of whether a memory gets decomposed
(enabled flag, content shape, memory type). Document ingestion needs that answer
to track its knowledge phase, and must take it from here rather than re-deriving
the heuristic and drifting from it.
"""

import pytest

from memorylayer_server.models.document import (
    Document,
    DocumentEnrichmentStatus,
    DocumentType,
)


def _doc(**overrides) -> Document:
    kwargs = dict(
        id="doc_enrich_1",
        workspace_id="default",
        filename="a.pdf",
        document_type=DocumentType.PDF,
        content_hash="c" * 64,
        size_bytes=10,
    )
    kwargs.update(overrides)
    return Document(**kwargs)


class TestEnrichmentStatusDefaults:
    def test_a_new_document_has_not_run_enrichment(self):
        doc = _doc()
        assert doc.enrichment_status == DocumentEnrichmentStatus.NOT_APPLICABLE
        assert doc.enrichment_memory_ids == []

    def test_enrichment_status_is_independent_of_ingestion_status(self):
        # The whole point of the split: a retrievable document may still have
        # fact extraction outstanding.
        from memorylayer_server.models.document import DocumentStatus

        doc = _doc(
            status=DocumentStatus.COMPLETED,
            enrichment_status=DocumentEnrichmentStatus.PENDING,
            enrichment_memory_ids=["mem_a"],
        )
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.enrichment_status == DocumentEnrichmentStatus.PENDING


class TestEnrichmentStatusPersistence:
    @pytest.mark.asyncio
    async def test_round_trips_through_storage(self, storage_backend, workspace_id):
        doc = _doc(
            workspace_id=workspace_id,
            enrichment_status=DocumentEnrichmentStatus.PENDING,
            enrichment_memory_ids=["mem_a", "mem_b"],
        )
        await storage_backend.create_document(workspace_id, doc)

        loaded = await storage_backend.get_document(workspace_id, doc.id)
        assert loaded.enrichment_status == DocumentEnrichmentStatus.PENDING
        assert loaded.enrichment_memory_ids == ["mem_a", "mem_b"]

    @pytest.mark.asyncio
    async def test_update_advances_the_phase(self, storage_backend, workspace_id):
        doc = _doc(
            id="doc_enrich_2",
            workspace_id=workspace_id,
            # Distinct hash: (workspace_id, content_hash) is unique, and the
            # storage fixture is shared across tests in this class.
            content_hash="e" * 64,
            enrichment_status=DocumentEnrichmentStatus.PENDING,
            enrichment_memory_ids=["mem_a"],
        )
        await storage_backend.create_document(workspace_id, doc)

        await storage_backend.update_document(
            workspace_id, doc.id,
            enrichment_status=DocumentEnrichmentStatus.COMPLETE,
        )

        loaded = await storage_backend.get_document(workspace_id, doc.id)
        assert loaded.enrichment_status == DocumentEnrichmentStatus.COMPLETE
        # Advancing the phase must not disturb the recorded scheduled set.
        assert loaded.enrichment_memory_ids == ["mem_a"]


class TestEnqueuePostStoreReportsScheduling:
    @pytest.mark.asyncio
    async def test_returns_false_when_nothing_was_scheduled(
        self, memory_service, workspace_id,
    ):
        from memorylayer_server.models.memory import MemoryType, RememberInput

        # Short single-sentence content fails the decompose heuristic.
        memory = await memory_service.remember(
            workspace_id, RememberInput(content="Short.", type=MemoryType.SEMANTIC),
        )
        scheduled = await memory_service.enqueue_post_store(
            workspace_id, memory, embedding=[0.1, 0.2, 0.3],
        )
        assert scheduled is False
