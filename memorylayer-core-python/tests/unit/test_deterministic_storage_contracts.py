"""SQLite contracts inherited by storage backends for deterministic memory."""

import hashlib
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.context_pack import (
    ContextDeltaInput,
    ContextPackInput,
    SessionCheckpointInput,
)
from memorylayer_server.models.entity_relation import (
    EntityRelation,
    EntityRelationEvidence,
    RelationEvidenceKind,
)
from memorylayer_server.models.generation import EnrichmentPolicy
from memorylayer_server.models.memory import MemoryType, RecallInput, RecallMode, RememberInput
from memorylayer_server.models.session import Session
from memorylayer_server.services.context_pack import ContextPackService
from memorylayer_server.services.memory.budget import estimate_tokens


@pytest.mark.asyncio
async def test_deterministic_remember_and_recall_never_attempt_generation(
    memory_service,
    workspace_id,
):
    llm = memory_service.llm_service
    original_policy = llm.policy
    before_attempts = sum(ledger.attempted for ledger in llm._ledgers.values())
    before_unlabeled = llm._unlabeled_calls
    llm.policy = EnrichmentPolicy.DETERMINISTIC
    try:
        await memory_service.remember(
            workspace_id,
            RememberInput(
                content="Deterministic contract memory about the amber launch checklist.",
                type=MemoryType.EPISODIC,
            ),
            inline=True,
        )
        result = await memory_service.recall(
            workspace_id,
            RecallInput(
                query="amber launch checklist",
                mode=RecallMode.RAG,
                include_associations=False,
                include_relations=False,
                traverse_depth=0,
                trace=True,
            ),
        )
    finally:
        llm.policy = original_policy

    after_attempts = sum(ledger.attempted for ledger in llm._ledgers.values())
    assert after_attempts == before_attempts
    assert llm._unlabeled_calls == before_unlabeled
    assert result.generation_summary is not None
    assert result.generation_summary.policy == EnrichmentPolicy.DETERMINISTIC
    assert result.generation_summary.calls == 0


@pytest.mark.asyncio
async def test_versioned_restore_reactivates_metadata_relation_evidence(memory_service, workspace_id):
    created = await memory_service.remember_versioned(
        workspace_id,
        RememberInput(
            content="Lifecycle record assigns the Borealis Cutover to Nia Brooks.",
            type=MemoryType.SEMANTIC,
            logical_key="connector/lifecycle/borealis-cutover",
            metadata={
                "knowledge_work": {
                    "subject": {"name": "Borealis Cutover", "type": "work_item"},
                    "assignee": "Nia Brooks",
                }
            },
        ),
        tenant_id="default_tenant",
        user_id=None,
        operation_id="op-borealis-create",
        expected_etag="*",
    )
    assert created.memory.relation_write_result is not None
    assert created.memory.relation_write_result.resolved == 1

    deleted = await memory_service.delete_versioned(
        workspace_id,
        created.memory.id,
        tenant_id="default_tenant",
        operation_id="op-borealis-delete",
        expected_etag=created.memory.etag,
    )
    restored = await memory_service.restore_versioned(
        workspace_id,
        created.memory.id,
        tenant_id="default_tenant",
        operation_id="op-borealis-restore",
        expected_etag=deleted.memory.etag,
    )
    assert restored.memory.relation_write_result is not None
    assert restored.memory.relation_write_result.resolved == 1

    recalled = await memory_service.recall(
        workspace_id,
        RecallInput(
            query="Who is assigned to Borealis Cutover?",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
            include_associations=False,
            traverse_depth=0,
            include_relations=True,
            include_global=False,
            include_global_user=False,
        ),
    )
    assert created.memory.id in {memory.id for memory in recalled.memories}
    assert recalled.relation_paths


@pytest.mark.asyncio
async def test_checkpoint_is_raw_first_and_idempotent(storage_backend, workspace_id):
    session = Session.create_with_ttl(
        "sess_checkpoint_contract",
        workspace_id,
        tenant_id="default_tenant",
    )
    await storage_backend.create_session(workspace_id, session)
    transcript = "User: preserve this exact transcript.\n"
    request = SessionCheckpointInput(
        transcript_segment=transcript,
        source_kind="test_transcript",
        source_boundary=len(transcript.encode()),
        content_hash=hashlib.sha256(transcript.encode()).hexdigest(),
        idempotency_key="checkpoint-contract-1",
    )

    first, replayed_first = await storage_backend.create_session_checkpoint(
        workspace_id,
        session.id,
        request,
    )
    second, replayed_second = await storage_backend.create_session_checkpoint(
        workspace_id,
        session.id,
        request,
    )
    raw = await storage_backend.get_memory(
        workspace_id,
        first.raw_memory_id,
        track_access=False,
    )

    assert replayed_first is False
    assert replayed_second is True
    assert second.id == first.id
    assert second.raw_memory_id == first.raw_memory_id
    assert raw is not None
    assert raw.content == transcript
    assert raw.session_id == session.id


@pytest.mark.asyncio
async def test_checkpoint_idempotency_key_rejects_different_content(storage_backend, workspace_id):
    session = Session.create_with_ttl(
        "sess_checkpoint_conflict",
        workspace_id,
        tenant_id="default_tenant",
    )
    await storage_backend.create_session(workspace_id, session)

    def request(content: str) -> SessionCheckpointInput:
        return SessionCheckpointInput(
            transcript_segment=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            idempotency_key="checkpoint-conflict",
        )

    await storage_backend.create_session_checkpoint(workspace_id, session.id, request("first"))
    with pytest.raises(ValueError, match="different checkpoint content"):
        await storage_backend.create_session_checkpoint(workspace_id, session.id, request("second"))


@pytest.mark.asyncio
async def test_relation_evidence_is_idempotent_and_final_removal_deactivates_edge(
    storage_backend,
    workspace_id,
):
    memory = await storage_backend.create_memory(
        workspace_id,
        RememberInput(content="Alice works for Acme Corp."),
    )
    now = datetime.now(UTC).isoformat()
    for entity_id, canonical_name, normalized_name, entity_type in (
        ("ent_alice", "Alice", "alice", "person"),
        ("ent_acme", "Acme Corp", "acme corp", "org"),
    ):
        await storage_backend.store_entity(
            {
                "id": entity_id,
                "workspace_id": workspace_id,
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "normalized_name": normalized_name,
                "created_at": now,
                "updated_at": now,
            }
        )

    relation = EntityRelation(
        id="erel_contract",
        workspace_id=workspace_id,
        source_entity_id="ent_alice",
        target_entity_id="ent_acme",
        relationship="employed_by",
        confidence=0.9,
    )
    evidence = EntityRelationEvidence(
        id="eev_contract",
        workspace_id=workspace_id,
        relation_id=relation.id,
        source_memory_id=memory.id,
        evidence_kind=RelationEvidenceKind.EXPLICIT,
        source_span_start=0,
        source_span_end=len(memory.content),
        excerpt_hash=hashlib.sha256(memory.content.encode()).hexdigest(),
        confidence=0.9,
        extraction_method="explicit_api",
    )

    stored, duplicate_first = await storage_backend.upsert_entity_relation(relation, evidence)
    replayed, duplicate_second = await storage_backend.upsert_entity_relation(relation, evidence)
    paths = await storage_backend.traverse_entity_relations(
        workspace_id,
        ["ent_alice"],
        relationships=["employed_by"],
        direction="outgoing",
        max_hops=1,
        max_edges=10,
    )

    assert duplicate_first is False
    assert duplicate_second is True
    assert replayed.id == stored.id
    assert paths[0].evidence_memory_ids == [memory.id]

    assert (
        await storage_backend.deactivate_relation_evidence_for_memory(
            workspace_id,
            memory.id,
        )
        == 1
    )
    assert (
        await storage_backend.traverse_entity_relations(
            workspace_id,
            ["ent_alice"],
            relationships=["employed_by"],
            direction="outgoing",
            max_hops=1,
            max_edges=10,
        )
        == []
    )


@pytest.mark.asyncio
async def test_metadata_first_relations_create_typed_entities_and_drive_recall(
    memory_service,
    workspace_id,
):
    memory = await memory_service.remember(
        workspace_id,
        RememberInput(
            content="The accountable delivery record is current and approved.",
            type=MemoryType.SEMANTIC,
            metadata={
                "knowledge_work": {
                    "subject": {"name": "Atlas Cutover", "type": "work_item"},
                    "owner": {"name": "Alice Chen", "type": "person"},
                    "project": {"name": "Atlas Program", "type": "project"},
                }
            },
        ),
        inline=True,
    )

    assert memory.relation_write_result is not None
    assert memory.relation_write_result.resolved == 2
    assert memory.relation_write_result.unresolved == 0
    recall = await memory_service.recall(
        workspace_id,
        RecallInput(
            query="Who owns Atlas Cutover?",
            mode=RecallMode.RAG,
            limit=5,
            min_relevance=0.0,
            include_associations=False,
            traverse_depth=0,
            include_relations=True,
            include_global=False,
            include_global_user=False,
        ),
    )
    assert memory.id in {item.id for item in recall.memories}
    assert recall.relation_paths
    assert recall.relation_paths[0].relations[0].relationship == "owns"


@pytest.mark.asyncio
async def test_context_pack_cursor_replay_and_delete_tombstone(
    storage_backend,
    workspace_id,
):
    session = Session.create_with_ttl(
        "sess_context_contract",
        workspace_id,
        tenant_id="default_tenant",
    )
    await storage_backend.create_session(workspace_id, session)
    await storage_backend.set_working_memory(
        workspace_id,
        session.id,
        "open_threads",
        ["finish deterministic storage"],
    )
    memory = await storage_backend.create_memory(
        workspace_id,
        RememberInput(
            content="A recent session event.",
            session_id=session.id,
        ),
    )
    service = ContextPackService(
        storage_backend,
        memory_service=object(),
        cursor_secret="test-context-secret",
    )
    pack = await service.build_pack(
        workspace_id,
        session.id,
        ContextPackInput(
            budget_tokens=64,
            include_directives=False,
            include_contradictions=False,
            include_sandbox_summary=False,
            include_checkpoint_recovery=False,
        ),
    )
    assert estimate_tokens(pack.rendered) <= 64
    assert pack.open_threads

    await storage_backend.delete_memory(workspace_id, memory.id)
    delta_input = ContextDeltaInput(cursor=pack.cursor, budget_tokens=64)
    first = await service.build_delta(workspace_id, session.id, delta_input)
    replay = await service.build_delta(workspace_id, session.id, delta_input)

    assert first == replay
    assert estimate_tokens(first.rendered) <= 64
    assert any(item.id == memory.id and item.tombstone for item in first.items)
