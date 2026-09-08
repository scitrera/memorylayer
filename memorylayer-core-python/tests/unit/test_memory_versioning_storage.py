"""Conformance tests for native semantic memory revision storage."""

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import (
    MemoryMutation,
    MemoryType,
    RememberInput,
)
from memorylayer_server.models.versioned_resource import (
    VersionedResourceConflictError,
    VersionedResourcePreconditionFailedError,
)
from memorylayer_server.services.memory.versioning import (
    canonical_hash,
    memory_semantic_state,
)
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend


@pytest.fixture(params=("memory", "sqlite", "turso"))
async def backend(request, tmp_path):
    if request.param == "memory":
        value = MemoryStorageBackend()
    elif request.param == "sqlite":
        value = SQLiteStorageBackend(str(tmp_path / "versioned-memory.db"))
    else:
        pytest.importorskip("turso", reason="pyturso not installed")
        from memorylayer_server.services.storage.turso import TursoStorageBackend

        value = TursoStorageBackend(
            mode="local", db_path=str(tmp_path / "versioned-memory.turso")
        )
    await value.connect()
    yield value
    await value.disconnect()


def _input() -> RememberInput:
    return RememberInput(
        tenant_id="tenant-a",
        logical_key="preferences/editor",
        content="Prefer vim.",
        type=MemoryType.SEMANTIC,
        subtype="preference",
        tags=["editor"],
        metadata={"source": "instruction"},
        refinement_metadata={"confidence": "explicit"},
        pinned=True,
    )


@pytest.mark.asyncio
async def test_semantic_revision_lifecycle_and_derived_update_independence(backend):
    created = await backend.create_memory("workspace-a", _input())
    assert created.revision == 1
    assert created.etag

    derived = await backend.update_memory(
        "workspace-a",
        created.id,
        importance=0.2,
        decay_factor=0.8,
        embedding=[0.1, 0.2],
    )
    assert derived.revision == 1
    assert derived.etag == created.etag

    desired = derived.model_copy(
        update={
            "content": "Prefer helix.",
            "content_hash": "replacement-hash",
            "refinement_metadata": {"confidence": "revised"},
            "updated_at": datetime.now(UTC),
        }
    )
    request_hash = canonical_hash(
        {
            "action": "replace",
            "state": memory_semantic_state(desired),
            "expected_etag": created.etag,
        }
    )
    mutation = MemoryMutation(
        action="replace",
        memory=desired,
        operation_id="memory-replace-1",
        request_hash=request_hash,
        expected_etag=created.etag,
    )
    replaced = await backend.mutate_memory(mutation)
    replay = await backend.mutate_memory(mutation)
    assert replaced.memory.revision == 2
    assert replay.replayed is True
    assert replay.memory.etag == replaced.memory.etag

    stale = mutation.model_copy(
        update={
            "operation_id": "memory-replace-stale",
            "request_hash": canonical_hash({"stale": True}),
        }
    )
    with pytest.raises(VersionedResourcePreconditionFailedError):
        await backend.mutate_memory(stale)

    assert await backend.delete_memory("workspace-a", created.id) is True
    tombstone = await backend.get_memory(
        "workspace-a",
        created.id,
        track_access=False,
        include_deleted=True,
    )
    assert tombstone.deleted_at is not None
    restored_desired = tombstone.model_copy(
        update={"deleted_at": None, "updated_at": datetime.now(UTC)}
    )
    restored = await backend.mutate_memory(
        MemoryMutation(
            action="restore",
            memory=restored_desired,
            operation_id="memory-restore-1",
            request_hash=canonical_hash(
                {
                    "action": "restore",
                    "id": created.id,
                    "expected_etag": tombstone.etag,
                }
            ),
            expected_etag=tombstone.etag,
        )
    )
    assert restored.memory.deleted_at is None
    assert restored.memory.revision == 4

    history = await backend.list_memory_revisions(
        "tenant-a", "workspace-a", created.id, limit=10
    )
    assert [item.action for item in history] == [
        "restore",
        "delete",
        "replace",
        "create",
    ]
    assert all(item.memory.embedding is None for item in history)


@pytest.mark.asyncio
async def test_logical_key_is_reserved_by_tombstone(backend):
    created = await backend.create_memory("workspace-a", _input())
    assert await backend.delete_memory("workspace-a", created.id) is True

    with pytest.raises(VersionedResourceConflictError):
        await backend.create_memory("workspace-a", _input())
