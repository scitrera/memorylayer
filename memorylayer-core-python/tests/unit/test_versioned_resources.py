"""Backend conformance for the internal typed-resource revision primitive."""

from __future__ import annotations

import asyncio

import pytest

from memorylayer_server.models.versioned_resource import (
    VersionedResourceConflictError,
    VersionedResourcePreconditionFailedError,
)
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.services.versioned_resources.base import VersionedResourceService


@pytest.fixture(params=["memory", "sqlite", "turso"])
async def backend(request, tmp_path):
    if request.param == "memory":
        instance = MemoryStorageBackend()
    elif request.param == "sqlite":
        instance = SQLiteStorageBackend(str(tmp_path / "versioned-resources.db"))
    else:
        pytest.importorskip("turso", reason="pyturso not installed")
        from memorylayer_server.services.storage.turso import TursoStorageBackend

        instance = TursoStorageBackend(mode="local", db_path=str(tmp_path / "versioned-resources.turso"))
    await instance.connect()
    yield instance
    await instance.disconnect()


def _content(title: str) -> dict:
    return {"title": title, "content": f"body: {title}", "enabled": True}


async def _create(service: VersionedResourceService, *, key: str, operation_id: str, workspace_id: str = "ws_a"):
    return await service.create(
        tenant_id="tenant_a",
        workspace_id=workspace_id,
        namespace="agent.prompt-note.v1",
        resource_key=key,
        schema_version=1,
        content=_content(key),
        metadata={"source": "test"},
        actor="user:alice",
        operation_id=operation_id,
        expected_etag="*",
    )


@pytest.mark.asyncio
async def test_cas_history_tombstones_and_idempotent_replay(backend):
    service = VersionedResourceService(backend)
    created = await _create(service, key="coding/style", operation_id="create-1")
    assert created.resource.revision == 1
    assert created.resource.etag.startswith('"vr-1-')
    assert created.replayed is False

    replay = await _create(service, key="coding/style", operation_id="create-1")
    assert replay.replayed is True
    assert replay.resource.id == created.resource.id
    assert replay.resource.etag == created.resource.etag

    with pytest.raises(VersionedResourceConflictError, match="idempotency key"):
        await service.create(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_key="different-key",
            schema_version=1,
            content=_content("different"),
            metadata={},
            actor="user:alice",
            operation_id="create-1",
            expected_etag="*",
        )

    replaced = await service.replace(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        schema_version=2,
        content=_content("replacement"),
        metadata={"source": "review"},
        actor="user:bob",
        operation_id="replace-1",
        expected_etag=created.resource.etag,
    )
    assert replaced.resource.revision == 2
    assert replaced.resource.content["title"] == "replacement"

    with pytest.raises(VersionedResourcePreconditionFailedError):
        await service.replace(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            schema_version=3,
            content=_content("stale"),
            metadata={},
            actor="user:carol",
            operation_id="replace-stale",
            expected_etag=created.resource.etag,
        )

    deleted = await service.delete(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="delete-1",
        expected_etag=replaced.resource.etag,
    )
    assert deleted.resource.revision == 3
    assert deleted.resource.is_deleted
    assert await service.get("tenant_a", "ws_a", "agent.prompt-note.v1", created.resource.id) is None

    delete_replay = await service.delete(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="delete-1",
        expected_etag=replaced.resource.etag,
    )
    assert delete_replay.replayed is True
    assert delete_replay.resource.etag == deleted.resource.etag

    replace_replay = await service.replace(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        schema_version=2,
        content=_content("replacement"),
        metadata={"source": "review"},
        actor="user:bob",
        operation_id="replace-1",
        expected_etag=created.resource.etag,
    )
    assert replace_replay.replayed is True
    assert replace_replay.resource.revision == 2
    assert replace_replay.resource.deleted_at is None

    create_replay_after_delete = await _create(
        service, key="coding/style", operation_id="create-1"
    )
    assert create_replay_after_delete.replayed is True
    assert create_replay_after_delete.resource.revision == 1

    with pytest.raises(VersionedResourceConflictError, match="already exists"):
        await _create(service, key="coding/style", operation_id="create-after-delete")

    with pytest.raises(VersionedResourcePreconditionFailedError):
        await service.restore(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            actor="user:bob",
            operation_id="restore-stale",
            expected_etag=replaced.resource.etag,
        )

    restored = await service.restore(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="restore-1",
        expected_etag=deleted.resource.etag,
    )
    assert restored.resource.revision == 4
    assert restored.resource.id == created.resource.id
    assert restored.resource.resource_key == created.resource.resource_key
    assert restored.resource.content == replaced.resource.content
    assert restored.resource.metadata == replaced.resource.metadata
    assert not restored.resource.is_deleted

    restore_replay = await service.restore(
        tenant_id="tenant_a",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_id=created.resource.id,
        actor="user:bob",
        operation_id="restore-1",
        expected_etag=deleted.resource.etag,
    )
    assert restore_replay.replayed is True
    assert restore_replay.resource.etag == restored.resource.etag

    with pytest.raises(VersionedResourceConflictError, match="not deleted"):
        await service.restore(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            actor="user:bob",
            operation_id="restore-active",
            expected_etag=restored.resource.etag,
        )

    revisions, token = await service.list_revision_page(
        "tenant_a", "ws_a", "agent.prompt-note.v1", created.resource.id, limit=10
    )
    assert token is None
    assert [item.action for item in revisions] == ["restore", "delete", "replace", "create"]
    assert [item.revision for item in revisions] == [4, 3, 2, 1]


@pytest.mark.asyncio
async def test_workspace_isolation_and_opaque_cursor(backend):
    service = VersionedResourceService(backend)
    for index in range(3):
        await _create(service, key=f"note-{index}", operation_id=f"create-{index}")
    other = await _create(service, key="note-other", operation_id="create-other", workspace_id="ws_b")

    first, token = await service.list_page(
        "tenant_a", "ws_a", "agent.prompt-note.v1", limit=2
    )
    assert len(first) == 2
    assert token is not None
    second, final_token = await service.list_page(
        "tenant_a", "ws_a", "agent.prompt-note.v1", limit=2, page_token=token
    )
    assert len(second) == 1
    assert final_token is None
    assert {item.resource_key for item in first + second} == {"note-0", "note-1", "note-2"}

    other_page, _ = await service.list_page(
        "tenant_a", "ws_b", "agent.prompt-note.v1", limit=10
    )
    assert [item.id for item in other_page] == [other.resource.id]

    tenant_other = await service.create(
        tenant_id="tenant_b",
        workspace_id="ws_a",
        namespace="agent.prompt-note.v1",
        resource_key="note-0",
        schema_version=1,
        content=_content("tenant-b"),
        metadata={},
        actor="user:bob",
        operation_id="tenant-b-create",
        expected_etag="*",
    )
    tenant_page, _ = await service.list_page(
        "tenant_b", "ws_a", "agent.prompt-note.v1", limit=10
    )
    assert [item.id for item in tenant_page] == [tenant_other.resource.id]

    with pytest.raises(ValueError, match="invalid page token"):
        await service.list_page(
            "tenant_a", "ws_a", "another.namespace", limit=2, page_token=token
        )
    with pytest.raises(ValueError, match="invalid page token"):
        await service.list_page(
            "tenant_a", "ws_b", "agent.prompt-note.v1", limit=2, page_token=token
        )


@pytest.mark.asyncio
async def test_filtered_page_makes_bounded_progress_across_sparse_matches(backend):
    service = VersionedResourceService(backend)
    for index in range(6):
        await _create(service, key=f"sparse-{index}", operation_id=f"sparse-create-{index}")

    first, token, scanned, truncated = await service.list_filtered_page(
        "tenant_a",
        "ws_a",
        "agent.prompt-note.v1",
        limit=2,
        predicate=lambda resource: resource.resource_key in {"sparse-0", "sparse-1"},
        filter_scope="oldest-two",
        scan_limit=3,
    )
    assert first == []
    assert token is not None
    assert scanned == 3
    assert truncated is True

    second, final_token, scanned, truncated = await service.list_filtered_page(
        "tenant_a",
        "ws_a",
        "agent.prompt-note.v1",
        limit=2,
        predicate=lambda resource: resource.resource_key in {"sparse-0", "sparse-1"},
        filter_scope="oldest-two",
        page_token=token,
        scan_limit=3,
    )
    assert [resource.resource_key for resource in second] == ["sparse-1", "sparse-0"]
    assert final_token is None
    assert scanned == 3
    assert truncated is False

    with pytest.raises(ValueError, match="invalid page token"):
        await service.list_filtered_page(
            "tenant_a",
            "ws_a",
            "agent.prompt-note.v1",
            limit=2,
            predicate=lambda resource: True,
            filter_scope="different-filter",
            page_token=token,
            scan_limit=3,
        )


@pytest.mark.asyncio
async def test_competing_writers_admit_exactly_one_revision(backend):
    service = VersionedResourceService(backend)
    created = await _create(service, key="race", operation_id="create-race")

    async def replace(label: str):
        return await service.replace(
            tenant_id="tenant_a",
            workspace_id="ws_a",
            namespace="agent.prompt-note.v1",
            resource_id=created.resource.id,
            schema_version=1,
            content=_content(label),
            metadata={},
            actor=f"user:{label}",
            operation_id=f"replace-{label}",
            expected_etag=created.resource.etag,
        )

    outcomes = await asyncio.gather(replace("one"), replace("two"), return_exceptions=True)
    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    assert sum(isinstance(item, VersionedResourcePreconditionFailedError) for item in outcomes) == 1
    revisions, _ = await service.list_revision_page(
        "tenant_a", "ws_a", "agent.prompt-note.v1", created.resource.id, limit=10
    )
    assert len(revisions) == 2


@pytest.mark.asyncio
async def test_sqlite_revisions_and_operations_survive_restart(tmp_path):
    path = str(tmp_path / "restart.db")
    first_backend = SQLiteStorageBackend(path)
    await first_backend.connect()
    first_service = VersionedResourceService(first_backend)
    created = await _create(first_service, key="restart", operation_id="create-restart")
    await first_backend.disconnect()

    second_backend = SQLiteStorageBackend(path)
    await second_backend.connect()
    try:
        second_service = VersionedResourceService(second_backend)
        stored = await second_service.get(
            "tenant_a", "ws_a", "agent.prompt-note.v1", created.resource.id
        )
        assert stored is not None
        assert stored.etag == created.resource.etag
        replay = await _create(second_service, key="restart", operation_id="create-restart")
        assert replay.replayed is True
        assert replay.resource.id == created.resource.id
    finally:
        await second_backend.disconnect()
