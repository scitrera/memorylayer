"""Workspace/view authority behavior over the backend-neutral revision store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.workspace_execution import (
    WorkspaceVCSObservation,
    WorkspaceViewCreateInput,
    WorkspaceViewObservationInput,
    WorkspaceViewObservationReplaceInput,
)
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend
from memorylayer_server.services.versioned_resources.base import VersionedResourceService
from memorylayer_server.services.workspace_execution import WorkspaceExecutionService


@pytest.fixture
async def service():
    backend = MemoryStorageBackend()
    await backend.connect()
    yield WorkspaceExecutionService(VersionedResourceService(backend))
    await backend.disconnect()


async def _create_view(service: WorkspaceExecutionService, view_id: str, operation_id: str):
    return await service.create_view(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        request=WorkspaceViewCreateInput(
            view_id=view_id,
            kind="git_worktree",
            display_name=view_id,
            capabilities=["workspace.write", "workspace.read", "workspace.write"],
        ),
        actor="user:alice",
        operation_id=operation_id,
        expected_etag="*",
    )


@pytest.mark.asyncio
async def test_multiple_views_share_workspace_without_collapsing_state(service: WorkspaceExecutionService) -> None:
    first, _ = await _create_view(service, "worktree-a", "view-create-a")
    second, _ = await _create_view(service, "worktree-b", "view-create-b")

    assert first.id != second.id
    assert first.workspace_id == second.workspace_id == "workspace-a"
    assert first.capabilities == ["workspace.read", "workspace.write"]

    views, token = await service.list_views(
        "tenant-a",
        "workspace-a",
        limit=10,
        page_token=None,
        include_deleted=False,
    )
    assert token is None
    assert {view.view_id for view in views} == {"worktree-a", "worktree-b"}


@pytest.mark.asyncio
async def test_observations_are_per_view_and_per_observer_with_history(service: WorkspaceExecutionService) -> None:
    await _create_view(service, "worktree-a", "view-create-a")
    await _create_view(service, "worktree-b", "view-create-b")

    created, replayed = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=WorkspaceViewObservationInput(
            observer_id="sahara-tui-1",
            generation="generation-a",
            sequence=1,
            tool_host_id="host-tui-1",
            execution_site="client",
            root_ref="local-root-1",
            capabilities=["workspace.write", "workspace.read"],
            vcs=WorkspaceVCSObservation(
                repository_id="git:repo-fingerprint",
                worktree_id="git:worktree-a",
                head_revision="0123456789abcdef",
                branch="feature/a",
                dirty=True,
            ),
            observed_at=datetime.now(UTC),
        ),
        observer_id=None,
        actor="client:sahara-tui-1",
        operation_id="observe-create-a",
        expected_etag="*",
        create=True,
    )
    assert replayed is False
    assert created.view_id == "worktree-a"
    assert created.vcs and created.vcs.branch == "feature/a"

    replaced, replayed = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=WorkspaceViewObservationReplaceInput(
            generation="generation-a",
            sequence=2,
            tool_host_id="host-tui-1",
            execution_site="client",
            root_ref="local-root-1",
            vcs=WorkspaceVCSObservation(
                repository_id="git:repo-fingerprint",
                worktree_id="git:worktree-a",
                head_revision="fedcba9876543210",
                branch="feature/a",
                dirty=False,
            ),
        ),
        observer_id="sahara-tui-1",
        actor="client:sahara-tui-1",
        operation_id="observe-replace-a",
        expected_etag=created.etag,
        create=False,
    )
    assert replayed is False
    assert replaced.sequence == 2
    assert replaced.resource_revision == 2

    revisions, token = await service.list_observation_revisions(
        "tenant-a",
        "workspace-a",
        "worktree-a",
        "sahara-tui-1",
        limit=10,
        page_token=None,
    )
    assert token is None
    assert [observation.sequence for observation, _action, _operation in revisions] == [2, 1]

    other_view, token = await service.list_observations(
        "tenant-a",
        "workspace-a",
        "worktree-b",
        limit=10,
        page_token=None,
    )
    assert token is None
    assert other_view == []


@pytest.mark.asyncio
async def test_observer_cursor_is_monotonic_but_retries_are_idempotent(service: WorkspaceExecutionService) -> None:
    await _create_view(service, "worktree-a", "view-create-a")
    created, _ = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=WorkspaceViewObservationInput(
            observer_id="observer-a",
            generation="generation-a",
            sequence=4,
        ),
        observer_id=None,
        actor=None,
        operation_id="observe-create",
        expected_etag="*",
        create=True,
    )
    update = WorkspaceViewObservationReplaceInput(generation="generation-a", sequence=5)
    replaced, _ = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=update,
        observer_id="observer-a",
        actor=None,
        operation_id="observe-update",
        expected_etag=created.etag,
        create=False,
    )

    replay, replayed = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=update,
        observer_id="observer-a",
        actor=None,
        operation_id="observe-update",
        expected_etag=created.etag,
        create=False,
    )
    assert replayed is True
    assert replay.etag == replaced.etag

    with pytest.raises(ValueError, match="must advance"):
        await service.put_observation(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            view_id="worktree-a",
            request=WorkspaceViewObservationReplaceInput(generation="generation-a", sequence=5),
            observer_id="observer-a",
            actor=None,
            operation_id="observe-stale",
            expected_etag=replaced.etag,
            create=False,
        )

    restarted, _ = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=WorkspaceViewObservationReplaceInput(generation="generation-b", sequence=0),
        observer_id="observer-a",
        actor=None,
        operation_id="observe-new-generation",
        expected_etag=replaced.etag,
        create=False,
    )
    assert restarted.generation == "generation-b"
    assert restarted.sequence == 0


@pytest.mark.asyncio
async def test_views_and_observations_tombstone_and_restore(service: WorkspaceExecutionService) -> None:
    view, _ = await _create_view(service, "worktree-a", "view-create-a")
    observation, _ = await service.put_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        request=WorkspaceViewObservationInput(observer_id="observer-a", generation="generation-a", sequence=1),
        observer_id=None,
        actor="user:alice",
        operation_id="observation-create",
        expected_etag="*",
        create=True,
    )
    deleted_observation, _ = await service.delete_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        observer_id="observer-a",
        actor="user:alice",
        operation_id="observation-delete",
        expected_etag=observation.etag,
    )
    assert deleted_observation.deleted_at is not None
    assert await service.get_observation("tenant-a", "workspace-a", "worktree-a", "observer-a") is None
    restored_observation, _ = await service.restore_observation(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        observer_id="observer-a",
        actor="user:alice",
        operation_id="observation-restore",
        expected_etag=deleted_observation.etag,
    )
    assert restored_observation.deleted_at is None

    deleted_view, _ = await service.delete_view(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        actor="user:alice",
        operation_id="view-delete",
        expected_etag=view.etag,
    )
    assert deleted_view.deleted_at is not None
    assert await service.get_view("tenant-a", "workspace-a", "worktree-a") is None
    tombstone = await service.get_view("tenant-a", "workspace-a", "worktree-a", include_deleted=True)
    assert tombstone is not None and tombstone.deleted_at is not None
    restored_view, _ = await service.restore_view(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        view_id="worktree-a",
        actor="user:alice",
        operation_id="view-restore",
        expected_etag=deleted_view.etag,
    )
    assert restored_view.deleted_at is None
