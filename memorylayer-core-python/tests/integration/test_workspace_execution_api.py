"""HTTP contract for durable workspace views and observer state."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from memorylayer_server.api.v1.deps import (
    get_auth_service,
    get_authz_service,
    get_versioned_resource_service,
    get_workspace_service,
)
from memorylayer_server.api.v1.workspace_execution import router
from memorylayer_server.models.auth import RequestContext
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend
from memorylayer_server.services.versioned_resources.base import VersionedResourceService
from memorylayer_server.services.workspace.default import WorkspaceService


class _Authentication:
    async def build_context(self, _request, _body):
        return RequestContext(tenant_id="_default", workspace_id="_default", user_id="test-user")


class _Authorization:
    async def require_authorization(self, *_args, **_kwargs) -> None:
        return None


@pytest.fixture
async def workspace_execution_client():
    backend = MemoryStorageBackend()
    await backend.connect()
    workspace_service = WorkspaceService(backend)
    app = FastAPI()
    app.include_router(router)
    app.state.workspace_service = workspace_service

    async def auth_service():
        return _Authentication()

    async def authz_service():
        return _Authorization()

    async def workspace_dependency():
        return workspace_service

    async def resources_dependency():
        return VersionedResourceService(backend)

    app.dependency_overrides[get_auth_service] = auth_service
    app.dependency_overrides[get_authz_service] = authz_service
    app.dependency_overrides[get_workspace_service] = workspace_dependency
    app.dependency_overrides[get_versioned_resource_service] = resources_dependency
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    yield client, workspace_service
    await client.aclose()
    await backend.disconnect()


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    return {}


async def _create_workspace(workspace_service: WorkspaceService, headers: dict[str, str], workspace_id: str) -> None:
    del headers
    await workspace_service.create_workspace(Workspace(id=workspace_id, tenant_id="_default", name=workspace_id))


def _mutation_headers(headers: dict[str, str], operation_id: str, conditional: tuple[str, str]) -> dict[str, str]:
    return {**headers, "Idempotency-Key": operation_id, conditional[0]: conditional[1]}


@pytest.mark.asyncio
async def test_workspace_views_and_observations_round_trip(
    workspace_execution_client: tuple[AsyncClient, WorkspaceService],
    workspace_headers: dict[str, str],
) -> None:
    test_client, workspace_service = workspace_execution_client
    workspace_id = "ws_view_api"
    await _create_workspace(workspace_service, workspace_headers, workspace_id)

    create_headers = _mutation_headers(workspace_headers, "create-view-a", ("If-None-Match", "*"))
    response = await test_client.post(
        f"/v1/workspaces/{workspace_id}/views",
        json={
            "view_id": "worktree-a",
            "kind": "git_worktree",
            "display_name": "Feature worktree",
            "memory_context_id": "ctx-worktree-a",
            "capabilities": ["workspace.write", "workspace.read", "workspace.write"],
        },
        headers=create_headers,
    )
    assert response.status_code == 201, response.text
    view = response.json()["view"]
    assert view["view_id"] == "worktree-a"
    assert view["capabilities"] == ["workspace.read", "workspace.write"]
    first_etag = response.headers["etag"]

    replay = await test_client.post(
        f"/v1/workspaces/{workspace_id}/views",
        json={
            "view_id": "worktree-a",
            "kind": "git_worktree",
            "display_name": "Feature worktree",
            "memory_context_id": "ctx-worktree-a",
            "capabilities": ["workspace.write", "workspace.read", "workspace.write"],
        },
        headers=create_headers,
    )
    assert replay.status_code == 201, replay.text
    assert replay.json()["replayed"] is True
    assert replay.headers["etag"] == first_etag

    missing_precondition = await test_client.put(
        f"/v1/workspaces/{workspace_id}/views/worktree-a",
        json={"kind": "git_worktree"},
        headers=workspace_headers,
    )
    assert missing_precondition.status_code == 428

    stale = await test_client.put(
        f"/v1/workspaces/{workspace_id}/views/worktree-a",
        json={"kind": "git_worktree"},
        headers=_mutation_headers(workspace_headers, "replace-view-stale", ("If-Match", '"stale"')),
    )
    assert stale.status_code == 412, stale.text

    replaced = await test_client.put(
        f"/v1/workspaces/{workspace_id}/views/worktree-a",
        json={"kind": "git_worktree", "display_name": "Renamed worktree"},
        headers=_mutation_headers(workspace_headers, "replace-view-a", ("If-Match", first_etag)),
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["view"]["revision"] == 2
    replaced_view_etag = replaced.headers["etag"]

    observation_url = f"/v1/workspaces/{workspace_id}/views/worktree-a/observations/sahara-tui-1"
    observed = await test_client.put(
        observation_url,
        json={
            "generation": "generation-a",
            "sequence": 1,
            "tool_host_id": "host-tui-1",
            "execution_site": "client",
            "root_ref": "local-root-1",
            "capabilities": ["workspace.write", "workspace.read"],
            "vcs": {
                "kind": "git",
                "repository_id": "git:repo-fingerprint",
                "worktree_id": "git:worktree-a",
                "head_revision": "0123456789abcdef",
                "branch": "feature/a",
                "dirty": True,
            },
        },
        headers=_mutation_headers(workspace_headers, "observe-view-a-1", ("If-None-Match", "*")),
    )
    assert observed.status_code == 201, observed.text
    observation_etag = observed.headers["etag"]
    assert observed.json()["observation"]["resource_revision"] == 1

    updated = await test_client.put(
        observation_url,
        json={
            "generation": "generation-a",
            "sequence": 2,
            "tool_host_id": "host-tui-1",
            "execution_site": "client",
            "root_ref": "local-root-1",
            "vcs": {"kind": "git", "head_revision": "fedcba9876543210", "branch": "feature/a", "dirty": False},
        },
        headers=_mutation_headers(workspace_headers, "observe-view-a-2", ("If-Match", observation_etag)),
    )
    assert updated.status_code == 200, updated.text
    updated_etag = updated.headers["etag"]
    assert updated.json()["observation"]["sequence"] == 2

    stale_cursor = await test_client.put(
        observation_url,
        json={"generation": "generation-a", "sequence": 2},
        headers=_mutation_headers(workspace_headers, "observe-view-a-stale", ("If-Match", updated_etag)),
    )
    assert stale_cursor.status_code == 400, stale_cursor.text
    assert "must advance" in stale_cursor.json()["detail"]

    observations = await test_client.get(
        f"/v1/workspaces/{workspace_id}/views/worktree-a/observations",
        headers=workspace_headers,
    )
    assert observations.status_code == 200, observations.text
    assert [item["observer_id"] for item in observations.json()["observations"]] == ["sahara-tui-1"]

    revisions = await test_client.get(
        f"{observation_url}/revisions",
        headers=workspace_headers,
    )
    assert revisions.status_code == 200, revisions.text
    assert [item["observation"]["sequence"] for item in revisions.json()["revisions"]] == [2, 1]

    deleted_observation = await test_client.delete(
        observation_url,
        headers=_mutation_headers(workspace_headers, "observe-view-a-delete", ("If-Match", updated_etag)),
    )
    assert deleted_observation.status_code == 200, deleted_observation.text
    assert deleted_observation.json()["observation"]["deleted_at"] is not None
    missing_observation = await test_client.get(observation_url, headers=workspace_headers)
    assert missing_observation.status_code == 404
    restored_observation = await test_client.post(
        f"{observation_url}/restore",
        headers=_mutation_headers(
            workspace_headers,
            "observe-view-a-restore",
            ("If-Match", deleted_observation.headers["etag"]),
        ),
    )
    assert restored_observation.status_code == 200, restored_observation.text
    assert restored_observation.json()["observation"]["deleted_at"] is None

    view_url = f"/v1/workspaces/{workspace_id}/views/worktree-a"
    deleted_view = await test_client.delete(
        view_url,
        headers=_mutation_headers(workspace_headers, "delete-view-a", ("If-Match", replaced_view_etag)),
    )
    assert deleted_view.status_code == 200, deleted_view.text
    assert deleted_view.json()["view"]["deleted_at"] is not None
    assert (await test_client.get(view_url, headers=workspace_headers)).status_code == 404
    tombstone = await test_client.get(view_url, params={"include_deleted": "true"}, headers=workspace_headers)
    assert tombstone.status_code == 200, tombstone.text
    restored_view = await test_client.post(
        f"{view_url}/restore",
        headers=_mutation_headers(
            workspace_headers,
            "restore-view-a",
            ("If-Match", deleted_view.headers["etag"]),
        ),
    )
    assert restored_view.status_code == 200, restored_view.text
    assert restored_view.json()["view"]["deleted_at"] is None


@pytest.mark.asyncio
async def test_workspace_view_rejects_host_local_path_fields(
    workspace_execution_client: tuple[AsyncClient, WorkspaceService],
    workspace_headers: dict[str, str],
) -> None:
    test_client, workspace_service = workspace_execution_client
    workspace_id = "ws_view_no_paths"
    await _create_workspace(workspace_service, workspace_headers, workspace_id)

    response = await test_client.post(
        f"/v1/workspaces/{workspace_id}/views",
        json={"view_id": "worktree-a", "kind": "git_worktree", "absolute_path": "/home/alice/project"},
        headers=_mutation_headers(workspace_headers, "create-view-path", ("If-None-Match", "*")),
    )
    assert response.status_code == 422, response.text
