"""Integration tests for workspace tag lookup via the API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    """Workspace headers for workspace operations."""
    return {"X-Workspace-ID": "test_workspace"}


class TestWorkspaceTags:
    """POST /v1/workspaces with tags + GET /v1/workspaces?tags=...&match=..."""

    def _create(self, client: TestClient, headers, ws_id, tags):
        response = client.post(
            "/v1/workspaces",
            json={"id": ws_id, "name": ws_id, "tags": tags},
            headers=headers,
        )
        assert response.status_code == 201, response.text
        return response.json()["workspace"]

    def test_create_persists_normalized_tags(self, test_client: TestClient, workspace_headers) -> None:
        ws = self._create(test_client, workspace_headers, "ws_api_tags", ["  Knowledge ", "knowledge", "Topic:Finance"])
        assert ws["tags"] == ["knowledge", "topic:finance"]

    def test_list_filters_by_tag(self, test_client: TestClient, workspace_headers) -> None:
        self._create(test_client, workspace_headers, "ws_api_kb_fin", ["knowledge", "topic:finance"])
        self._create(test_client, workspace_headers, "ws_api_kb_legal", ["knowledge", "topic:legal"])
        self._create(test_client, workspace_headers, "ws_api_plain", ["project"])

        def ids(resp):
            return {w["id"] for w in resp.json()["workspaces"]}

        # single tag
        resp = test_client.get("/v1/workspaces", params={"tags": ["knowledge"]}, headers=workspace_headers)
        assert resp.status_code == 200
        got = ids(resp)
        assert {"ws_api_kb_fin", "ws_api_kb_legal"} <= got
        assert "ws_api_plain" not in got

        # match=all
        resp = test_client.get(
            "/v1/workspaces",
            params={"tags": ["knowledge", "topic:finance"], "match": "all"},
            headers=workspace_headers,
        )
        got = ids(resp)
        assert "ws_api_kb_fin" in got
        assert "ws_api_kb_legal" not in got

        # match=any
        resp = test_client.get(
            "/v1/workspaces",
            params={"tags": ["topic:finance", "project"], "match": "any"},
            headers=workspace_headers,
        )
        got = ids(resp)
        assert {"ws_api_kb_fin", "ws_api_plain"} <= got
        assert "ws_api_kb_legal" not in got


class TestWorkspaceContexts:
    """CRUD for /v1/workspaces/{workspace_id}/contexts."""

    def _create_workspace(self, client: TestClient, headers, ws_id: str) -> None:
        resp = client.post("/v1/workspaces", json={"id": ws_id, "name": ws_id}, headers=headers)
        assert resp.status_code == 201, resp.text

    def test_create_list_delete_round_trip(self, test_client: TestClient, workspace_headers) -> None:
        ws_id = "ws_ctx_crud"
        self._create_workspace(test_client, workspace_headers, ws_id)

        # Create
        resp = test_client.post(
            f"/v1/workspaces/{ws_id}/contexts",
            json={"id": "ctx_alpha", "name": "project-alpha", "description": "Alpha"},
            headers=workspace_headers,
        )
        assert resp.status_code == 201, resp.text
        ctx = resp.json()["context"]
        assert ctx["id"] == "ctx_alpha"
        assert ctx["name"] == "project-alpha"
        assert ctx["workspace_id"] == ws_id

        # List (includes the auto-created _default context + our new one)
        resp = test_client.get(f"/v1/workspaces/{ws_id}/contexts", headers=workspace_headers)
        assert resp.status_code == 200, resp.text
        ids = {c["id"] for c in resp.json()["contexts"]}
        assert "ctx_alpha" in ids

        # Delete
        resp = test_client.delete(f"/v1/workspaces/{ws_id}/contexts/ctx_alpha", headers=workspace_headers)
        assert resp.status_code == 204, resp.text

        # No longer listed
        resp = test_client.get(f"/v1/workspaces/{ws_id}/contexts", headers=workspace_headers)
        assert resp.status_code == 200
        ids = {c["id"] for c in resp.json()["contexts"]}
        assert "ctx_alpha" not in ids

    def test_delete_missing_returns_404(self, test_client: TestClient, workspace_headers) -> None:
        ws_id = "ws_ctx_missing"
        self._create_workspace(test_client, workspace_headers, ws_id)
        resp = test_client.delete(f"/v1/workspaces/{ws_id}/contexts/ctx_nope", headers=workspace_headers)
        assert resp.status_code == 404, resp.text

    def test_second_delete_returns_404(self, test_client: TestClient, workspace_headers) -> None:
        ws_id = "ws_ctx_double"
        self._create_workspace(test_client, workspace_headers, ws_id)

        resp = test_client.post(
            f"/v1/workspaces/{ws_id}/contexts",
            json={"id": "ctx_once", "name": "once"},
            headers=workspace_headers,
        )
        assert resp.status_code == 201, resp.text

        resp = test_client.delete(f"/v1/workspaces/{ws_id}/contexts/ctx_once", headers=workspace_headers)
        assert resp.status_code == 204, resp.text

        # Second delete -> 404
        resp = test_client.delete(f"/v1/workspaces/{ws_id}/contexts/ctx_once", headers=workspace_headers)
        assert resp.status_code == 404, resp.text


class TestWorkspaceDeletion:
    """DELETE /v1/workspaces/{workspace_id} reports persisted state truthfully."""

    def test_delete_round_trip_and_second_delete(self, test_client: TestClient, workspace_headers) -> None:
        workspace_id = "ws_delete_api"
        created = test_client.post(
            "/v1/workspaces",
            json={"id": workspace_id, "name": workspace_id},
            headers=workspace_headers,
        )
        assert created.status_code == 201, created.text

        deleted = test_client.delete(
            f"/v1/workspaces/{workspace_id}",
            headers=workspace_headers,
        )
        assert deleted.status_code == 204, deleted.text

        missing = test_client.get(
            f"/v1/workspaces/{workspace_id}",
            headers=workspace_headers,
        )
        assert missing.status_code == 404, missing.text

        deleted_again = test_client.delete(
            f"/v1/workspaces/{workspace_id}",
            headers=workspace_headers,
        )
        assert deleted_again.status_code == 404, deleted_again.text
