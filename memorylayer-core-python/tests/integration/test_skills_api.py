"""Integration tests for /v1/skills API endpoints."""

import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    return {"X-Workspace-ID": "test_workspace"}


class TestSkillCreate:
    def test_create_skill_minimal(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/skills",
            json={"name": "pdf-processing", "description": "Extract text and tables from PDF files"},
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "skill" in data
        skill = data["skill"]
        assert skill["name"] == "pdf-processing"
        assert skill["version"] == "0.1.0"
        assert skill["source_mode"] == "server"
        assert skill["enabled"] is True
        assert skill["id"].startswith("skl_")

    def test_create_skill_full(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/skills",
            json={
                "name": "table-extractor",
                "description": "Specialized table extraction tool",
                "version": "1.0.0",
                "license": "MIT",
                "body": "## Usage\nExtract tables from documents.",
                "source_mode": "server",
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        skill = data["skill"]
        assert skill["license"] == "MIT"
        assert skill["version"] == "1.0.0"

    def test_create_skill_invalid_name(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/skills",
            json={"name": "PDF Processing", "description": "Bad name with spaces"},
            headers=workspace_headers,
        )
        assert response.status_code == 422

    def test_create_skill_missing_description(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/skills",
            json={"name": "valid-name"},
            headers=workspace_headers,
        )
        assert response.status_code == 422


class TestSkillGet:
    def test_get_skill(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "get-test-skill", "description": "Skill for GET test"},
            headers=workspace_headers,
        )
        assert create_resp.status_code == 201
        skill_id = create_resp.json()["skill"]["id"]

        get_resp = test_client.get(f"/v1/skills/{skill_id}", headers=workspace_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["skill"]["id"] == skill_id

    def test_get_skill_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.get("/v1/skills/nonexistent_id", headers=workspace_headers)
        assert response.status_code == 404

    def test_get_skill_manifest(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "manifest-skill", "description": "Manifest test skill", "body": "## Usage\nDo things."},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        manifest_resp = test_client.get(f"/v1/skills/{skill_id}/manifest", headers=workspace_headers)
        assert manifest_resp.status_code == 200
        assert "text/markdown" in manifest_resp.headers["content-type"]
        text = manifest_resp.text
        assert "---" in text
        assert "manifest-skill" in text
        assert "## Usage" in text


class TestSkillList:
    def test_list_skills(self, test_client: TestClient, workspace_headers: dict) -> None:
        test_client.post(
            "/v1/skills",
            json={"name": "list-skill-a", "description": "First list skill"},
            headers=workspace_headers,
        )
        test_client.post(
            "/v1/skills",
            json={"name": "list-skill-b", "description": "Second list skill"},
            headers=workspace_headers,
        )

        response = test_client.get("/v1/skills", headers=workspace_headers)
        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert "total_count" in data
        names = [s["name"] for s in data["skills"]]
        assert "list-skill-a" in names
        assert "list-skill-b" in names

    def test_list_skills_filter_name(self, test_client: TestClient, workspace_headers: dict) -> None:
        test_client.post(
            "/v1/skills",
            json={"name": "filter-unique-skill", "description": "Unique skill for filter test"},
            headers=workspace_headers,
        )

        response = test_client.get(
            "/v1/skills",
            params={"name": "filter-unique-skill"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        skills = response.json()["skills"]
        assert all(s["name"] == "filter-unique-skill" for s in skills)


class TestSkillUpdate:
    def test_update_skill(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "update-test-skill", "description": "Original description"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        update_resp = test_client.put(
            f"/v1/skills/{skill_id}",
            json={"description": "Updated description", "version": "0.2.0"},
            headers=workspace_headers,
        )
        assert update_resp.status_code == 200
        skill = update_resp.json()["skill"]
        assert skill["description"] == "Updated description"
        assert skill["version"] == "0.2.0"

    def test_update_skill_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.put(
            "/v1/skills/nonexistent_id",
            json={"description": "New desc"},
            headers=workspace_headers,
        )
        assert response.status_code == 404


class TestSkillDelete:
    def test_delete_skill(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "delete-test-skill", "description": "To be deleted"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        del_resp = test_client.delete(f"/v1/skills/{skill_id}", headers=workspace_headers)
        assert del_resp.status_code == 204

        get_resp = test_client.get(f"/v1/skills/{skill_id}", headers=workspace_headers)
        assert get_resp.status_code == 404

    def test_delete_skill_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.delete("/v1/skills/nonexistent_id", headers=workspace_headers)
        assert response.status_code == 404


class TestSkillFiles:
    def test_upsert_and_get_file(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "file-test-skill", "description": "Skill with files"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        content = b"print('hello world')"
        content_b64 = base64.b64encode(content).decode()

        upsert_resp = test_client.put(
            f"/v1/skills/{skill_id}/files/scripts/hello.py",
            json={"content_b64": content_b64, "mime_type": "text/x-python"},
            headers=workspace_headers,
        )
        assert upsert_resp.status_code == 200
        file_info = upsert_resp.json()
        assert file_info["path"] == "scripts/hello.py"
        assert file_info["kind"] == "script"
        assert file_info["size_bytes"] == len(content)

    def test_list_files(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "list-files-skill", "description": "Skill for list files test"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        for name, kind_prefix in [("scripts/a.py", "script"), ("references/ref.md", "reference")]:
            content_b64 = base64.b64encode(b"content").decode()
            test_client.put(
                f"/v1/skills/{skill_id}/files/{name}",
                json={"content_b64": content_b64},
                headers=workspace_headers,
            )

        list_resp = test_client.get(f"/v1/skills/{skill_id}/files", headers=workspace_headers)
        assert list_resp.status_code == 200
        files = list_resp.json()["files"]
        paths = {f["path"] for f in files}
        assert "scripts/a.py" in paths
        assert "references/ref.md" in paths

    def test_stream_file(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "stream-file-skill", "description": "Skill for stream test"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        content = b"#!/usr/bin/env python\nprint('stream test')"
        content_b64 = base64.b64encode(content).decode()
        test_client.put(
            f"/v1/skills/{skill_id}/files/scripts/stream.py",
            json={"content_b64": content_b64, "mime_type": "text/x-python"},
            headers=workspace_headers,
        )

        stream_resp = test_client.get(
            f"/v1/skills/{skill_id}/files/scripts/stream.py",
            headers=workspace_headers,
        )
        assert stream_resp.status_code == 200
        assert stream_resp.content == content


class TestSkillStubs:
    def test_resolve_missing_params_returns_400(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post("/v1/skills/resolve", json={}, headers=workspace_headers)
        assert response.status_code == 400

    def test_resolve_by_name_not_found_returns_skill_none(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post("/v1/skills/resolve", json={"name": "nonexistent-skill"}, headers=workspace_headers)
        assert response.status_code == 200
        assert response.json()["skill"] is None

    def test_sync_returns_404_for_unknown_skill(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/skills/nonexistent-id/sync",
            json={"manifest_hash": "", "bundle_hash": ""},
            headers=workspace_headers,
        )
        assert response.status_code == 404

    def test_sync_in_sync(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "sync-test-skill", "description": "sync test"},
            headers=workspace_headers,
        )
        assert create_resp.status_code == 201
        skill = create_resp.json()["skill"]
        skill_id = skill["id"]

        sync_resp = test_client.post(
            f"/v1/skills/{skill_id}/sync",
            json={"manifest_hash": skill["manifest_hash"], "bundle_hash": skill["bundle_hash"]},
            headers=workspace_headers,
        )
        assert sync_resp.status_code == 200
        assert sync_resp.json()["action"] == "in_sync"

    def test_sync_pull_action_when_client_empty(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "sync-push-skill", "description": "push test"},
            headers=workspace_headers,
        )
        assert create_resp.status_code == 201
        skill_id = create_resp.json()["skill"]["id"]

        sync_resp = test_client.post(
            f"/v1/skills/{skill_id}/sync",
            json={"manifest_hash": "", "bundle_hash": ""},
            headers=workspace_headers,
        )
        assert sync_resp.status_code == 200
        assert sync_resp.json()["action"] == "pull"

    def test_bundle_returns_404_for_unknown_skill(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.get("/v1/skills/nonexistent-id/bundle", headers=workspace_headers)
        assert response.status_code == 404
