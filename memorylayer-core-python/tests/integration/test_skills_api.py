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

    def test_create_skill_persists_inline_files(self, test_client: TestClient, workspace_headers: dict) -> None:
        """Bundle files sent inline on create (SDK save(files=...)) must persist —
        regression for silently dropping them because SkillCreateInput had no files
        field, which left skill_files empty and /skills/<name>/utils.py unresolvable.
        """
        resp = test_client.post(
            "/v1/skills",
            json={
                "name": "with-files",
                "description": "a skill that ships bundle files",
                "files": [
                    {"path": "utils.py", "content": "def helper():\n    return 42\n"},
                    {"path": "references/qa.md", "content": "# QA\nnotes\n"},
                ],
            },
            headers=workspace_headers,
        )
        assert resp.status_code == 201, resp.text
        skill_id = resp.json()["skill"]["id"]

        files_resp = test_client.get(f"/v1/skills/{skill_id}/files", headers=workspace_headers)
        assert files_resp.status_code == 200
        paths = {f["path"] for f in files_resp.json()["files"]}
        assert {"utils.py", "references/qa.md"} <= paths, paths

        one = test_client.get(f"/v1/skills/{skill_id}/files/utils.py", headers=workspace_headers)
        assert one.status_code == 200
        assert "return 42" in one.text

    def test_create_shared_skill_does_not_inherit_caller_identity(
        self, test_client: TestClient, workspace_headers: dict
    ) -> None:
        """A skill created without an explicit ``user_id`` must NOT inherit the
        caller's identity into ``user_id``.

        ``user_id`` is the user-PRIVATE owner (scope): a non-null value makes the
        skill user-private and drops it from the ``_global``/shared union (which
        filters ``user_id IS NULL``). Regression for the bug where the endpoint fell
        back to ``ctx.user_id``, so tenant-shared seeded skills got the seeder's
        principal as ``user_id`` and vanished from workspace listings. Authorship is
        recorded as provenance in ``metadata.created_by`` instead.
        """
        from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service
        from memorylayer_server.models.auth import RequestContext

        class _UserAuth:
            async def build_context(self, request, body=None):  # noqa: ANN001
                return RequestContext(
                    tenant_id="test_tenant", workspace_id="test_workspace", user_id="alice@example.com"
                )

        class _AllowAuthz:
            async def require_authorization(self, *args, **kwargs):  # noqa: ANN002, ANN003
                return None

        app = test_client.app
        app.dependency_overrides[get_auth_service] = lambda: _UserAuth()
        app.dependency_overrides[get_authz_service] = lambda: _AllowAuthz()
        try:
            # No explicit user_id -> shared skill: user_id stays NULL, authorship recorded.
            resp = test_client.post(
                "/v1/skills",
                json={"name": "shared-skill", "description": "A tenant-shared skill"},
                headers=workspace_headers,
            )
            assert resp.status_code == 201, resp.text
            skill = resp.json()["skill"]
            assert skill["user_id"] is None
            assert skill["metadata"].get("created_by") == "alice@example.com"

            # An explicit user_id IS honored (genuine user-private skill).
            resp2 = test_client.post(
                "/v1/skills",
                json={"name": "private-skill", "description": "A user-private skill", "user_id": "bob"},
                headers=workspace_headers,
            )
            assert resp2.status_code == 201, resp2.text
            assert resp2.json()["skill"]["user_id"] == "bob"
        finally:
            app.dependency_overrides.pop(get_auth_service, None)
            app.dependency_overrides.pop(get_authz_service, None)


class TestVersionedSkillManifest:
    def test_conditional_lifecycle_replay_history_and_bundle_independence(
        self,
        test_client: TestClient,
        workspace_headers: dict,
    ) -> None:
        create_headers = {
            **workspace_headers,
            "Idempotency-Key": "skill-create-1",
            "If-None-Match": "*",
        }
        payload = {
            "name": "revisioned-skill",
            "description": "Initial description",
            "body": "## Initial",
        }
        created = test_client.post("/v1/skills", json=payload, headers=create_headers)
        assert created.status_code == 201, created.text
        first = created.json()["skill"]
        assert first["revision"] == 1
        assert first["etag"] == created.headers["etag"]
        assert created.json()["replayed"] is False

        replay = test_client.post("/v1/skills", json=payload, headers=create_headers)
        assert replay.status_code == 201, replay.text
        assert replay.json()["replayed"] is True
        assert replay.json()["skill"] == first

        conflicting = test_client.post(
            "/v1/skills",
            json={**payload, "description": "Different request"},
            headers=create_headers,
        )
        assert conflicting.status_code == 409

        # Bundle files are child resources. Their aggregate hash changes, but
        # manifest history and the manifest ETag do not.
        file_put = test_client.put(
            f"/v1/skills/{first['id']}/files/scripts/run.py",
            json={"content_b64": base64.b64encode(b"print('ok')").decode()},
            headers=workspace_headers,
        )
        assert file_put.status_code == 200, file_put.text
        after_file = test_client.get(
            f"/v1/skills/{first['id']}", headers=workspace_headers
        )
        assert after_file.status_code == 200
        assert after_file.json()["skill"]["bundle_hash"]
        assert after_file.json()["skill"]["etag"] == first["etag"]
        assert after_file.json()["skill"]["revision"] == 1

        replace_payload = {
            "description": "Refined description",
            "version": "0.2.0",
            "body": "## Refined",
            "metadata": {"source": "refinement"},
            "source_mode": "server",
            "enabled": True,
        }
        replace_headers = {
            **workspace_headers,
            "Idempotency-Key": "skill-replace-1",
            "If-Match": first["etag"],
        }
        replaced = test_client.put(
            f"/v1/skills/{first['id']}/manifest",
            json=replace_payload,
            headers=replace_headers,
        )
        assert replaced.status_code == 200, replaced.text
        second = replaced.json()["skill"]
        assert second["revision"] == 2
        assert second["etag"] != first["etag"]
        assert second["bundle_hash"] == after_file.json()["skill"]["bundle_hash"]

        stale = test_client.put(
            f"/v1/skills/{first['id']}/manifest",
            json={**replace_payload, "description": "Stale writer"},
            headers={
                **workspace_headers,
                "Idempotency-Key": "skill-replace-stale",
                "If-Match": first["etag"],
            },
        )
        assert stale.status_code == 412

        first_history = test_client.get(
            f"/v1/skills/{first['id']}/revisions",
            params={"limit": 1},
            headers=workspace_headers,
        )
        assert first_history.status_code == 200
        assert first_history.json()["revisions"][0]["action"] == "replace"
        assert first_history.json()["next_page_token"]
        older = test_client.get(
            f"/v1/skills/{first['id']}/revisions",
            params={"limit": 1, "page_token": first_history.json()["next_page_token"]},
            headers=workspace_headers,
        )
        assert older.status_code == 200
        assert older.json()["revisions"][0]["action"] == "create"

        deleted = test_client.post(
            f"/v1/skills/{first['id']}/delete",
            headers={
                **workspace_headers,
                "Idempotency-Key": "skill-delete-1",
                "If-Match": second["etag"],
            },
        )
        assert deleted.status_code == 200, deleted.text
        tombstone = deleted.json()["skill"]
        assert tombstone["deleted_at"] is not None
        assert tombstone["revision"] == 3
        assert test_client.get(
            f"/v1/skills/{first['id']}", headers=workspace_headers
        ).status_code == 404

        fetched_tombstone = test_client.get(
            f"/v1/skills/{first['id']}",
            params={"include_deleted": "true"},
            headers=workspace_headers,
        )
        assert fetched_tombstone.status_code == 200
        assert fetched_tombstone.headers["etag"] == tombstone["etag"]

        restored = test_client.post(
            f"/v1/skills/{first['id']}/restore",
            headers={
                **workspace_headers,
                "Idempotency-Key": "skill-restore-1",
                "If-Match": tombstone["etag"],
            },
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["skill"]["revision"] == 4
        assert restored.json()["skill"]["deleted_at"] is None

    def test_conditional_routes_require_headers(
        self,
        test_client: TestClient,
        workspace_headers: dict,
    ) -> None:
        created = test_client.post(
            "/v1/skills",
            json={"name": "required-headers", "description": "Header test"},
            headers=workspace_headers,
        )
        skill = created.json()["skill"]
        response = test_client.put(
            f"/v1/skills/{skill['id']}/manifest",
            json={"description": "Changed"},
            headers=workspace_headers,
        )
        assert response.status_code == 428

    def test_scoped_name_is_reserved_across_tombstone(
        self,
        test_client: TestClient,
        workspace_headers: dict,
    ) -> None:
        created = test_client.post(
            "/v1/skills",
            json={"name": "reserved-name", "description": "Original identity"},
            headers={
                **workspace_headers,
                "Idempotency-Key": "reserved-create-1",
                "If-None-Match": "*",
            },
        )
        assert created.status_code == 201, created.text
        skill = created.json()["skill"]
        deleted = test_client.post(
            f"/v1/skills/{skill['id']}/delete",
            headers={
                **workspace_headers,
                "Idempotency-Key": "reserved-delete-1",
                "If-Match": skill["etag"],
            },
        )
        assert deleted.status_code == 200, deleted.text

        duplicate = test_client.post(
            "/v1/skills",
            json={"name": "reserved-name", "description": "Replacement identity"},
            headers={
                **workspace_headers,
                "Idempotency-Key": "reserved-create-2",
                "If-None-Match": "*",
            },
        )
        assert duplicate.status_code == 409


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

    def test_get_skill_manifest_can_include_accepted_addenda(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "manifest-addenda-skill", "description": "Manifest addenda test", "body": "## Usage\nDo things."},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]
        accepted = "Always pass `--reuse-cache` after a timeout retry."
        draft = "Draft note that should not be attached."

        for content, note_status in ((accepted, "accepted"), (draft, "draft")):
            memory_resp = test_client.post(
                "/v1/memories",
                json={
                    "content": content,
                    "type": "procedural",
                    "subtype": "skill_addendum",
                    "metadata": {"skill_id": skill_id, "status": note_status},
                },
                headers=workspace_headers,
            )
            assert memory_resp.status_code == 201

        default_resp = test_client.get(f"/v1/skills/{skill_id}/manifest", headers=workspace_headers)
        assert default_resp.status_code == 200
        assert "Learned Addenda" not in default_resp.text
        assert accepted not in default_resp.text

        composed_resp = test_client.get(
            f"/v1/skills/{skill_id}/manifest",
            params={"include_addenda": "true"},
            headers=workspace_headers,
        )
        assert composed_resp.status_code == 200
        assert "## Learned Addenda" in composed_resp.text
        assert accepted in composed_resp.text
        assert draft not in composed_resp.text


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

    def test_list_skills_can_include_accepted_addenda(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "list-addenda-skill", "description": "List addenda test", "body": "## Usage\nOriginal."},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]
        note = "Use the JSONL importer when CSV rows contain nested payloads."
        memory_resp = test_client.post(
            "/v1/memories",
            json={
                "content": note,
                "type": "procedural",
                "subtype": "skill_addendum",
                "metadata": {"skill_id": skill_id, "status": "accepted"},
            },
            headers=workspace_headers,
        )
        assert memory_resp.status_code == 201

        plain_resp = test_client.get("/v1/skills", params={"name": "list-addenda-skill"}, headers=workspace_headers)
        assert plain_resp.status_code == 200
        assert note not in plain_resp.json()["skills"][0]["body"]

        composed_resp = test_client.get(
            "/v1/skills",
            params={"name": "list-addenda-skill", "include_addenda": "true"},
            headers=workspace_headers,
        )
        assert composed_resp.status_code == 200
        assert note in composed_resp.json()["skills"][0]["body"]


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

    def test_delete_file(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "delete-file-skill", "description": "Skill for delete file test"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        content_b64 = base64.b64encode(b"content").decode()
        for name in ("scripts/a.py", "scripts/b.py"):
            test_client.put(
                f"/v1/skills/{skill_id}/files/{name}",
                json={"content_b64": content_b64},
                headers=workspace_headers,
            )

        # DELETE one file → 204, and it disappears from the listing.
        del_resp = test_client.delete(
            f"/v1/skills/{skill_id}/files/scripts/b.py",
            headers=workspace_headers,
        )
        assert del_resp.status_code == 204

        list_resp = test_client.get(f"/v1/skills/{skill_id}/files", headers=workspace_headers)
        paths = {f["path"] for f in list_resp.json()["files"]}
        assert "scripts/a.py" in paths
        assert "scripts/b.py" not in paths

    def test_delete_file_idempotent_when_missing(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "delete-missing-file-skill", "description": "Skill for idempotent delete test"},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]

        # Deleting a file that was never uploaded is idempotent (204).
        del_resp = test_client.delete(
            f"/v1/skills/{skill_id}/files/scripts/never.py",
            headers=workspace_headers,
        )
        assert del_resp.status_code == 204

    def test_delete_file_unknown_skill_returns_404(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.delete(
            "/v1/skills/nonexistent_id/files/scripts/x.py",
            headers=workspace_headers,
        )
        assert response.status_code == 404


class TestSkillStubs:
    def test_resolve_missing_params_returns_400(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post("/v1/skills/resolve", json={}, headers=workspace_headers)
        assert response.status_code == 400

    def test_resolve_by_name_not_found_returns_skill_none(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post("/v1/skills/resolve", json={"name": "nonexistent-skill"}, headers=workspace_headers)
        assert response.status_code == 200
        assert response.json()["skill"] is None

    def test_resolve_by_name_can_include_accepted_addenda(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/skills",
            json={"name": "resolve-addenda-skill", "description": "Resolve addenda test", "body": "## Usage\nOriginal."},
            headers=workspace_headers,
        )
        skill_id = create_resp.json()["skill"]["id"]
        note = "Prefer the deterministic parser for repeated import failures."
        memory_resp = test_client.post(
            "/v1/memories",
            json={
                "content": note,
                "type": "procedural",
                "subtype": "skill_addendum",
                "metadata": {"skill_id": skill_id, "status": "accepted"},
            },
            headers=workspace_headers,
        )
        assert memory_resp.status_code == 201

        response = test_client.post(
            "/v1/skills/resolve",
            json={"name": "resolve-addenda-skill", "include_addenda": True},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        assert note in response.json()["skill"]["body"]

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
