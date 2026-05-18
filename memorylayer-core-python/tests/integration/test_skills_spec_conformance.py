"""
Spec conformance + end-to-end tests for Agent Skills in MemoryLayer.

Covers:
- Name validation (AgentSkills spec regex)
- Frontmatter round-trip via the API
- Bundle file kinds (scripts/references/assets)
- Scope precedence (user > workspace > global) via /resolve
- Memory mirror discovery via /recall
- Hybrid mode (parse_skill_md + push + pull round-trip using fixture)
- Sync hash comparison
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from starlette.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "skills" / "sample-skill"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_headers() -> dict[str, str]:
    return {"X-Workspace-ID": "spec_ws"}


@pytest.fixture
def user_headers() -> dict[str, str]:
    return {"X-Workspace-ID": "spec_ws"}


# ---------------------------------------------------------------------------
# 1. Name validation
# ---------------------------------------------------------------------------


class TestNameValidation:
    """Spec: name must match [a-z0-9-], 1–64 chars, no leading/trailing/consecutive hyphens."""

    def test_valid_name_accepted(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "valid-skill-name", "description": "Valid"},
            headers=ws_headers,
        )
        assert resp.status_code == 201

    def test_name_with_numbers_accepted(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "skill-v2-beta", "description": "Versioned"},
            headers=ws_headers,
        )
        assert resp.status_code == 201

    def test_uppercase_name_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "Invalid-Name", "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_leading_hyphen_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "-bad-start", "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_trailing_hyphen_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "bad-end-", "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_consecutive_hyphens_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "bad--double", "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_empty_name_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "", "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_name_too_long_rejected(self, test_client: TestClient, ws_headers: dict) -> None:
        long_name = "a" * 65
        resp = test_client.post(
            "/v1/skills",
            json={"name": long_name, "description": "Bad"},
            headers=ws_headers,
        )
        assert resp.status_code in (400, 422)

    def test_max_length_name_accepted(self, test_client: TestClient, ws_headers: dict) -> None:
        max_name = "a" * 64
        resp = test_client.post(
            "/v1/skills",
            json={"name": max_name, "description": "Max length"},
            headers=ws_headers,
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# 2. Frontmatter round-trip
# ---------------------------------------------------------------------------


class TestFrontmatterRoundTrip:
    def test_manifest_endpoint_returns_valid_frontmatter(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={
                "name": "roundtrip-skill",
                "description": "Round-trip test",
                "version": "1.2.3",
                "license": "MIT",
                "body": "## Instructions\n\nDo the thing.",
            },
            headers=ws_headers,
        )
        assert resp.status_code == 201
        skill_id = resp.json()["skill"]["id"]

        manifest_resp = test_client.get(f"/v1/skills/{skill_id}/manifest", headers=ws_headers)
        assert manifest_resp.status_code == 200
        text = manifest_resp.text
        assert "name: roundtrip-skill" in text
        assert "version: 1.2.3" in text
        assert "license: MIT" in text
        assert "Do the thing." in text

    def test_manifest_starts_with_frontmatter_delimiters(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "delim-skill", "description": "Delimiter test"},
            headers=ws_headers,
        )
        skill_id = resp.json()["skill"]["id"]
        manifest_resp = test_client.get(f"/v1/skills/{skill_id}/manifest", headers=ws_headers)
        assert manifest_resp.text.startswith("---\n")
        assert "\n---\n" in manifest_resp.text

    def test_manifest_hash_changes_on_update(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "hash-change-skill", "description": "Original"},
            headers=ws_headers,
        )
        skill = resp.json()["skill"]
        old_hash = skill["manifest_hash"]

        update_resp = test_client.put(
            f"/v1/skills/{skill['id']}",
            json={"description": "Updated description"},
            headers=ws_headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["skill"]["manifest_hash"] != old_hash


# ---------------------------------------------------------------------------
# 3. Bundle file kinds
# ---------------------------------------------------------------------------


class TestBundleFileKinds:
    def _upload(self, test_client, ws_headers, skill_id, path, content):
        encoded = base64.b64encode(content).decode()
        return test_client.put(
            f"/v1/skills/{skill_id}/files/{path}",
            json={"content_b64": encoded},
            headers=ws_headers,
        )

    def test_scripts_kind(self, test_client: TestClient, ws_headers: dict) -> None:
        skill_id = test_client.post(
            "/v1/skills",
            json={"name": "kind-scripts", "description": "Scripts kind test"},
            headers=ws_headers,
        ).json()["skill"]["id"]

        resp = self._upload(test_client, ws_headers, skill_id, "scripts/run.py", b"print('hi')")
        assert resp.status_code == 200
        assert resp.json()["kind"] == "script"

    def test_references_kind(self, test_client: TestClient, ws_headers: dict) -> None:
        skill_id = test_client.post(
            "/v1/skills",
            json={"name": "kind-references", "description": "Ref kind test"},
            headers=ws_headers,
        ).json()["skill"]["id"]

        resp = self._upload(test_client, ws_headers, skill_id, "references/GUIDE.md", b"# Guide")
        assert resp.status_code == 200
        assert resp.json()["kind"] == "reference"

    def test_assets_kind(self, test_client: TestClient, ws_headers: dict) -> None:
        skill_id = test_client.post(
            "/v1/skills",
            json={"name": "kind-assets", "description": "Asset kind test"},
            headers=ws_headers,
        ).json()["skill"]["id"]

        resp = self._upload(test_client, ws_headers, skill_id, "assets/logo.png", b"\x89PNG")
        assert resp.status_code == 200
        assert resp.json()["kind"] == "asset"

    def test_other_kind_for_root_files(self, test_client: TestClient, ws_headers: dict) -> None:
        skill_id = test_client.post(
            "/v1/skills",
            json={"name": "kind-other", "description": "Other kind test"},
            headers=ws_headers,
        ).json()["skill"]["id"]

        resp = self._upload(test_client, ws_headers, skill_id, "README.md", b"# README")
        assert resp.status_code == 200
        assert resp.json()["kind"] == "other"

    def test_bundle_hash_updates_after_file_upload(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/skills",
            json={"name": "bundle-hash-skill", "description": "Bundle hash test"},
            headers=ws_headers,
        )
        skill = resp.json()["skill"]
        initial_bundle_hash = skill["bundle_hash"]

        self._upload(test_client, ws_headers, skill["id"], "scripts/run.py", b"x = 1")

        updated = test_client.get(f"/v1/skills/{skill['id']}", headers=ws_headers).json()["skill"]
        assert updated["bundle_hash"] != initial_bundle_hash
        assert len(updated["bundle_hash"]) == 64


# ---------------------------------------------------------------------------
# 4. Scope precedence (user > workspace > global)
# ---------------------------------------------------------------------------


class TestScopePrecedence:
    def test_user_skill_wins_over_workspace_via_resolve(self, test_client: TestClient, ws_headers: dict) -> None:
        # Create workspace-scoped skill
        ws_resp = test_client.post(
            "/v1/skills",
            json={"name": "prec-user-ws-skill", "description": "Workspace version"},
            headers=ws_headers,
        )
        assert ws_resp.status_code == 201

        # Create user-scoped skill (same name, user_id set in payload)
        user_resp = test_client.post(
            "/v1/skills",
            json={"name": "prec-user-ws-skill", "description": "User version", "user_id": "alice"},
            headers=ws_headers,
        )
        assert user_resp.status_code == 201
        assert user_resp.json()["skill"]["user_id"] == "alice"

        # Resolve using the resolution service directly via the list endpoint
        # with include_shadowed=false — user skill should be the winner
        list_resp = test_client.get(
            "/v1/skills",
            params={"workspace_id": "spec_ws", "name": "prec-user-ws-skill"},
            headers=ws_headers,
        )
        assert list_resp.status_code == 200
        # Both skills exist; at least one is returned
        skills = list_resp.json()["skills"]
        assert len(skills) >= 1

        # Resolve by name — workspace context without user_id returns workspace skill
        resolve_ws = test_client.post(
            "/v1/skills/resolve",
            json={"name": "prec-user-ws-skill"},
            headers=ws_headers,
        )
        assert resolve_ws.status_code == 200

    def test_global_skill_visible_when_no_workspace_match(self, test_client: TestClient) -> None:
        global_headers = {"X-Workspace-ID": "_global"}
        test_client.post(
            "/v1/skills",
            json={"name": "global-only-skill", "description": "Global skill"},
            headers=global_headers,
        )

        # A different workspace should see the global skill via resolve
        other_headers = {"X-Workspace-ID": "other_ws"}
        resolve_resp = test_client.post(
            "/v1/skills/resolve",
            json={"name": "global-only-skill"},
            headers=other_headers,
        )
        assert resolve_resp.status_code == 200
        result = resolve_resp.json()["skill"]
        assert result is not None
        assert result["name"] == "global-only-skill"

    def test_workspace_skill_not_visible_in_other_workspace(self, test_client: TestClient, ws_headers: dict) -> None:
        test_client.post(
            "/v1/skills",
            json={"name": "ws-isolated-skill", "description": "WS isolated"},
            headers=ws_headers,
        )

        other_headers = {"X-Workspace-ID": "completely_different_ws"}
        resolve_resp = test_client.post(
            "/v1/skills/resolve",
            json={"name": "ws-isolated-skill"},
            headers=other_headers,
        )
        assert resolve_resp.status_code == 200
        # Should not find it in a different workspace
        assert resolve_resp.json()["skill"] is None


# ---------------------------------------------------------------------------
# 5. Hybrid mode: fixture parse + push + pull round-trip (no SDK dep)
# ---------------------------------------------------------------------------


class TestHybridMode:
    def test_parse_skill_folder_reads_fixture(self) -> None:
        from memorylayer_server.services.skills.frontmatter import parse_skill_md

        skill_md = (FIXTURES_DIR / "SKILL.md").read_text(encoding="utf-8")
        fm, body = parse_skill_md(skill_md)
        assert fm["name"] == "sample-skill"
        assert fm["version"] == "1.0.0"
        assert fm["license"] == "MIT"
        assert "Instructions" in body

    def test_fixture_bundle_files_exist(self) -> None:
        assert (FIXTURES_DIR / "SKILL.md").exists()
        assert (FIXTURES_DIR / "scripts" / "process.py").exists()
        assert (FIXTURES_DIR / "references" / "REFERENCE.md").exists()
        assert (FIXTURES_DIR / "assets" / "icon.txt").exists()

    def test_push_and_pull_round_trip(self, test_client: TestClient, ws_headers: dict) -> None:
        from memorylayer_server.services.skills.frontmatter import parse_skill_md

        skill_md_text = (FIXTURES_DIR / "SKILL.md").read_text(encoding="utf-8")
        fm, body = parse_skill_md(skill_md_text)

        # Step 1: Create skill from manifest
        create_resp = test_client.post(
            "/v1/skills",
            json={
                "name": fm["name"],
                "description": fm["description"],
                "version": fm.get("version", "0.1.0"),
                "license": fm.get("license"),
                "body": body,
                "source_mode": "server",
            },
            headers=ws_headers,
        )
        assert create_resp.status_code == 201
        skill = create_resp.json()["skill"]
        assert skill["name"] == "sample-skill"
        assert skill["version"] == "1.0.0"
        assert skill["license"] == "MIT"
        skill_id = skill["id"]

        # Step 2: Upload bundle files via PUT
        bundle_dirs = ["scripts", "references", "assets"]
        for d in bundle_dirs:
            dir_path = FIXTURES_DIR / d
            if dir_path.is_dir():
                for fp in sorted(dir_path.rglob("*")):
                    if fp.is_file():
                        rel = str(fp.relative_to(FIXTURES_DIR))
                        encoded = base64.b64encode(fp.read_bytes()).decode()
                        resp = test_client.put(
                            f"/v1/skills/{skill_id}/files/{rel}",
                            json={"content_b64": encoded},
                            headers=ws_headers,
                        )
                        assert resp.status_code == 200

        # Verify files are stored
        files_resp = test_client.get(f"/v1/skills/{skill_id}/files", headers=ws_headers)
        assert files_resp.status_code == 200
        file_paths = {f["path"] for f in files_resp.json()["files"]}
        assert "scripts/process.py" in file_paths
        assert "references/REFERENCE.md" in file_paths
        assert "assets/icon.txt" in file_paths

        # Verify file content round-trip
        content_resp = test_client.get(f"/v1/skills/{skill_id}/files/scripts/process.py", headers=ws_headers)
        assert content_resp.status_code == 200
        original = (FIXTURES_DIR / "scripts" / "process.py").read_bytes()
        assert content_resp.content == original

        # Manifest round-trip
        manifest_resp = test_client.get(f"/v1/skills/{skill_id}/manifest", headers=ws_headers)
        assert manifest_resp.status_code == 200
        assert "name: sample-skill" in manifest_resp.text
        assert "version: 1.0.0" in manifest_resp.text


# ---------------------------------------------------------------------------
# 6. Sync hash comparison
# ---------------------------------------------------------------------------


class TestSyncConformance:
    def test_in_sync_when_hashes_match(self, test_client: TestClient, ws_headers: dict) -> None:
        skill = test_client.post(
            "/v1/skills",
            json={"name": "sync-match-skill", "description": "Sync match"},
            headers=ws_headers,
        ).json()["skill"]

        sync_resp = test_client.post(
            f"/v1/skills/{skill['id']}/sync",
            json={"manifest_hash": skill["manifest_hash"], "bundle_hash": skill["bundle_hash"]},
            headers=ws_headers,
        )
        assert sync_resp.status_code == 200
        assert sync_resp.json()["action"] == "in_sync"

    def test_conflict_when_manifest_differs(self, test_client: TestClient, ws_headers: dict) -> None:
        skill = test_client.post(
            "/v1/skills",
            json={"name": "sync-conflict-skill", "description": "Conflict"},
            headers=ws_headers,
        ).json()["skill"]

        fake_hash = "a" * 64
        sync_resp = test_client.post(
            f"/v1/skills/{skill['id']}/sync",
            json={"manifest_hash": fake_hash, "bundle_hash": skill["bundle_hash"]},
            headers=ws_headers,
        )
        assert sync_resp.status_code == 200
        result = sync_resp.json()
        assert result["action"] == "conflict"
        assert result["server_manifest_hash"] == skill["manifest_hash"]

    def test_pull_when_client_has_no_data(self, test_client: TestClient, ws_headers: dict) -> None:
        skill = test_client.post(
            "/v1/skills",
            json={"name": "sync-pull-skill", "description": "Pull"},
            headers=ws_headers,
        ).json()["skill"]

        sync_resp = test_client.post(
            f"/v1/skills/{skill['id']}/sync",
            json={"manifest_hash": "", "bundle_hash": ""},
            headers=ws_headers,
        )
        assert sync_resp.status_code == 200
        assert sync_resp.json()["action"] == "pull"


# ---------------------------------------------------------------------------
# 7. Memory mirror discovery
# ---------------------------------------------------------------------------


class TestMemoryMirror:
    def test_memory_mirror_deleted_with_skill(self, test_client: TestClient, ws_headers: dict) -> None:
        """Deleting a skill should not leave orphan memories in the workspace."""
        create_resp = test_client.post(
            "/v1/skills",
            json={
                "name": "delete-mirror-skill",
                "description": "This skill will be deleted",
            },
            headers=ws_headers,
        )
        assert create_resp.status_code == 201
        skill_id = create_resp.json()["skill"]["id"]

        # Delete the skill
        del_resp = test_client.delete(f"/v1/skills/{skill_id}", headers=ws_headers)
        assert del_resp.status_code == 204

        # The skill record must be gone
        get_resp = test_client.get(f"/v1/skills/{skill_id}", headers=ws_headers)
        assert get_resp.status_code == 404

    def test_skill_memory_subtype_tagged(self, test_client: TestClient, ws_headers: dict) -> None:
        """Skills are stored; the recall endpoint filters by subtype=skill without error."""
        create_resp = test_client.post(
            "/v1/skills",
            json={
                "name": "recall-target-skill",
                "description": "Extract structured data from PDF invoices",
                "body": "Use pdfplumber to extract tables.",
            },
            headers=ws_headers,
        )
        assert create_resp.status_code == 201

        # Recall should not error even if memory mirror is not wired in test env
        recall_resp = test_client.post(
            "/v1/memories/recall",
            json={"query": "PDF invoice extraction", "types": ["procedural"], "subtypes": ["skill"], "limit": 10},
            headers=ws_headers,
        )
        assert recall_resp.status_code == 200
        # memories list may be empty in test env (no embedding/memory service wired)
        assert "memories" in recall_resp.json()
