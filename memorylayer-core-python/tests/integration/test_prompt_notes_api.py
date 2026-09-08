"""Integration coverage for the typed prompt-note HTTP contract."""

import uuid

from fastapi.testclient import TestClient


def test_prompt_note_cas_idempotency_history_and_workspace_isolation(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    workspace = f"prompt_notes_{suffix}"
    other_workspace = f"prompt_notes_other_{suffix}"
    payload = {
        "key": f"coding/{suffix}",
        "title": "Coding conventions",
        "content": "Prefer deterministic state transitions.",
        "enabled": True,
        "metadata": {"source": "review"},
        "workspace_id": workspace,
    }

    missing = test_client.post("/v1/prompt-notes", json=payload)
    assert missing.status_code == 428

    create_headers = {"Idempotency-Key": f"create-{suffix}", "If-None-Match": "*"}
    created = test_client.post("/v1/prompt-notes", json=payload, headers=create_headers)
    assert created.status_code == 201, created.text
    first = created.json()
    note = first["note"]
    assert first["replayed"] is False
    assert note["revision"] == 1
    assert note["workspace_id"] == workspace
    assert created.headers["etag"] == note["etag"]

    replay = test_client.post("/v1/prompt-notes", json=payload, headers=create_headers)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["note"]["id"] == note["id"]

    conflict_payload = {**payload, "title": "Different request"}
    conflict = test_client.post("/v1/prompt-notes", json=conflict_payload, headers=create_headers)
    assert conflict.status_code == 409

    replace_payload = {
        "title": "Reviewed conventions",
        "content": "Use CAS and stable cursors.",
        "enabled": True,
        "schema_version": 2,
        "metadata": {"source": "accepted-review"},
        "workspace_id": workspace,
    }
    replace_headers = {"Idempotency-Key": f"replace-{suffix}", "If-Match": note["etag"]}
    replaced = test_client.put(
        f"/v1/prompt-notes/{note['id']}", json=replace_payload, headers=replace_headers
    )
    assert replaced.status_code == 200, replaced.text
    second = replaced.json()["note"]
    assert second["revision"] == 2
    assert second["title"] == "Reviewed conventions"

    stale = test_client.put(
        f"/v1/prompt-notes/{note['id']}",
        json=replace_payload,
        headers={"Idempotency-Key": f"stale-{suffix}", "If-Match": note["etag"]},
    )
    assert stale.status_code == 412

    listed = test_client.get("/v1/prompt-notes", params={"workspace_id": workspace})
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["notes"]] == [note["id"]]

    isolated = test_client.get("/v1/prompt-notes", params={"workspace_id": other_workspace})
    assert isolated.status_code == 200
    assert isolated.json()["notes"] == []

    history = test_client.get(
        f"/v1/prompt-notes/{note['id']}/revisions", params={"workspace_id": workspace}
    )
    assert history.status_code == 200
    assert [item["action"] for item in history.json()["revisions"]] == ["replace", "create"]

    delete_headers = {"Idempotency-Key": f"delete-{suffix}", "If-Match": second["etag"]}
    deleted = test_client.delete(
        f"/v1/prompt-notes/{note['id']}",
        params={"workspace_id": workspace},
        headers=delete_headers,
    )
    assert deleted.status_code == 200
    assert deleted.json()["note"]["deleted_at"] is not None

    delete_replay = test_client.delete(
        f"/v1/prompt-notes/{note['id']}",
        params={"workspace_id": workspace},
        headers=delete_headers,
    )
    assert delete_replay.status_code == 200
    assert delete_replay.json()["replayed"] is True

    hidden = test_client.get(
        f"/v1/prompt-notes/{note['id']}", params={"workspace_id": workspace}
    )
    assert hidden.status_code == 404
    tombstone = test_client.get(
        f"/v1/prompt-notes/{note['id']}",
        params={"workspace_id": workspace, "include_deleted": True},
    )
    assert tombstone.status_code == 200
    assert tombstone.json()["note"]["revision"] == 3

    tombstone_note = tombstone.json()["note"]
    stale_restore = test_client.post(
        f"/v1/prompt-notes/{note['id']}/restore",
        params={"workspace_id": workspace},
        headers={"Idempotency-Key": f"restore-stale-{suffix}", "If-Match": second["etag"]},
    )
    assert stale_restore.status_code == 412

    restore_headers = {
        "Idempotency-Key": f"restore-{suffix}",
        "If-Match": tombstone_note["etag"],
    }
    restored = test_client.post(
        f"/v1/prompt-notes/{note['id']}/restore",
        params={"workspace_id": workspace},
        headers=restore_headers,
    )
    assert restored.status_code == 200, restored.text
    restored_note = restored.json()["note"]
    assert restored_note["id"] == note["id"]
    assert restored_note["key"] == note["key"]
    assert restored_note["title"] == second["title"]
    assert restored_note["revision"] == 4
    assert restored_note["deleted_at"] is None

    restore_replay = test_client.post(
        f"/v1/prompt-notes/{note['id']}/restore",
        params={"workspace_id": workspace},
        headers=restore_headers,
    )
    assert restore_replay.status_code == 200
    assert restore_replay.json()["replayed"] is True

    restore_active = test_client.post(
        f"/v1/prompt-notes/{note['id']}/restore",
        params={"workspace_id": workspace},
        headers={"Idempotency-Key": f"restore-active-{suffix}", "If-Match": restored_note["etag"]},
    )
    assert restore_active.status_code == 409

    restored_history = test_client.get(
        f"/v1/prompt-notes/{note['id']}/revisions", params={"workspace_id": workspace}
    )
    assert [item["action"] for item in restored_history.json()["revisions"]] == [
        "restore",
        "delete",
        "replace",
        "create",
    ]
