"""Integration coverage for reusable agent-specification resources."""

import uuid

from fastapi.testclient import TestClient


def test_agent_specification_cas_history_and_workspace_isolation(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    workspace = f"agent_specs_{suffix}"
    other_workspace = f"agent_specs_other_{suffix}"
    payload = {
        "key": f"review/{suffix}",
        "name": "Evidence reviewer",
        "description": "Review claims against recorded evidence",
        "instructions": "Inspect the supplied evidence and report unsupported claims.",
        "invocation_guidance": "Use before accepting a completion claim.",
        "model": "review-model",
        "max_turns": 4,
        "allowed_tools": ["read_file"],
        "denied_tools": ["shell"],
        "skills": ["code-review"],
        "mcp_servers": ["docs"],
        "permission_mode": "read_only",
        "enabled": True,
        "workspace_id": workspace,
    }
    create_headers = {"Idempotency-Key": f"agent-create-{suffix}", "If-None-Match": "*"}

    created = test_client.post("/v1/agent-specifications", json=payload, headers=create_headers)
    assert created.status_code == 201, created.text
    specification = created.json()["specification"]
    assert specification["key"] == payload["key"]
    assert specification["permission_mode"] == "read_only"
    assert specification["revision"] == 1
    assert created.headers["etag"] == specification["etag"]

    replay = test_client.post("/v1/agent-specifications", json=payload, headers=create_headers)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True

    isolated = test_client.get("/v1/agent-specifications", params={"workspace_id": other_workspace})
    assert isolated.status_code == 200
    assert isolated.json()["specifications"] == []

    replacement = dict(payload)
    replacement.pop("key")
    replacement["instructions"] = "Reject unsupported claims and cite the evidence reference."
    stale = test_client.put(
        f"/v1/agent-specifications/{specification['id']}",
        json=replacement,
        headers={"Idempotency-Key": f"agent-stale-{suffix}", "If-Match": '"stale"'},
    )
    assert stale.status_code == 412

    replaced = test_client.put(
        f"/v1/agent-specifications/{specification['id']}",
        json=replacement,
        headers={"Idempotency-Key": f"agent-replace-{suffix}", "If-Match": specification["etag"]},
    )
    assert replaced.status_code == 200, replaced.text
    replaced_specification = replaced.json()["specification"]
    assert replaced_specification["revision"] == 2

    history = test_client.get(
        f"/v1/agent-specifications/{specification['id']}/revisions",
        params={"workspace_id": workspace},
    )
    assert history.status_code == 200
    assert [item["action"] for item in history.json()["revisions"]] == ["replace", "create"]

    deleted = test_client.delete(
        f"/v1/agent-specifications/{specification['id']}",
        params={"workspace_id": workspace},
        headers={"Idempotency-Key": f"agent-delete-{suffix}", "If-Match": replaced_specification["etag"]},
    )
    assert deleted.status_code == 200
    deleted_specification = deleted.json()["specification"]
    assert deleted_specification["revision"] == 3

    restored = test_client.post(
        f"/v1/agent-specifications/{specification['id']}/restore",
        params={"workspace_id": workspace},
        headers={
            "Idempotency-Key": f"agent-restore-{suffix}",
            "If-Match": deleted_specification["etag"],
        },
    )
    assert restored.status_code == 200, restored.text
    restored_specification = restored.json()["specification"]
    assert restored_specification["id"] == specification["id"]
    assert restored_specification["instructions"] == replacement["instructions"]
    assert restored_specification["revision"] == 4
    assert restored_specification["deleted_at"] is None

    restored_history = test_client.get(
        f"/v1/agent-specifications/{specification['id']}/revisions",
        params={"workspace_id": workspace},
    )
    assert [item["action"] for item in restored_history.json()["revisions"]] == [
        "restore",
        "delete",
        "replace",
        "create",
    ]


def test_agent_specification_rejects_duplicate_policy_names(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    response = test_client.post(
        "/v1/agent-specifications",
        json={
            "key": f"invalid/{suffix}",
            "name": "Invalid",
            "description": "Invalid duplicate tool policy",
            "instructions": "Do something.",
            "allowed_tools": ["read_file", "read_file"],
            "workspace_id": f"agent_specs_invalid_{suffix}",
        },
        headers={"Idempotency-Key": f"agent-invalid-{suffix}", "If-None-Match": "*"},
    )
    assert response.status_code == 422
