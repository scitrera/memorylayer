"""Integration coverage for append-only refinement records."""

import uuid

from fastapi.testclient import TestClient


def _proposal(suffix: str, workspace: str) -> dict:
    return {
        "key": f"refinements/{suffix}/proposal",
        "refinement_id": f"refine-{suffix}",
        "phase": "proposal",
        "trigger": "user",
        "scope": "workspace",
        "summary": "Add evidence-review guidance",
        "rationale": "Repeated completion claims lacked artifact references.",
        "expected_outcome": "Completion claims consistently include verifiable evidence.",
        "evidence": [
            {
                "kind": "user_instruction",
                "reference": f"message:{suffix}",
                "description": "The user requested evidence-backed refinement.",
            }
        ],
        "edits": [
            {
                "action": "create",
                "resource_kind": "agent_specification",
                "resource_key": "review/evidence",
                "reason": "Make the repeated review role reusable.",
                "content": {
                    "name": "Evidence reviewer",
                    "description": "Checks completion claims.",
                    "instructions": "Check cited artifacts before accepting a completion claim.",
                },
            }
        ],
        "outcome": "proposed",
        "task_ref": {"system": "aether", "id": f"task-{suffix}"},
        "workspace_id": workspace,
    }


def test_refinement_records_are_idempotent_append_only_and_workspace_scoped(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    workspace = f"refinements_{suffix}"
    other_workspace = f"refinements_other_{suffix}"
    payload = _proposal(suffix, workspace)
    headers = {"Idempotency-Key": f"refinement-create-{suffix}", "If-None-Match": "*"}

    created = test_client.post("/v1/refinement-records", json=payload, headers=headers)
    assert created.status_code == 201, created.text
    record = created.json()["record"]
    assert record["refinement_id"] == f"refine-{suffix}"
    assert record["revision"] == 1
    assert record["task_ref"]["system"] == "aether"
    assert record["edits"][0]["content"]["name"] == "Evidence reviewer"

    replay = test_client.post("/v1/refinement-records", json=payload, headers=headers)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True

    page = test_client.get("/v1/refinement-records", params={"workspace_id": workspace, "limit": 1})
    assert page.status_code == 200
    assert [item["id"] for item in page.json()["records"]] == [record["id"]]

    isolated = test_client.get("/v1/refinement-records", params={"workspace_id": other_workspace})
    assert isolated.status_code == 200
    assert isolated.json()["records"] == []

    fetched = test_client.get(f"/v1/refinement-records/{record['id']}", params={"workspace_id": workspace})
    assert fetched.status_code == 200
    assert fetched.headers["etag"] == record["etag"]

    immutable = test_client.put(
        f"/v1/refinement-records/{record['id']}",
        json=payload,
        headers={"Idempotency-Key": f"refinement-update-{suffix}", "If-Match": record["etag"]},
    )
    assert immutable.status_code == 405

    changed = _proposal(suffix, workspace)
    changed["summary"] = "Conflicting summary"
    conflict = test_client.post(
        "/v1/refinement-records",
        json=changed,
        headers={"Idempotency-Key": f"refinement-conflict-{suffix}", "If-None-Match": "*"},
    )
    assert conflict.status_code == 409


def test_refinement_record_enforces_phase_outcome_and_evidence(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    payload = _proposal(suffix, f"refinements_invalid_{suffix}")
    payload["outcome"] = "applied"
    invalid_phase = test_client.post(
        "/v1/refinement-records",
        json=payload,
        headers={"Idempotency-Key": f"refinement-invalid-phase-{suffix}", "If-None-Match": "*"},
    )
    assert invalid_phase.status_code == 422

    payload = _proposal(suffix + "e", f"refinements_invalid_{suffix}")
    payload["evidence"] = []
    no_evidence = test_client.post(
        "/v1/refinement-records",
        json=payload,
        headers={"Idempotency-Key": f"refinement-no-evidence-{suffix}", "If-None-Match": "*"},
    )
    assert no_evidence.status_code == 422

    restore_payload = _proposal(suffix + "r", f"refinements_invalid_{suffix}")
    restore_payload["edits"] = [
        {
            "action": "restore",
            "resource_kind": "prompt_note",
            "resource_key": "coding/style",
            "resource_id": "note-1",
            "expected_etag": '"vr-3-tombstone"',
            "reason": "Undo the approved deletion.",
        }
    ]
    schema_v1_restore = test_client.post(
        "/v1/refinement-records",
        json=restore_payload,
        headers={"Idempotency-Key": f"refinement-restore-v1-{suffix}", "If-None-Match": "*"},
    )
    assert schema_v1_restore.status_code == 422

    restore_payload["schema_version"] = 2
    schema_v2_restore = test_client.post(
        "/v1/refinement-records",
        json=restore_payload,
        headers={"Idempotency-Key": f"refinement-restore-v2-{suffix}", "If-None-Match": "*"},
    )
    assert schema_v2_restore.status_code == 201, schema_v2_restore.text
    assert schema_v2_restore.json()["record"]["edits"][0]["action"] == "restore"


def test_refinement_record_query_filters_search_and_scopes_cursor(test_client: TestClient) -> None:
    suffix = uuid.uuid4().hex[:10]
    workspace = f"refinements_query_{suffix}"
    first = _proposal(suffix + "a", workspace)
    first["summary"] = "Investigate durable evidence mismatch"
    second = _proposal(suffix + "b", workspace)
    second["summary"] = "Unrelated prompt guidance"
    second["edits"][0]["resource_kind"] = "prompt_note"

    def headers(key: str) -> dict[str, str]:
        return {"Idempotency-Key": key, "If-None-Match": "*"}

    for index, payload in enumerate((first, second)):
        response = test_client.post(
            "/v1/refinement-records",
            json=payload,
            headers=headers(f"refinement-query-{suffix}-{index}"),
        )
        assert response.status_code == 201, response.text

    page = test_client.get(
        "/v1/refinement-records",
        params={
            "workspace_id": workspace,
            "phase": "proposal",
            "outcome": "proposed",
            "scope": "workspace",
            "resource_kind": "agent_specification",
            "search": "EVIDENCE MISMATCH",
            "limit": 1,
        },
    )
    assert page.status_code == 200, page.text
    body = page.json()
    assert [record["refinement_id"] for record in body["records"]] == [first["refinement_id"]]
    assert body["scanned_count"] == 2
    assert body["scan_truncated"] is False

    invalid = test_client.get(
        "/v1/refinement-records",
        params={"workspace_id": workspace, "phase": "future"},
    )
    assert invalid.status_code == 400

    first_page = test_client.get(
        "/v1/refinement-records",
        params={"workspace_id": workspace, "phase": "proposal", "limit": 1},
    )
    assert first_page.status_code == 200
    cursor = first_page.json()["next_page_token"]
    assert cursor
    mismatched = test_client.get(
        "/v1/refinement-records",
        params={"workspace_id": workspace, "phase": "decision", "limit": 1, "page_token": cursor},
    )
    assert mismatched.status_code == 400
