"""HTTP contracts for raw checkpoints and deterministic context delivery."""

import hashlib
from uuid import uuid4

from fastapi.testclient import TestClient


def test_checkpoint_and_context_delta_round_trip(
    test_client: TestClient,
    workspace_headers: dict[str, str],
) -> None:
    session_id = f"sess_deterministic_{uuid4().hex[:12]}"
    created = test_client.post(
        "/v1/sessions",
        json={
            "session_id": session_id,
            "ttl_seconds": 3600,
            "working_memory": {"open_threads": ["finish the context contract"]},
        },
        headers=workspace_headers,
    )
    assert created.status_code == 201, created.text

    transcript = "User: retain this exact pre-compaction segment.\n"
    checkpoint_request = {
        "transcript_segment": transcript,
        "source_kind": "test_transcript",
        "source_boundary": len(transcript.encode()),
        "content_hash": hashlib.sha256(transcript.encode()).hexdigest(),
        "idempotency_key": f"checkpoint-{uuid4().hex}",
    }
    first = test_client.post(
        f"/v1/sessions/{session_id}/checkpoints",
        json=checkpoint_request,
        headers=workspace_headers,
    )
    replay = test_client.post(
        f"/v1/sessions/{session_id}/checkpoints",
        json=checkpoint_request,
        headers=workspace_headers,
    )
    assert first.status_code == 201, first.text
    assert replay.status_code == 201, replay.text
    checkpoint = first.json()
    assert replay.json()["id"] == checkpoint["id"]
    assert replay.json()["raw_memory_id"] == checkpoint["raw_memory_id"]
    assert checkpoint["capture_status"] == "durable"
    assert checkpoint["byte_count"] == len(transcript.encode())

    fetched = test_client.get(
        f"/v1/sessions/{session_id}/checkpoints/{checkpoint['id']}",
        headers=workspace_headers,
    )
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == replay.json()
    raw = test_client.get(
        f"/v1/memories/{checkpoint['raw_memory_id']}",
        headers=workspace_headers,
    )
    assert raw.status_code == 200, raw.text
    assert raw.json()["memory"]["content"] == transcript

    invalid = test_client.post(
        f"/v1/sessions/{session_id}/checkpoints",
        json={
            **checkpoint_request,
            "idempotency_key": f"bad-hash-{uuid4().hex}",
            "content_hash": "0" * 64,
        },
        headers=workspace_headers,
    )
    assert invalid.status_code == 400

    pack = test_client.post(
        f"/v1/sessions/{session_id}/context-pack",
        json={
            "budget_tokens": 64,
            "include_directives": False,
            "include_recent_activity": False,
            "include_contradictions": False,
            "include_sandbox_summary": False,
            "include_checkpoint_recovery": False,
        },
        headers=workspace_headers,
    )
    assert pack.status_code == 200, pack.text
    pack_body = pack.json()
    assert pack_body["budget_summary"]["used"] <= 64
    assert pack_body["generation_summary"]["calls"] == 0
    assert pack_body["open_threads"]

    changed = test_client.post(
        f"/v1/sessions/{session_id}/memory",
        json={"key": "next_step", "value": "ship deterministic context"},
        headers=workspace_headers,
    )
    assert changed.status_code == 201, changed.text
    delta_request = {"cursor": pack_body["cursor"], "budget_tokens": 64}
    first_delta = test_client.post(
        f"/v1/sessions/{session_id}/context-delta",
        json=delta_request,
        headers=workspace_headers,
    )
    replay_delta = test_client.post(
        f"/v1/sessions/{session_id}/context-delta",
        json=delta_request,
        headers=workspace_headers,
    )
    assert first_delta.status_code == 200, first_delta.text
    assert replay_delta.status_code == 200, replay_delta.text
    assert first_delta.json() == replay_delta.json()
    assert first_delta.json()["budget_summary"]["used"] <= 64
    assert any(item["id"] == "working:next_step" for item in first_delta.json()["items"])
