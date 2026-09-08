"""Integration tests for the email ingest endpoint (cross-source ingestion).

POST /v1/ingest/email normalizes an email and routes it through the SAME
``MemoryService.remember()`` pipeline chat + document ingestion use. These tests
assert the end-to-end behavior over the real memory service (sqlite + mock
embeddings, the test stack):

  * an email -> a stored memory with source=EMAIL, observer_id=sender, and
    content grounded with a ``[date] Subject:`` prefix.
  * the memory rides the SHARED pipeline: ``enqueue_post_store`` is invoked for
    the email memory (same enrichment hook chat/doc memories get) — asserted via
    a spy on the live memory service.
  * required-field validation (empty sender/body -> 4xx).
"""

import pytest
from fastapi.testclient import TestClient
from scitrera_app_framework import get_extension

from memorylayer_server.models.memory import SourceType
from memorylayer_server.services.memory import EXT_MEMORY_SERVICE

WS = "test_workspace"


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    return {"X-Workspace-ID": WS}


class TestEmailIngest:
    def test_ingest_email_creates_email_sourced_memory(
        self, test_client: TestClient, workspace_headers: dict[str, str]
    ) -> None:
        """An ingested email -> a memory with source=EMAIL, observer_id=sender,
        grounded content."""
        response = test_client.post(
            "/v1/ingest/email",
            json={
                "sender": "Alice",
                "to": ["Bob", "Carol"],
                "subject": "Phoenix launch",
                "body": "Phoenix will go live on March 14 per the Acme roadmap.",
                "timestamp": "2026-01-08T00:00:00Z",
                "thread_id": "thread-phoenix",
                "message_id": "msg-1",
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201, response.text
        memory = response.json()["memory"]
        # observer_id is the sender (the perspective anchor).
        assert memory["observer_id"] == "Alice"
        # source=EMAIL is stamped in metadata (read by coverage/attribution scorers).
        assert memory["metadata"]["source"] == SourceType.EMAIL.value
        assert memory["metadata"]["sender"] == "Alice"
        assert memory["metadata"]["recipients"] == ["Bob", "Carol"]
        # Content is grounded with the [date] Subject: prefix.
        assert memory["content"].startswith("[2026-01-08] Phoenix launch:")
        assert "Phoenix will go live on March 14" in memory["content"]
        # event_time carries the email timestamp (temporal grounding).
        assert memory["event_time"] is not None
        # Thread linkage round-trips through the shared remember() pipeline.
        assert memory["source_thread_id"] == "thread-phoenix"

    def test_ingest_email_routes_through_shared_pipeline(
        self, test_client: TestClient, workspace_headers: dict[str, str], monkeypatch
    ) -> None:
        """The email memory hits the SAME post-store hook chat/doc use:
        ``enqueue_post_store`` is invoked (no parallel enrichment fork)."""
        memory_service = get_extension(EXT_MEMORY_SERVICE, test_client.app.state.v)
        calls: list[str] = []
        original = memory_service.enqueue_post_store

        async def _spy(workspace_id, memory, embedding, **kwargs):
            calls.append(memory.id)
            return await original(workspace_id, memory, embedding, **kwargs)

        monkeypatch.setattr(memory_service, "enqueue_post_store", _spy)

        response = test_client.post(
            "/v1/ingest/email",
            json={
                "sender": "Dave",
                "subject": "Status",
                "body": "Hydra slipped two weeks.",
                "timestamp": "2026-02-01T00:00:00Z",
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201, response.text
        memory_id = response.json()["memory"]["id"]
        # The shared post-store hook ran for THIS email memory.
        assert memory_id in calls

    def test_ingest_email_appears_in_observer_scoped_recall(
        self, test_client: TestClient, workspace_headers: dict[str, str]
    ) -> None:
        """The ingested email is recallable with an observer_id filter = sender
        (perspective-scoped recall, the cross-source benchmark's expectation)."""
        ingest = test_client.post(
            "/v1/ingest/email",
            json={
                "sender": "Erin",
                "subject": "Globex sync",
                "body": "Globex agreed to the Denver pilot terms.",
                "timestamp": "2026-03-01T00:00:00Z",
            },
            headers=workspace_headers,
        )
        assert ingest.status_code == 201, ingest.text
        mem_id = ingest.json()["memory"]["id"]

        recall = test_client.post(
            "/v1/memories/recall",
            json={"query": "Globex Denver pilot terms", "observer_id": "Erin", "limit": 25},
            headers=workspace_headers,
        )
        assert recall.status_code == 200, recall.text
        ids = {m["id"] for m in recall.json()["memories"]}
        assert mem_id in ids

    def test_ingest_email_empty_sender_rejected(
        self, test_client: TestClient, workspace_headers: dict[str, str]
    ) -> None:
        """Empty sender is rejected (FastAPI validation -> 422)."""
        response = test_client.post(
            "/v1/ingest/email",
            json={"sender": "", "body": "x"},
            headers=workspace_headers,
        )
        assert response.status_code == 422

    def test_ingest_email_empty_body_rejected(
        self, test_client: TestClient, workspace_headers: dict[str, str]
    ) -> None:
        """Empty body is rejected (FastAPI validation -> 422)."""
        response = test_client.post(
            "/v1/ingest/email",
            json={"sender": "Alice", "body": ""},
            headers=workspace_headers,
        )
        assert response.status_code == 422
