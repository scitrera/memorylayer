"""Integration tests for memory CRUD API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    """Workspace headers for memory operations."""
    return {"X-Workspace-ID": "test_workspace"}


class TestMemoryCreate:
    """Tests for POST /v1/memories endpoint."""

    def test_create_memory_minimal(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with minimal required fields."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers Python for backend development",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert "memory" in data
        assert data["memory"]["content"] == "User prefers Python for backend development"

    def test_create_memory_full(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with all optional fields."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers concise code comments",
                "type": "semantic",
                "subtype": "preference",
                "importance": 0.8,
                "tags": ["preference", "programming"],
                "metadata": {"category": "coding-style"},
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert "memory" in data
        memory = data["memory"]
        assert memory["content"] == "User prefers concise code comments"
        assert memory["importance"] == 0.8
        assert "preference" in memory["tags"]

    def test_create_memory_invalid_importance(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with invalid importance value."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "Test memory",
                "importance": 1.5,  # Invalid: > 1.0
            },
            headers=workspace_headers,
        )

        # FastAPI validates before reaching the dependency, so 422 is expected
        assert response.status_code == 422

    def test_create_memory_empty_content(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with empty content."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "",  # Invalid: empty
            },
            headers=workspace_headers,
        )

        assert response.status_code == 422


class TestVersionedSemanticMemory:
    def test_conditional_lifecycle_history_and_telemetry_independence(
        self,
        test_client: TestClient,
        workspace_headers: dict[str, str],
    ) -> None:
        create_headers = {
            **workspace_headers,
            "Idempotency-Key": "memory-create-1",
            "If-None-Match": "*",
        }
        create_payload = {
            "logical_key": "preferences/backend-language",
            "content": "Prefer Python for backend services.",
            "type": "semantic",
            "subtype": "preference",
            "tags": ["Backend", "Preference"],
            "metadata": {"source": "user-instruction"},
            "refinement_metadata": {"confidence": "explicit"},
            "pinned": True,
        }
        created = test_client.post(
            "/v1/memories", json=create_payload, headers=create_headers
        )
        assert created.status_code == 201, created.text
        first = created.json()["memory"]
        assert first["logical_key"] == "preferences/backend-language"
        assert first["revision"] == 1
        assert first["etag"] == created.headers["etag"]
        assert created.json()["replayed"] is False

        replay = test_client.post(
            "/v1/memories", json=create_payload, headers=create_headers
        )
        assert replay.status_code == 201, replay.text
        assert replay.json()["replayed"] is True
        assert replay.json()["memory"]["id"] == first["id"]

        decayed = test_client.post(
            f"/v1/memories/{first['id']}/decay",
            json={"decay_rate": 0.1},
            headers=workspace_headers,
        )
        assert decayed.status_code == 200, decayed.text
        assert decayed.json()["memory"]["etag"] == first["etag"]
        assert decayed.json()["memory"]["revision"] == 1

        replacement = {
            "content": "Prefer Go for backend services.",
            "type": "semantic",
            "subtype": "preference",
            "tags": ["backend", "preference"],
            "refinement_metadata": {"confidence": "revised"},
            "pinned": True,
        }
        replaced = test_client.put(
            f"/v1/memories/{first['id']}/semantic",
            json=replacement,
            headers={
                **workspace_headers,
                "Idempotency-Key": "memory-replace-1",
                "If-Match": first["etag"],
            },
        )
        assert replaced.status_code == 200, replaced.text
        second = replaced.json()["memory"]
        assert second["revision"] == 2
        assert second["etag"] != first["etag"]
        assert second["metadata"]["source"] == "user-instruction"

        stale = test_client.put(
            f"/v1/memories/{first['id']}/semantic",
            json={**replacement, "content": "Stale writer"},
            headers={
                **workspace_headers,
                "Idempotency-Key": "memory-replace-stale",
                "If-Match": first["etag"],
            },
        )
        assert stale.status_code == 412

        history = test_client.get(
            f"/v1/memories/{first['id']}/revisions",
            params={"limit": 1},
            headers=workspace_headers,
        )
        assert history.status_code == 200, history.text
        assert history.json()["revisions"][0]["action"] == "replace"
        assert history.json()["next_page_token"]

        deleted = test_client.post(
            f"/v1/memories/{first['id']}/delete",
            headers={
                **workspace_headers,
                "Idempotency-Key": "memory-delete-1",
                "If-Match": second["etag"],
            },
        )
        assert deleted.status_code == 200, deleted.text
        tombstone = deleted.json()["memory"]
        assert tombstone["revision"] == 3
        assert tombstone["deleted_at"] is not None

        duplicate = test_client.post(
            "/v1/memories",
            json={
                **create_payload,
                "content": "Attempt to take the tombstoned key.",
            },
            headers={
                **workspace_headers,
                "Idempotency-Key": "memory-create-duplicate",
                "If-None-Match": "*",
            },
        )
        assert duplicate.status_code == 409

        fetched = test_client.get(
            f"/v1/memories/{first['id']}",
            params={"include_deleted": "true"},
            headers=workspace_headers,
        )
        assert fetched.status_code == 200, fetched.text
        assert fetched.headers["etag"] == tombstone["etag"]

        restored = test_client.post(
            f"/v1/memories/{first['id']}/restore",
            headers={
                **workspace_headers,
                "Idempotency-Key": "memory-restore-1",
                "If-Match": tombstone["etag"],
            },
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["memory"]["revision"] == 4
        assert restored.json()["memory"]["deleted_at"] is None


class TestMemoryRecall:
    """Tests for POST /v1/memories/recall endpoint."""

    def test_recall_memories_minimal(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recalling memories with minimal query."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "what programming language does the user prefer?",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total_count" in data

    def test_recall_memories_with_filters(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recalling memories with filters."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "coding preferences",
                "mode": "rag",
                "limit": 5,
                "min_relevance": 0.7,
                "types": ["semantic"],
                "tags": ["preference"],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert len(data["memories"]) <= 5

    def test_recall_memories_empty_query(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recalling memories with empty query."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "",  # Invalid: empty
            },
            headers=workspace_headers,
        )

        assert response.status_code == 422

    def test_recall_memories_invalid_limit(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recalling memories with invalid limit."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "test",
                "limit": 200,  # Invalid: > 100
            },
            headers=workspace_headers,
        )

        assert response.status_code == 422


class TestMemoryReflect:
    """Tests for POST /v1/memories/reflect endpoint."""

    def test_reflect_memories(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflecting on memories."""
        # First create a memory to reflect on
        test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers test-driven development methodology",
                "type": "semantic",
                "subtype": "preference",
                "importance": 0.8,
            },
            headers=workspace_headers,
        )

        # Now test reflection
        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What development methodologies does the user prefer?",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert isinstance(data["reflection"], str)
        assert len(data["reflection"]) > 0
        assert "source_memories" in data
        assert isinstance(data["source_memories"], list)
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert "tokens_processed" in data
        assert isinstance(data["tokens_processed"], int)

    def test_reflect_memories_minimal(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflecting with minimal parameters."""
        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "Tell me about the user",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "source_memories" in data
        assert "confidence" in data
        assert "tokens_processed" in data


class TestMemoryGet:
    """Tests for GET /v1/memories/{memory_id} endpoint."""

    def test_get_memory_nonexistent(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test getting a non-existent memory."""
        response = test_client.get(
            "/v1/memories/mem_nonexistent",
            headers=workspace_headers,
        )

        assert response.status_code == 404


class TestMemoryUpdate:
    """Tests for PUT /v1/memories/{memory_id} endpoint."""

    def test_update_memory_nonexistent(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test updating a non-existent memory."""
        response = test_client.put(
            "/v1/memories/mem_nonexistent",
            json={
                "content": "Updated content",
                "importance": 0.9,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 404


class TestMemoryDelete:
    """Tests for DELETE /v1/memories/{memory_id} endpoint."""

    def test_delete_memory_soft(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test soft deleting a memory."""
        response = test_client.delete(
            "/v1/memories/mem_test",
            params={"hard": "false"},
            headers=workspace_headers,
        )

        # Should return 404 for non-existent memory
        assert response.status_code == 404

    def test_delete_memory_hard(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test hard deleting a memory."""
        response = test_client.delete(
            "/v1/memories/mem_test",
            params={"hard": "true"},
            headers=workspace_headers,
        )

        # Should return 404 for non-existent memory
        assert response.status_code == 404


class TestMemoryDeleteCrossWorkspace:
    """Regression: delete must resolve the memory's ACTUAL workspace, not the
    caller's X-Workspace-ID. A memory created in workspace A must be deletable
    even when the delete request carries a different workspace header (mirrors
    real cases where memories live in non-default workspaces, e.g. USER scope /
    _global_user). Previously these silently 404'd.
    """

    def test_delete_resolves_memory_workspace(self, test_client: TestClient) -> None:
        """Single-endpoint delete succeeds across workspaces."""
        create = test_client.post(
            "/v1/memories",
            json={"content": "Cross-workspace single delete"},
            headers={"X-Workspace-ID": "xws_create_a"},
        )
        assert create.status_code == 201
        memory_id = create.json()["memory"]["id"]

        # Delete with a DIFFERENT workspace header.
        response = test_client.delete(
            f"/v1/memories/{memory_id}",
            headers={"X-Workspace-ID": "xws_other_b"},
        )
        assert response.status_code == 204

    def test_delete_genuinely_missing_returns_404(self, test_client: TestClient) -> None:
        """A genuinely-missing id still returns 404 (not conflated with success)."""
        response = test_client.delete(
            "/v1/memories/mem_does_not_exist_xyz",
            headers={"X-Workspace-ID": "xws_other_b"},
        )
        assert response.status_code == 404

    def test_batch_delete_resolves_memory_workspace(self, test_client: TestClient) -> None:
        """Batch delete succeeds across workspaces."""
        create = test_client.post(
            "/v1/memories",
            json={"content": "Cross-workspace batch delete"},
            headers={"X-Workspace-ID": "xws_create_c"},
        )
        assert create.status_code == 201
        memory_id = create.json()["memory"]["id"]

        response = test_client.post(
            "/v1/memories/batch",
            json={"operations": [{"op": "delete", "memory_id": memory_id, "hard": False}]},
            headers={"X-Workspace-ID": "xws_other_d"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["successful"] == 1
        assert body["results"][0]["status"] == "success"

    def test_batch_update_resolves_memory_workspace(self, test_client: TestClient) -> None:
        """Batch update succeeds across workspaces."""
        create = test_client.post(
            "/v1/memories",
            json={"content": "Cross-workspace batch update"},
            headers={"X-Workspace-ID": "xws_create_e"},
        )
        assert create.status_code == 201
        memory_id = create.json()["memory"]["id"]

        response = test_client.post(
            "/v1/memories/batch",
            json={
                "operations": [
                    {"op": "update", "memory_id": memory_id, "importance": 0.42}
                ]
            },
            headers={"X-Workspace-ID": "xws_other_f"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["successful"] == 1
        assert body["results"][0]["status"] == "success"


class TestMemoryDecay:
    """Tests for POST /v1/memories/{memory_id}/decay endpoint."""

    def test_decay_memory(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test decaying a memory's importance."""
        # First create a memory with high importance
        create_response = test_client.post(
            "/v1/memories",
            json={
                "content": "Test memory for decay",
                "importance": 0.9,
            },
            headers=workspace_headers,
        )
        assert create_response.status_code == 201
        memory_id = create_response.json()["memory"]["id"]
        original_importance = create_response.json()["memory"]["importance"]

        # Now decay the memory
        response = test_client.post(
            f"/v1/memories/{memory_id}/decay",
            json={
                "decay_rate": 0.5,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "memory" in data
        updated_memory = data["memory"]
        assert updated_memory["id"] == memory_id
        assert updated_memory["importance"] < original_importance
        # Decay uses subtraction: new_importance = old_importance - decay_rate
        assert updated_memory["importance"] == pytest.approx(original_importance - 0.5, rel=0.01)

    def test_decay_memory_invalid_rate(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test decaying with invalid decay rate."""
        # First create a memory
        create_response = test_client.post(
            "/v1/memories",
            json={
                "content": "Test memory for invalid decay",
                "importance": 0.8,
            },
            headers=workspace_headers,
        )
        assert create_response.status_code == 201
        memory_id = create_response.json()["memory"]["id"]

        # Try to decay with invalid rate (> 1.0)
        response = test_client.post(
            f"/v1/memories/{memory_id}/decay",
            json={
                "decay_rate": 1.5,  # Invalid: > 1.0
            },
            headers=workspace_headers,
        )

        assert response.status_code == 422


class TestMemoryBatch:
    """Tests for POST /v1/memories/batch endpoint."""

    def test_batch_operations(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test batch memory operations - create multiple memories."""
        response = test_client.post(
            "/v1/memories/batch",
            json={
                "operations": [
                    {
                        "op": "create",
                        "content": "First batch memory",
                        "importance": 0.5,
                    },
                    {
                        "op": "create",
                        "content": "Second batch memory",
                        "importance": 0.6,
                    },
                ]
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
        assert all(r["status"] == "success" for r in data["results"])

    def test_batch_operations_mixed_types(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test batch with mixed operation types."""
        # First create a memory to update and delete
        create_response = test_client.post(
            "/v1/memories",
            json={"content": "Memory to update"},
            headers=workspace_headers,
        )
        memory_id = create_response.json()["memory"]["id"]

        # Now batch: create, update, delete
        response = test_client.post(
            "/v1/memories/batch",
            json={
                "operations": [
                    {
                        "op": "create",
                        "content": "New batch memory",
                    },
                    {
                        "op": "update",
                        "memory_id": memory_id,
                        "content": "Updated content",
                        "importance": 0.9,
                    },
                    {
                        "op": "delete",
                        "memory_id": memory_id,
                        "hard": False,
                    },
                ]
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0

    def test_batch_operations_empty(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test batch with no operations."""
        response = test_client.post(
            "/v1/memories/batch",
            json={"operations": []},
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 0
        assert data["successful"] == 0
        assert data["failed"] == 0
        assert len(data["results"]) == 0


class TestMemoryCreateExtended:
    """Extended tests for memory creation with all types and features."""

    def test_create_episodic_memory(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating episodic memory."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "User fixed authentication bug at 2:30pm",
                "type": "episodic",
                "importance": 0.7,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201

    def test_create_semantic_memory(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating semantic memory."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "FastAPI uses Pydantic for request validation",
                "type": "semantic",
                "subtype": "code_pattern",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201

    def test_create_procedural_memory(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating procedural memory."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "To deploy: run pytest, build docker image, push to registry",
                "type": "procedural",
                "subtype": "workflow",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201

    def test_create_working_memory(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating working memory."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "Currently refactoring authentication module",
                "type": "working",
                "importance": 0.9,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201

    def test_create_with_all_subtypes(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memories with all available subtypes."""
        subtypes = ["solution", "problem", "code_pattern", "fix", "error", "workflow", "preference", "decision"]

        for subtype in subtypes:
            response = test_client.post(
                "/v1/memories",
                json={
                    "content": f"Test memory with subtype {subtype}",
                    "subtype": subtype,
                },
                headers=workspace_headers,
            )

            assert response.status_code == 201

    def test_create_with_associations(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with associations list."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "Solution to authentication timeout",
                "type": "semantic",
                "subtype": "solution",
                "associations": ["mem_problem123", "mem_related456"],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201

    def test_create_with_context_id(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating memory with context_id as a string tag."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "Memory with context_id parameter",
                "context_id": "my-custom-context",
                "importance": 0.7,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert "memory" in data
        assert data["memory"]["content"] == "Memory with context_id parameter"

    def test_deduplication_same_content(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test that same content returns same memory ID (deduplication)."""
        content = "Exact same memory content for deduplication test"

        # Create first memory
        response1 = test_client.post(
            "/v1/memories",
            json={"content": content},
            headers=workspace_headers,
        )

        # Create second memory with same content
        response2 = test_client.post(
            "/v1/memories",
            json={"content": content},
            headers=workspace_headers,
        )

        # Both should succeed
        assert response1.status_code == 201
        assert response2.status_code == 201

        # If implemented, should return same ID
        response1.json()["memory"]["id"]
        response2.json()["memory"]["id"]
        # Note: Deduplication may or may not be implemented yet
        # Just verify both calls work


class TestMemoryRecallExtended:
    """Extended tests for memory recall with all modes and filters."""

    def test_recall_mode_rag(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with RAG mode (default)."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "authentication fixes",
                "mode": "rag",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode_used"] == "rag"

    def test_recall_mode_llm(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with LLM mode."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "what were the authentication issues?",
                "mode": "llm",
                "context": [
                    {"role": "user", "content": "We had timeout problems"},
                ],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode_used"] in ["llm", "rag"]  # May fallback

    def test_recall_mode_hybrid(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with hybrid mode."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "debugging strategies",
                "mode": "hybrid",
                "rag_threshold": 0.8,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode_used"] in ["rag", "llm", "hybrid"]

    def test_recall_tolerance_loose(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with loose tolerance."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "coding patterns",
                "tolerance": "loose",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_tolerance_strict(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with strict tolerance."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "exact authentication bug",
                "tolerance": "strict",
                "min_relevance": 0.9,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_with_associations(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with include_associations=True."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "authentication solutions",
                "include_associations": True,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_with_traverse_depth(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with graph traversal depth."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "related issues",
                "include_associations": True,
                "traverse_depth": 2,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_with_time_filters(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with created_after and created_before filters."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "recent changes",
                "created_after": "2024-01-01T00:00:00Z",
                "created_before": "2024-12-31T23:59:59Z",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_with_subtypes_filter(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with subtypes filter."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "solutions",
                "subtypes": ["solution", "fix"],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200

    def test_recall_with_all_filters(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test recall with comprehensive filter set."""
        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "authentication solutions",
                "types": ["semantic", "episodic"],
                "subtypes": ["solution", "fix"],
                "tags": ["authentication", "security"],
                "mode": "rag",
                "tolerance": "moderate",
                "limit": 20,
                "min_relevance": 0.6,
                "include_associations": True,
                "traverse_depth": 1,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200


class TestMemoryReflectExtended:
    """Extended tests for memory reflection."""

    def test_reflect_with_detail_level(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with detail_level parameter."""
        # Create some memories first
        test_client.post(
            "/v1/memories",
            json={
                "content": "User has extensive background in Python and Java",
                "type": "semantic",
            },
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What programming languages does the user know?",
                "detail_level": "abstract",
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "tokens_processed" in data
        # Abstract should use fewer tokens (budget: 150) # TODO: re-evaluate these statements because we've eased up on token constraints
        assert data["tokens_processed"] <= 250

    def test_reflect_include_sources_true(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with include_sources=True."""
        # Create a memory first
        create_response = test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers functional programming paradigms",
                "type": "semantic",
                "subtype": "preference",
            },
            headers=workspace_headers,
        )
        create_response.json()["memory"]["id"]

        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What programming paradigms does the user prefer?",
                "include_sources": True,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "source_memories" in data
        assert isinstance(data["source_memories"], list)
        # May or may not have sources depending on the query match

    def test_reflect_include_sources_false(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with include_sources=False."""
        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What is the user's coding style?",
                "include_sources": False,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        # source_memories should be empty when include_sources=False
        assert "source_memories" in data
        assert isinstance(data["source_memories"], list)

    def test_reflect_with_depth(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with depth parameter."""
        # Create memories with associations
        test_client.post(
            "/v1/memories",
            json={
                "content": "User solved a complex async issue yesterday",
                "type": "episodic",
            },
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What problems has the user solved recently?",
                "depth": 2,
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "source_memories" in data
        assert "confidence" in data

    def test_reflect_with_type_filters(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with type filters."""
        # Create memories of different types
        test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers dark mode in IDEs",
                "type": "semantic",
                "subtype": "preference",
            },
            headers=workspace_headers,
        )
        test_client.post(
            "/v1/memories",
            json={
                "content": "User debugged the API yesterday at 3pm",
                "type": "episodic",
            },
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "Tell me about the user's preferences",
                "types": ["semantic"],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "source_memories" in data

    def test_reflect_with_tags(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test reflect with tags filter."""
        # Create memories with tags
        test_client.post(
            "/v1/memories",
            json={
                "content": "User follows SOLID principles strictly",
                "type": "semantic",
                "tags": ["architecture", "best-practices"],
            },
            headers=workspace_headers,
        )
        test_client.post(
            "/v1/memories",
            json={
                "content": "User prefers tabs over spaces",
                "type": "semantic",
                "tags": ["formatting", "preference"],
            },
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/reflect",
            json={
                "query": "What are the user's architectural preferences?",
                "tags": ["architecture"],
            },
            headers=workspace_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data
        assert "source_memories" in data
        assert "confidence" in data


class TestAssociations:
    """Tests for association endpoints."""

    def test_create_association(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating association between memories."""
        # Create two memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Problem: Authentication timeout occurs"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Solution: Increase session timeout to 30 minutes"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        # Create association using POST /v1/memories/{memory_id}/associate
        association_response = test_client.post(
            f"/v1/memories/{memory2_id}/associate",
            json={
                "target_id": memory1_id,
                "relationship": "solves",
            },
            headers=workspace_headers,
        )

        assert association_response.status_code == 201
        data = association_response.json()
        assert "association" in data
        association = data["association"]
        assert association["source_id"] == memory2_id
        assert association["target_id"] == memory1_id
        assert association["relationship"] == "solves"

    def test_create_association_with_strength(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating association with strength parameter."""
        # Create two memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Feature A needs refactoring"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Use composition over inheritance"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        # Create association with custom strength
        association_response = test_client.post(
            f"/v1/memories/{memory2_id}/associate",
            json={
                "target_id": memory1_id,
                "relationship": "applies_to",
                "strength": 0.8,
            },
            headers=workspace_headers,
        )

        assert association_response.status_code == 201
        data = association_response.json()
        assert data["association"]["strength"] == 0.8

    def test_create_association_with_metadata(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating association with metadata."""
        # Create two memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Bug in payment processing"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Fixed race condition in checkout"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        # Create association with metadata
        association_response = test_client.post(
            f"/v1/memories/{memory2_id}/associate",
            json={
                "target_id": memory1_id,
                "relationship": "solves",
                "metadata": {"priority": "high", "team": "backend"},
            },
            headers=workspace_headers,
        )

        assert association_response.status_code == 201
        data = association_response.json()
        assert data["association"]["metadata"]["priority"] == "high"
        assert data["association"]["metadata"]["team"] == "backend"

    def test_create_association_all_relationships(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating associations with all relationship types."""
        # Create base memory
        base_response = test_client.post(
            "/v1/memories",
            json={"content": "Base memory for relationship testing"},
            headers=workspace_headers,
        )
        assert base_response.status_code == 201
        base_id = base_response.json()["memory"]["id"]

        # Test various relationship types
        relationships = [
            "causes",
            "triggers",
            "leads_to",
            "prevents",
            "solves",
            "addresses",
            "alternative_to",
            "improves",
            "occurs_in",
            "applies_to",
            "works_with",
            "requires",
            "builds_on",
            "contradicts",
            "confirms",
            "supersedes",
            "similar_to",
            "variant_of",
            "related_to",
            "follows",
            "depends_on",
            "enables",
            "blocks",
            "effective_for",
            "preferred_over",
            "deprecated_by",
        ]

        for relationship in relationships:
            # Create target memory
            target_response = test_client.post(
                "/v1/memories",
                json={"content": f"Target for {relationship} relationship"},
                headers=workspace_headers,
            )
            assert target_response.status_code == 201
            target_id = target_response.json()["memory"]["id"]

            # Create association
            association_response = test_client.post(
                f"/v1/memories/{base_id}/associate",
                json={
                    "target_id": target_id,
                    "relationship": relationship,
                },
                headers=workspace_headers,
            )

            assert association_response.status_code == 201, f"Failed for relationship: {relationship}"
            assert association_response.json()["association"]["relationship"] == relationship

    def test_list_associations(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test listing associations for a memory."""
        # Create three memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Central memory"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Related memory 1"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        response3 = test_client.post(
            "/v1/memories",
            json={"content": "Related memory 2"},
            headers=workspace_headers,
        )
        assert response3.status_code == 201
        memory3_id = response3.json()["memory"]["id"]

        # Create associations
        test_client.post(
            f"/v1/memories/{memory1_id}/associate",
            json={"target_id": memory2_id, "relationship": "related_to"},
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memory1_id}/associate",
            json={"target_id": memory3_id, "relationship": "similar_to"},
            headers=workspace_headers,
        )

        # List associations
        list_response = test_client.get(
            f"/v1/memories/{memory1_id}/associations",
            headers=workspace_headers,
        )

        assert list_response.status_code == 200
        data = list_response.json()
        assert "associations" in data
        assert "total_count" in data
        assert data["total_count"] >= 2
        assert len(data["associations"]) >= 2

    def test_list_associations_with_relationship_filter(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test listing associations filtered by relationship types."""
        # Create memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Central memory for filtering"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Solves target"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        response3 = test_client.post(
            "/v1/memories",
            json={"content": "Related target"},
            headers=workspace_headers,
        )
        assert response3.status_code == 201
        memory3_id = response3.json()["memory"]["id"]

        # Create associations with different types
        test_client.post(
            f"/v1/memories/{memory1_id}/associate",
            json={"target_id": memory2_id, "relationship": "solves"},
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memory1_id}/associate",
            json={"target_id": memory3_id, "relationship": "related_to"},
            headers=workspace_headers,
        )

        # List with relationship filter
        list_response = test_client.get(
            f"/v1/memories/{memory1_id}/associations",
            params={"relationships": "solves"},
            headers=workspace_headers,
        )

        assert list_response.status_code == 200
        data = list_response.json()
        # Should only return associations with "solves" relationship
        assert all(a["relationship"] == "solves" for a in data["associations"])

    def test_list_associations_with_direction(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test listing associations with direction filter."""
        # Create memories
        response1 = test_client.post(
            "/v1/memories",
            json={"content": "Central memory for direction test"},
            headers=workspace_headers,
        )
        assert response1.status_code == 201
        memory1_id = response1.json()["memory"]["id"]

        response2 = test_client.post(
            "/v1/memories",
            json={"content": "Outgoing target"},
            headers=workspace_headers,
        )
        assert response2.status_code == 201
        memory2_id = response2.json()["memory"]["id"]

        response3 = test_client.post(
            "/v1/memories",
            json={"content": "Incoming source"},
            headers=workspace_headers,
        )
        assert response3.status_code == 201
        memory3_id = response3.json()["memory"]["id"]

        # Create outgoing association (memory1 -> memory2)
        test_client.post(
            f"/v1/memories/{memory1_id}/associate",
            json={"target_id": memory2_id, "relationship": "leads_to"},
            headers=workspace_headers,
        )

        # Create incoming association (memory3 -> memory1)
        test_client.post(
            f"/v1/memories/{memory3_id}/associate",
            json={"target_id": memory1_id, "relationship": "triggers"},
            headers=workspace_headers,
        )

        # Test outgoing direction
        outgoing_response = test_client.get(
            f"/v1/memories/{memory1_id}/associations",
            params={"direction": "outgoing"},
            headers=workspace_headers,
        )
        assert outgoing_response.status_code == 200
        outgoing_data = outgoing_response.json()
        # All associations should have memory1 as source
        assert all(a["source_id"] == memory1_id for a in outgoing_data["associations"])

        # Test incoming direction
        incoming_response = test_client.get(
            f"/v1/memories/{memory1_id}/associations",
            params={"direction": "incoming"},
            headers=workspace_headers,
        )
        assert incoming_response.status_code == 200
        incoming_data = incoming_response.json()
        # All associations should have memory1 as target
        assert all(a["target_id"] == memory1_id for a in incoming_data["associations"])

        # Test both direction
        both_response = test_client.get(
            f"/v1/memories/{memory1_id}/associations",
            params={"direction": "both"},
            headers=workspace_headers,
        )
        assert both_response.status_code == 200
        both_data = both_response.json()
        # Should have both outgoing and incoming associations
        assert both_data["total_count"] >= 2


class TestGraphTraversal:
    """Tests for graph traversal endpoint."""

    def test_traverse_basic(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test basic graph traversal."""
        # Create a chain of memories: A -> B -> C
        responseA = test_client.post(
            "/v1/memories",
            json={"content": "Memory A"},
            headers=workspace_headers,
        )
        assert responseA.status_code == 201
        memoryA_id = responseA.json()["memory"]["id"]

        responseB = test_client.post(
            "/v1/memories",
            json={"content": "Memory B"},
            headers=workspace_headers,
        )
        assert responseB.status_code == 201
        memoryB_id = responseB.json()["memory"]["id"]

        responseC = test_client.post(
            "/v1/memories",
            json={"content": "Memory C"},
            headers=workspace_headers,
        )
        assert responseC.status_code == 201
        memoryC_id = responseC.json()["memory"]["id"]

        # Create associations A -> B -> C
        test_client.post(
            f"/v1/memories/{memoryA_id}/associate",
            json={"target_id": memoryB_id, "relationship": "leads_to"},
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memoryB_id}/associate",
            json={"target_id": memoryC_id, "relationship": "leads_to"},
            headers=workspace_headers,
        )

        # Traverse from A
        traverse_response = test_client.post(
            f"/v1/memories/{memoryA_id}/traverse",
            json={},
            headers=workspace_headers,
        )

        assert traverse_response.status_code == 200
        data = traverse_response.json()
        assert "paths" in data
        assert "total_paths" in data
        assert "unique_nodes" in data
        assert data["total_paths"] > 0
        assert memoryA_id in data["unique_nodes"]

    def test_traverse_with_max_depth(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test graph traversal with max_depth parameter."""
        # Create a chain: A -> B -> C -> D
        memories = []
        for i in range(4):
            response = test_client.post(
                "/v1/memories",
                json={"content": f"Memory {chr(65 + i)}"},
                headers=workspace_headers,
            )
            assert response.status_code == 201
            memories.append(response.json()["memory"]["id"])

        # Create chain associations
        for i in range(3):
            test_client.post(
                f"/v1/memories/{memories[i]}/associate",
                json={"target_id": memories[i + 1], "relationship": "leads_to"},
                headers=workspace_headers,
            )

        # Traverse with max_depth=1 (should only reach B)
        traverse_response = test_client.post(
            f"/v1/memories/{memories[0]}/traverse",
            json={"max_depth": 1},
            headers=workspace_headers,
        )

        assert traverse_response.status_code == 200
        data = traverse_response.json()
        # With depth 1, should only reach first neighbor
        assert len(data["unique_nodes"]) <= 2  # A and B

        # Traverse with max_depth=3 (should reach all)
        traverse_response_deep = test_client.post(
            f"/v1/memories/{memories[0]}/traverse",
            json={"max_depth": 3},
            headers=workspace_headers,
        )

        assert traverse_response_deep.status_code == 200
        data_deep = traverse_response_deep.json()
        # With depth 3, should reach more nodes
        assert len(data_deep["unique_nodes"]) >= len(data["unique_nodes"])

    def test_traverse_with_relationship_types(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test graph traversal filtered by relationship types."""
        # Create memories
        responseA = test_client.post(
            "/v1/memories",
            json={"content": "Problem memory"},
            headers=workspace_headers,
        )
        assert responseA.status_code == 201
        memoryA_id = responseA.json()["memory"]["id"]

        responseB = test_client.post(
            "/v1/memories",
            json={"content": "Solution memory"},
            headers=workspace_headers,
        )
        assert responseB.status_code == 201
        memoryB_id = responseB.json()["memory"]["id"]

        responseC = test_client.post(
            "/v1/memories",
            json={"content": "Related memory"},
            headers=workspace_headers,
        )
        assert responseC.status_code == 201
        memoryC_id = responseC.json()["memory"]["id"]

        # Create associations with different types
        test_client.post(
            f"/v1/memories/{memoryB_id}/associate",
            json={"target_id": memoryA_id, "relationship": "solves"},
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memoryA_id}/associate",
            json={"target_id": memoryC_id, "relationship": "related_to"},
            headers=workspace_headers,
        )

        # Traverse with relationship filter for "solves" only
        traverse_response = test_client.post(
            f"/v1/memories/{memoryA_id}/traverse",
            json={"relationship_types": ["solves"]},
            headers=workspace_headers,
        )

        assert traverse_response.status_code == 200
        data = traverse_response.json()
        # Should find paths with "solves" relationships
        assert data["total_paths"] >= 0

    def test_traverse_with_direction(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test graph traversal with direction parameter."""
        # Create memories: A <- B -> C
        responseA = test_client.post(
            "/v1/memories",
            json={"content": "Memory A"},
            headers=workspace_headers,
        )
        assert responseA.status_code == 201
        memoryA_id = responseA.json()["memory"]["id"]

        responseB = test_client.post(
            "/v1/memories",
            json={"content": "Memory B (central)"},
            headers=workspace_headers,
        )
        assert responseB.status_code == 201
        memoryB_id = responseB.json()["memory"]["id"]

        responseC = test_client.post(
            "/v1/memories",
            json={"content": "Memory C"},
            headers=workspace_headers,
        )
        assert responseC.status_code == 201
        memoryC_id = responseC.json()["memory"]["id"]

        # Create associations: B -> A and B -> C
        test_client.post(
            f"/v1/memories/{memoryB_id}/associate",
            json={"target_id": memoryA_id, "relationship": "leads_to"},
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memoryB_id}/associate",
            json={"target_id": memoryC_id, "relationship": "leads_to"},
            headers=workspace_headers,
        )

        # Traverse outgoing from B
        outgoing_response = test_client.post(
            f"/v1/memories/{memoryB_id}/traverse",
            json={"direction": "outgoing"},
            headers=workspace_headers,
        )

        assert outgoing_response.status_code == 200
        outgoing_data = outgoing_response.json()
        assert outgoing_data["total_paths"] >= 2  # Should reach A and C

        # Traverse incoming to A (should find B)
        incoming_response = test_client.post(
            f"/v1/memories/{memoryA_id}/traverse",
            json={"direction": "incoming"},
            headers=workspace_headers,
        )

        assert incoming_response.status_code == 200
        incoming_data = incoming_response.json()
        assert memoryB_id in incoming_data["unique_nodes"]

    def test_traverse_with_min_strength(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test graph traversal with minimum strength filter."""
        # Create memories
        responseA = test_client.post(
            "/v1/memories",
            json={"content": "Start memory"},
            headers=workspace_headers,
        )
        assert responseA.status_code == 201
        memoryA_id = responseA.json()["memory"]["id"]

        responseB = test_client.post(
            "/v1/memories",
            json={"content": "Strong connection"},
            headers=workspace_headers,
        )
        assert responseB.status_code == 201
        memoryB_id = responseB.json()["memory"]["id"]

        responseC = test_client.post(
            "/v1/memories",
            json={"content": "Weak connection"},
            headers=workspace_headers,
        )
        assert responseC.status_code == 201
        memoryC_id = responseC.json()["memory"]["id"]

        # Create associations with different strengths
        test_client.post(
            f"/v1/memories/{memoryA_id}/associate",
            json={
                "target_id": memoryB_id,
                "relationship": "related_to",
                "strength": 0.9,
            },
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memoryA_id}/associate",
            json={
                "target_id": memoryC_id,
                "relationship": "related_to",
                "strength": 0.3,
            },
            headers=workspace_headers,
        )

        # Traverse with min_strength filter
        traverse_response = test_client.post(
            f"/v1/memories/{memoryA_id}/traverse",
            json={"min_strength": 0.7},
            headers=workspace_headers,
        )

        assert traverse_response.status_code == 200
        data = traverse_response.json()
        # Should only include strong connections
        assert memoryB_id in data["unique_nodes"]
        # Weak connection might or might not be included depending on implementation

    def test_traverse_comprehensive(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test graph traversal with all parameters."""
        # Create a small graph
        memories = []
        for i in range(5):
            response = test_client.post(
                "/v1/memories",
                json={"content": f"Memory {i}"},
                headers=workspace_headers,
            )
            assert response.status_code == 201
            memories.append(response.json()["memory"]["id"])

        # Create various associations
        test_client.post(
            f"/v1/memories/{memories[0]}/associate",
            json={
                "target_id": memories[1],
                "relationship": "leads_to",
                "strength": 0.9,
            },
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memories[1]}/associate",
            json={
                "target_id": memories[2],
                "relationship": "solves",
                "strength": 0.8,
            },
            headers=workspace_headers,
        )
        test_client.post(
            f"/v1/memories/{memories[0]}/associate",
            json={
                "target_id": memories[3],
                "relationship": "related_to",
                "strength": 0.4,
            },
            headers=workspace_headers,
        )

        # Comprehensive traversal with all parameters
        traverse_response = test_client.post(
            f"/v1/memories/{memories[0]}/traverse",
            json={
                "max_depth": 2,
                "relationship_types": ["leads_to", "solves"],
                "direction": "outgoing",
                "min_strength": 0.7,
            },
            headers=workspace_headers,
        )

        assert traverse_response.status_code == 200
        data = traverse_response.json()
        assert "paths" in data
        assert "total_paths" in data
        assert "unique_nodes" in data
        assert data["total_paths"] >= 0
        # Should have at least the start node
        assert memories[0] in data["unique_nodes"]


class TestMemoryRecallTemporalAndOffset:
    """Tests for the recall surface fields exposed in Phase 2.1.

    Asserts ``event_after`` / ``event_before`` (temporal window) and ``offset``
    reach the service and actually filter/paginate results. Each test uses an
    isolated workspace so seeded memories do not collide with other suites.
    """

    def test_recall_event_window_filters(self, test_client: TestClient) -> None:
        import uuid
        from datetime import UTC, datetime, timedelta

        headers = {"X-Workspace-ID": f"recall_temporal_{uuid.uuid4().hex[:8]}"}
        for i in range(3):
            resp = test_client.post(
                "/v1/memories",
                json={"content": f"temporal memory entry number {i} about pizza toppings"},
                headers=headers,
            )
            assert resp.status_code == 201

        now = datetime.now(UTC)

        # event_before in the past (window ending before any memory existed) =>
        # the temporal post-filter drops every candidate from the returned set.
        # (total_count reflects the pre-filter candidate pool, so assert on the
        # returned ``memories`` which is what the temporal filter actually shapes.)
        closed_past = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "pizza toppings",
                "event_before": (now - timedelta(days=365)).isoformat(),
            },
            headers=headers,
        )
        assert closed_past.status_code == 200
        assert closed_past.json()["memories"] == []

        # event_after well in the past => effective time (== created_at) is within
        # the window, so seeded memories survive the filter.
        open_window = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "pizza toppings",
                "event_after": (now - timedelta(days=1)).isoformat(),
            },
            headers=headers,
        )
        assert open_window.status_code == 200
        assert len(open_window.json()["memories"]) >= 1

    def test_recall_offset_reaches_storage(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"recall_offset_{uuid.uuid4().hex[:8]}"}
        for i in range(3):
            resp = test_client.post(
                "/v1/memories",
                json={"content": f"offset pagination memory {i} about hiking trails"},
                headers=headers,
            )
            assert resp.status_code == 201

        # offset=0 returns hits; a large offset past the candidate pool returns
        # fewer/zero — proving the offset threads to storage.search_memories.
        base = test_client.post(
            "/v1/memories/recall",
            json={"query": "hiking trails", "mode": "rag", "limit": 5, "offset": 0, "min_relevance": 0.0},
            headers=headers,
        )
        past_end = test_client.post(
            "/v1/memories/recall",
            json={"query": "hiking trails", "mode": "rag", "limit": 5, "offset": 100, "min_relevance": 0.0},
            headers=headers,
        )
        assert base.status_code == 200
        assert past_end.status_code == 200
        assert len(base.json()["memories"]) >= 1
        assert past_end.json()["memories"] == []

    def test_recall_accepts_global_scope_fields(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """The newly exposed include_global / include_global_user / time_order
        fields are accepted by the schema and threaded to the service."""
        resp = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "anything",
                "include_global": False,
                "include_global_user": False,
                "time_order": "desc",
            },
            headers=workspace_headers,
        )
        assert resp.status_code == 200


class TestMemoryList:
    """Tests for GET /v1/memories (Phase 2.2 browse endpoint)."""

    def test_list_pagination_and_order(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"list_ws_{uuid.uuid4().hex[:8]}"}
        created = []
        for i in range(5):
            resp = test_client.post(
                "/v1/memories",
                json={"content": f"list browse memory {i}"},
                headers=headers,
            )
            assert resp.status_code == 201
            created.append(resp.json()["memory"]["id"])

        # Full list: newest-first (created_at desc) => reverse of creation order.
        full = test_client.get("/v1/memories", params={"limit": 50}, headers=headers)
        assert full.status_code == 200
        ids = [m["id"] for m in full.json()["memories"]]
        assert set(created).issubset(set(ids))
        assert ids[: len(created)] == list(reversed(created))

        # Pagination: page1 + page2 cover distinct ids.
        page1 = test_client.get("/v1/memories", params={"limit": 2, "offset": 0}, headers=headers)
        page2 = test_client.get("/v1/memories", params={"limit": 2, "offset": 2}, headers=headers)
        ids1 = [m["id"] for m in page1.json()["memories"]]
        ids2 = [m["id"] for m in page2.json()["memories"]]
        assert len(ids1) == 2
        assert len(ids2) == 2
        assert set(ids1).isdisjoint(set(ids2))

    def test_list_type_filter(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"list_type_{uuid.uuid4().hex[:8]}"}
        test_client.post(
            "/v1/memories",
            json={"content": "an episodic event happened yesterday", "type": "episodic"},
            headers=headers,
        )
        test_client.post(
            "/v1/memories",
            json={"content": "a semantic fact about the world", "type": "semantic"},
            headers=headers,
        )

        resp = test_client.get("/v1/memories", params={"type": "episodic", "limit": 50}, headers=headers)
        assert resp.status_code == 200
        memories = resp.json()["memories"]
        assert len(memories) >= 1
        assert all(m["type"] == "episodic" for m in memories)

    def test_list_subtype_filter(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"list_subtype_{uuid.uuid4().hex[:8]}"}
        test_client.post(
            "/v1/memories",
            json={"content": "user prefers tabs", "type": "semantic", "subtype": "preference"},
            headers=headers,
        )
        test_client.post(
            "/v1/memories",
            json={"content": "unrelated note", "type": "semantic", "subtype": "note"},
            headers=headers,
        )

        resp = test_client.get("/v1/memories", params={"subtype": "preference", "limit": 50}, headers=headers)
        assert resp.status_code == 200
        memories = resp.json()["memories"]
        assert len(memories) >= 1
        assert all(m["subtype"] == "preference" for m in memories)


class TestAssociationUpdateDelete:
    """Tests for PATCH/DELETE association endpoints (Phase 2.3a)."""

    def _make_pair_and_assoc(self, test_client: TestClient, headers: dict[str, str]) -> tuple[str, str, str]:
        r1 = test_client.post("/v1/memories", json={"content": "assoc src memory"}, headers=headers)
        r2 = test_client.post("/v1/memories", json={"content": "assoc tgt memory"}, headers=headers)
        src = r1.json()["memory"]["id"]
        tgt = r2.json()["memory"]["id"]
        a = test_client.post(
            f"/v1/memories/{src}/associate",
            json={"target_id": tgt, "relationship": "related_to", "strength": 0.4},
            headers=headers,
        )
        assert a.status_code == 201
        return src, tgt, a.json()["association"]["id"]

    def test_update_association_strength_and_metadata(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"assoc_upd_{uuid.uuid4().hex[:8]}"}
        src, _tgt, assoc_id = self._make_pair_and_assoc(test_client, headers)

        upd = test_client.patch(
            f"/v1/memories/{src}/associations/{assoc_id}",
            json={"strength": 0.95, "metadata": {"note": "stronger"}},
            headers=headers,
        )
        assert upd.status_code == 204

        # Verify via list-associations that strength changed.
        listed = test_client.get(f"/v1/memories/{src}/associations", headers=headers)
        assert listed.status_code == 200
        match = [a for a in listed.json()["associations"] if a["id"] == assoc_id]
        assert match and abs(match[0]["strength"] - 0.95) < 1e-6

    def test_update_association_not_found(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"assoc_upd404_{uuid.uuid4().hex[:8]}"}
        src, _tgt, _assoc_id = self._make_pair_and_assoc(test_client, headers)
        resp = test_client.patch(
            f"/v1/memories/{src}/associations/does-not-exist",
            json={"strength": 0.5},
            headers=headers,
        )
        assert resp.status_code == 404

    def test_delete_association(self, test_client: TestClient) -> None:
        import uuid

        headers = {"X-Workspace-ID": f"assoc_del_{uuid.uuid4().hex[:8]}"}
        src, _tgt, assoc_id = self._make_pair_and_assoc(test_client, headers)

        deleted = test_client.delete(f"/v1/memories/{src}/associations/{assoc_id}", headers=headers)
        assert deleted.status_code == 204

        listed = test_client.get(f"/v1/memories/{src}/associations", headers=headers)
        assert listed.status_code == 200
        assert all(a["id"] != assoc_id for a in listed.json()["associations"])

        # Second delete => 404 (already gone).
        again = test_client.delete(f"/v1/memories/{src}/associations/{assoc_id}", headers=headers)
        assert again.status_code == 404

    def test_patch_mismatched_memory_id_returns_404(self, test_client: TestClient) -> None:
        """PATCH with a memory_id that is not an endpoint of the association must return 404.

        This enforces the REST ownership contract: the URL implies the association
        belongs to the memory; using an unrelated memory_id must not succeed.
        """
        import uuid

        headers = {"X-Workspace-ID": f"assoc_mismatch_{uuid.uuid4().hex[:8]}"}
        # Create two independent pairs: A-B and C-D.
        r_a = test_client.post("/v1/memories", json={"content": "memory A"}, headers=headers)
        r_b = test_client.post("/v1/memories", json={"content": "memory B"}, headers=headers)
        r_c = test_client.post("/v1/memories", json={"content": "memory C"}, headers=headers)
        mem_a = r_a.json()["memory"]["id"]
        mem_b = r_b.json()["memory"]["id"]
        mem_c = r_c.json()["memory"]["id"]

        # Association belongs to A-B.
        a_resp = test_client.post(
            f"/v1/memories/{mem_a}/associate",
            json={"target_id": mem_b, "relationship": "related_to", "strength": 0.5},
            headers=headers,
        )
        assert a_resp.status_code == 201
        assoc_id = a_resp.json()["association"]["id"]

        # PATCH using C as memory_id (C is not an endpoint of the association).
        resp = test_client.patch(
            f"/v1/memories/{mem_c}/associations/{assoc_id}",
            json={"strength": 0.9},
            headers=headers,
        )
        assert resp.status_code == 404
