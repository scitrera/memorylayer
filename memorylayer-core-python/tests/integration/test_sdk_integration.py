"""Integration tests for Python SDK against FastAPI server.

These tests verify the SDK client works correctly against the actual server API.
"""

import pytest
from fastapi.testclient import TestClient


class TestSDKMemoryOperations:
    """Tests for SDK memory operations against server using TestClient."""

    def test_sdk_remember_minimal(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style remember with minimal parameters."""
        response = test_client.post(
            "/v1/memories",
            json={"content": "SDK test: Python is great for AI"},
            headers=workspace_headers,
        )
        # Service should be wired up now
        assert response.status_code == 201
        data = response.json()
        assert "memory" in data
        assert data["memory"]["content"] == "SDK test: Python is great for AI"

    def test_sdk_remember_full(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style remember with all parameters."""
        response = test_client.post(
            "/v1/memories",
            json={
                "content": "SDK full test: Use async/await for I/O operations",
                "type": "semantic",
                "subtype": "preference",
                "importance": 0.9,
                "tags": ["python", "async", "best-practice"],
                "metadata": {"source": "sdk_test"},
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["memory"]["importance"] == 0.9
        assert "python" in data["memory"]["tags"]

    def test_sdk_recall_minimal(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style recall with minimal parameters."""
        # First create a memory
        test_client.post(
            "/v1/memories",
            json={"content": "SDK recall test: FastAPI is fast"},
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/recall",
            json={"query": "FastAPI"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data

    def test_sdk_recall_with_filters(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style recall with type and subtype filters."""
        # Create memory with specific type
        test_client.post(
            "/v1/memories",
            json={
                "content": "SDK filter test: Always validate input",
                "type": "semantic",
                "subtype": "preference",
                "tags": ["validation"],
            },
            headers=workspace_headers,
        )

        response = test_client.post(
            "/v1/memories/recall",
            json={
                "query": "validate input",
                "types": ["semantic"],
                "subtypes": ["preference"],
                "tags": ["validation"],
            },
            headers=workspace_headers,
        )
        assert response.status_code == 200

    def test_sdk_reflect(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style reflect for memory synthesis."""
        # Create some memories
        test_client.post(
            "/v1/memories",
            json={"content": "Python is great for AI"},
            headers=workspace_headers,
        )
        test_client.post(
            "/v1/memories",
            json={"content": "Use FastAPI for REST APIs"},
            headers=workspace_headers,
        )

        # Test reflection
        response = test_client.post(
            "/v1/memories/reflect",
            json={"query": "Python and APIs"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "reflection" in data

    def test_sdk_get_memory_nonexistent(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test getting non-existent memory returns 404."""
        response = test_client.get(
            "/v1/memories/nonexistent-id",
            headers=workspace_headers,
        )
        assert response.status_code == 404

    def test_sdk_forget(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test SDK-style forget (delete)."""
        # Create memory first
        create_response = test_client.post(
            "/v1/memories",
            json={"content": "SDK delete test: temporary memory"},
            headers=workspace_headers,
        )
        memory_id = create_response.json()["memory"]["id"]

        # Delete it
        response = test_client.delete(
            f"/v1/memories/{memory_id}",
            headers=workspace_headers,
        )
        assert response.status_code == 204


class TestSDKFullRoundTrip:
    """Full round-trip tests for SDK-like operations."""

    def test_full_roundtrip_remember_recall_forget(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test complete memory lifecycle."""
        # Remember
        create_response = test_client.post(
            "/v1/memories",
            json={
                "content": "SDK roundtrip: Use dependency injection for testability",
                "type": "semantic",
                "importance": 0.75,
                "tags": ["architecture", "testing"],
            },
            headers=workspace_headers,
        )
        assert create_response.status_code == 201
        memory_id = create_response.json()["memory"]["id"]

        # Recall
        recall_response = test_client.post(
            "/v1/memories/recall",
            json={"query": "dependency injection testing"},
            headers=workspace_headers,
        )
        assert recall_response.status_code == 200

        # Get specific memory
        get_response = test_client.get(
            f"/v1/memories/{memory_id}",
            headers=workspace_headers,
        )
        assert get_response.status_code == 200
        assert get_response.json()["memory"]["id"] == memory_id

        # Forget
        delete_response = test_client.delete(
            f"/v1/memories/{memory_id}",
            headers=workspace_headers,
        )
        assert delete_response.status_code == 204

        # Verify deleted
        verify_response = test_client.get(
            f"/v1/memories/{memory_id}",
            headers=workspace_headers,
        )
        assert verify_response.status_code == 404

    def test_full_roundtrip_with_reflection(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test full lifecycle including reflection."""
        # Create memories
        test_client.post(
            "/v1/memories",
            json={"content": "Use async/await for I/O"},
            headers=workspace_headers,
        )
        test_client.post(
            "/v1/memories",
            json={"content": "FastAPI supports async"},
            headers=workspace_headers,
        )

        # Test reflection
        response = test_client.post(
            "/v1/memories/reflect",
            json={"query": "async programming"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        assert "reflection" in response.json()


class TestSDKAssociations:
    """Tests for SDK association operations."""

    def test_sdk_associate_memories(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating associations between memories."""
        # Create two memories
        mem1_response = test_client.post(
            "/v1/memories",
            json={"content": "SDK assoc: Problem - slow database queries"},
            headers=workspace_headers,
        )
        mem1_id = mem1_response.json()["memory"]["id"]

        mem2_response = test_client.post(
            "/v1/memories",
            json={"content": "SDK assoc: Solution - add database indexes"},
            headers=workspace_headers,
        )
        mem2_id = mem2_response.json()["memory"]["id"]

        # Create association - endpoint doesn't exist yet
        assoc_response = test_client.post(
            f"/v1/memories/{mem1_id}/associations",
            json={
                "target_id": mem2_id,
                "relationship": "solves",
                "strength": 0.9,
            },
            headers=workspace_headers,
        )
        # Endpoint not implemented - expecting 404 (not found) or 405 (method not allowed)
        assert assoc_response.status_code in [404, 405]


class TestSDKSessions:
    """Tests for SDK session operations."""

    def test_sdk_create_session(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating a session."""
        response = test_client.post(
            "/v1/sessions",
            json={
                "session_id": "test_session",
                "ttl_seconds": 3600,
                "metadata": {"test": "data"},
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "session" in data
        assert data["session"]["id"] == "test_session"

    def test_sdk_session_context(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test setting session working memory (v2: /memory endpoint)."""
        import uuid
        session_id = f"test_session_ctx_{uuid.uuid4().hex[:8]}"

        # Create session first
        test_client.post(
            "/v1/sessions",
            json={"session_id": session_id, "ttl_seconds": 3600},
            headers=workspace_headers,
        )

        # Set working memory (v2 endpoint: /memory instead of /context)
        response = test_client.post(
            f"/v1/sessions/{session_id}/memory",
            json={"key": "user_preference", "value": "dark_mode"},
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["key"] == "user_preference"
        assert data["value"] == "dark_mode"


class TestSDKWorkspaces:
    """Tests for SDK workspace operations."""

    def test_sdk_create_workspace(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Test creating a workspace."""
        response = test_client.post(
            "/v1/workspaces",
            json={
                "name": "Test Workspace",
                "settings": {"retention_days": 90},
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "workspace" in data
        assert data["workspace"]["name"] == "Test Workspace"
