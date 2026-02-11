"""Tests for contradiction API endpoints."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.contradiction.base import ContradictionRecord


@pytest_asyncio.fixture
async def async_client(fastapi_app, v):
    """Create async HTTP client for API testing with state.v set."""
    # ASGITransport does not trigger lifespan, so set state.v manually
    fastapi_app.state.v = v
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
class TestListContradictions:
    """Test GET /v1/workspaces/{workspace_id}/contradictions endpoint."""

    async def test_list_contradictions_empty(self, async_client):
        """Fresh workspace should return empty list."""
        response = await async_client.get(
            "/v1/workspaces/test_empty_contra_ws/contradictions",
            headers={"X-Workspace-ID": "test_empty_contra_ws"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "contradictions" in data
        assert "count" in data
        assert data["count"] == 0

    async def test_list_contradictions_with_data(
            self, async_client, storage_backend, workspace_id
    ):
        """Should return unresolved contradictions."""
        # Create real memories for FK constraints
        input_a = RememberInput(content="API test memory A", importance=0.5)
        input_b = RememberInput(content="API test memory B", importance=0.5)
        mem_a = await storage_backend.create_memory(workspace_id, input_a)
        mem_b = await storage_backend.create_memory(workspace_id, input_b)

        record = ContradictionRecord(
            workspace_id=workspace_id,
            memory_a_id=mem_a.id,
            memory_b_id=mem_b.id,
            contradiction_type="negation",
            confidence=0.85,
            detection_method="negation_pattern",
        )
        await storage_backend.create_contradiction(record)

        response = await async_client.get(
            f"/v1/workspaces/{workspace_id}/contradictions",
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

        found = any(c["id"] == record.id for c in data["contradictions"])
        assert found

    async def test_list_contradictions_with_limit(
            self, async_client, workspace_id
    ):
        """Limit parameter should be respected."""
        response = await async_client.get(
            f"/v1/workspaces/{workspace_id}/contradictions?limit=1",
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] <= 1


@pytest.mark.asyncio
class TestResolveContradiction:
    """Test POST /v1/contradictions/{contradiction_id}/resolve endpoint."""

    async def test_resolve_keep_both(
            self, async_client, storage_backend, workspace_id
    ):
        """Resolve a contradiction with keep_both strategy."""
        # Create real memories for FK constraints
        input_a = RememberInput(content="Resolve API test A", importance=0.5)
        input_b = RememberInput(content="Resolve API test B", importance=0.5)
        mem_a = await storage_backend.create_memory(workspace_id, input_a)
        mem_b = await storage_backend.create_memory(workspace_id, input_b)

        record = ContradictionRecord(
            workspace_id=workspace_id,
            memory_a_id=mem_a.id,
            memory_b_id=mem_b.id,
            contradiction_type="negation",
            confidence=0.9,
            detection_method="negation_pattern",
        )
        stored = await storage_backend.create_contradiction(record)

        response = await async_client.post(
            f"/v1/contradictions/{stored.id}/resolve?workspace_id={workspace_id}",
            json={"resolution": "keep_both"},
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == stored.id
        assert data["resolution"] == "keep_both"

    async def test_resolve_invalid_strategy(
            self, async_client, workspace_id
    ):
        """Invalid resolution strategy should return 400."""
        response = await async_client.post(
            "/v1/contradictions/contra_fake/resolve",
            json={"resolution": "invalid_strategy"},
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 400

    async def test_resolve_merge_without_content(
            self, async_client, workspace_id
    ):
        """Merge without merged_content should return 400."""
        response = await async_client.post(
            "/v1/contradictions/contra_fake/resolve",
            json={"resolution": "merge"},
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 400

    async def test_resolve_nonexistent_contradiction(
            self, async_client, workspace_id
    ):
        """Resolving nonexistent contradiction should return 404."""
        response = await async_client.post(
            "/v1/contradictions/contra_nonexistent_xxx/resolve",
            json={"resolution": "keep_a"},
            headers={"X-Workspace-ID": workspace_id},
        )
        assert response.status_code == 404
