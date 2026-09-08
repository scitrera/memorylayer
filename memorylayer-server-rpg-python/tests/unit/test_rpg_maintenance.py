# SPDX-License-Identifier: Apache-2.0
"""Tests for RPG maintenance service and maintenance API endpoint.

Tests validate_graph, cleanup_stale_nodes, and compute_statistics operations
both at the service layer and through the POST /v1/rpg/maintenance/{operation} endpoint.
"""

import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from memorylayer_server.models.workspace import Workspace

from memorylayer_server_rpg.services.maintenance import RpgMaintenanceService
from memorylayer_server_rpg.services.service import RpgService

# ---------------------------------------------------------------------------
# Re-use sample data from the RPG service tests
# ---------------------------------------------------------------------------

SAMPLE_NODES = [
    {
        "node_id": "src",
        "node_type": "rpg_directory",
        "path": "src",
        "name": "src",
        "description": "Source root directory",
    },
    {
        "node_id": "src/main.py",
        "node_type": "rpg_file",
        "path": "src/main.py",
        "name": "main.py",
        "description": "Application entry point",
        "language": "python",
    },
    {
        "node_id": "src/models.py",
        "node_type": "rpg_file",
        "path": "src/models.py",
        "name": "models.py",
        "description": "Data model definitions",
        "language": "python",
    },
    {
        "node_id": "src/models.py:User",
        "node_type": "rpg_class",
        "path": "src/models.py",
        "name": "User",
        "description": "User data model with authentication fields",
        "language": "python",
        "parent_id": "src/models.py",
    },
    {
        "node_id": "src/models.py:User.validate",
        "node_type": "rpg_method",
        "path": "src/models.py",
        "name": "validate",
        "description": "Validate user data before saving",
        "language": "python",
        "parent_id": "src/models.py:User",
    },
    {
        "node_id": "src/main.py:main",
        "node_type": "rpg_function",
        "path": "src/main.py",
        "name": "main",
        "description": "Main application entry function",
        "language": "python",
        "parent_id": "src/main.py",
    },
]

SAMPLE_EDGES = [
    {"source_id": "src", "target_id": "src/main.py", "relationship": "contains"},
    {"source_id": "src", "target_id": "src/models.py", "relationship": "contains"},
    {"source_id": "src/models.py", "target_id": "src/models.py:User", "relationship": "contains"},
    {"source_id": "src/models.py:User", "target_id": "src/models.py:User.validate", "relationship": "contains"},
    {"source_id": "src/main.py", "target_id": "src/main.py:main", "relationship": "contains"},
    {"source_id": "src/main.py", "target_id": "src/models.py", "relationship": "imports"},
    {"source_id": "src/main.py:main", "target_id": "src/models.py:User.validate", "relationship": "invokes"},
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def maintenance_service(storage_backend) -> RpgMaintenanceService:
    """Create an RpgMaintenanceService backed by the test storage."""
    return RpgMaintenanceService(storage_backend)


@pytest_asyncio.fixture
async def rpg_service(storage_backend) -> RpgService:
    """Create an RpgService backed by the test storage."""
    return RpgService(storage_backend)


@pytest_asyncio.fixture
async def maint_workspace(storage_backend) -> str:
    """Create a unique workspace for each maintenance test to ensure isolation."""
    ws_id = f"maint_test_{uuid.uuid4().hex[:8]}"
    ws = Workspace(
        id=ws_id,
        tenant_id="test_tenant",
        name="Maintenance Test Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await storage_backend.create_workspace(ws)
    return ws_id


@pytest_asyncio.fixture
async def async_client(fastapi_app, v):
    """Create async HTTP client for API tests with state.v set."""
    fastapi_app.state.v = v
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Tests: validate_graph
# ---------------------------------------------------------------------------


class TestValidateGraph:
    """Tests for RpgMaintenanceService.validate_graph."""

    async def test_validate_healthy_graph_reports_healthy(self, maintenance_service, rpg_service, maint_workspace):
        """A fully-connected synced graph should report healthy with no dangling edges or orphans."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        report = await maintenance_service.validate_graph(maint_workspace)

        assert report["healthy"] is True
        assert report["dangling_edges"] == 0
        assert report["orphan_nodes"] == 0
        assert report["node_count"] == len(SAMPLE_NODES)
        assert report["workspace_id"] == maint_workspace

    async def test_validate_empty_workspace_reports_healthy(self, maintenance_service, maint_workspace):
        """An empty workspace has nothing wrong — should also be reported healthy."""
        report = await maintenance_service.validate_graph(maint_workspace)

        assert report["healthy"] is True
        assert report["node_count"] == 0
        assert report["edge_count"] == 0
        assert report["dangling_edges"] == 0
        assert report["orphan_nodes"] == 0

    async def test_validate_detects_orphan_nodes(self, maintenance_service, rpg_service, maint_workspace):
        """Nodes that have no outgoing or incoming edges are detected as orphans."""
        # Sync nodes without any edges — all nodes are orphans
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=[],
        )

        report = await maintenance_service.validate_graph(maint_workspace)

        assert report["orphan_nodes"] > 0
        assert report["healthy"] is False

    async def test_validate_reports_correct_node_count(self, maintenance_service, rpg_service, maint_workspace):
        """Node count in validation report matches the number of public RPG nodes."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES[:3],
            edges=[],
        )

        report = await maintenance_service.validate_graph(maint_workspace)

        assert report["node_count"] == 3

    async def test_validate_result_contains_required_keys(self, maintenance_service, maint_workspace):
        """Validation result dict always contains all expected keys."""
        report = await maintenance_service.validate_graph(maint_workspace)

        required_keys = {
            "workspace_id",
            "node_count",
            "edge_count",
            "dangling_edges",
            "orphan_nodes",
            "orphan_node_ids",
            "healthy",
        }
        assert required_keys.issubset(report.keys())


# ---------------------------------------------------------------------------
# Tests: cleanup_stale_nodes
# ---------------------------------------------------------------------------


class TestCleanupStaleNodes:
    """Tests for RpgMaintenanceService.cleanup_stale_nodes."""

    async def test_cleanup_removes_orphan_nodes(self, maintenance_service, rpg_service, maint_workspace):
        """Nodes with no edges are deleted during cleanup."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=[],  # no edges → all orphans
        )

        result = await maintenance_service.cleanup_stale_nodes(maint_workspace)

        # All sample nodes are deleted; internal sync metadata is not counted.
        assert result["deleted_count"] == len(SAMPLE_NODES)
        assert result["checked_count"] == len(SAMPLE_NODES)

    async def test_cleanup_keeps_nodes_with_edges(self, maintenance_service, rpg_service, maint_workspace):
        """Nodes that participate in at least one edge are preserved."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        result = await maintenance_service.cleanup_stale_nodes(maint_workspace)

        assert result["deleted_count"] == 0
        assert result["checked_count"] == len(SAMPLE_NODES)

    async def test_cleanup_empty_workspace_deletes_nothing(self, maintenance_service, maint_workspace):
        """Cleanup on an empty workspace reports zero deletions."""
        result = await maintenance_service.cleanup_stale_nodes(maint_workspace)

        assert result["deleted_count"] == 0
        assert result["checked_count"] == 0

    async def test_cleanup_result_contains_required_keys(self, maintenance_service, maint_workspace):
        """Cleanup result dict always contains workspace_id, checked_count, deleted_count."""
        result = await maintenance_service.cleanup_stale_nodes(maint_workspace)

        assert "workspace_id" in result
        assert "checked_count" in result
        assert "deleted_count" in result
        assert result["workspace_id"] == maint_workspace

    async def test_cleanup_after_cleanup_leaves_no_orphans(self, maintenance_service, rpg_service, maint_workspace):
        """Validating after cleanup shows no remaining orphans."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=[],
        )

        await maintenance_service.cleanup_stale_nodes(maint_workspace)
        report = await maintenance_service.validate_graph(maint_workspace)

        assert report["orphan_nodes"] == 0
        assert report["node_count"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    """Tests for RpgMaintenanceService.compute_statistics."""

    async def test_statistics_correct_total_node_count(self, maintenance_service, rpg_service, maint_workspace):
        """Total node count matches the number of synced nodes."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        stats = await maintenance_service.compute_statistics(maint_workspace)

        assert stats["total_nodes"] == len(SAMPLE_NODES)

    async def test_statistics_correct_total_edge_count(self, maintenance_service, rpg_service, maint_workspace):
        """Total edge count matches the number of synced edges."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        stats = await maintenance_service.compute_statistics(maint_workspace)

        assert stats["total_edges"] == len(SAMPLE_EDGES)

    async def test_statistics_node_type_counts_match_sample_data(self, maintenance_service, rpg_service, maint_workspace):
        """Node type counts are broken down correctly per type."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        stats = await maintenance_service.compute_statistics(maint_workspace)
        type_counts = stats["node_type_counts"]

        # Verify known type counts from SAMPLE_NODES
        assert type_counts.get("rpg_directory", 0) == 1
        assert type_counts.get("rpg_file", 0) == 2
        assert type_counts.get("rpg_class", 0) == 1
        assert type_counts.get("rpg_method", 0) == 1
        assert type_counts.get("rpg_function", 0) == 1

    async def test_statistics_edge_type_counts_non_empty_after_sync(self, maintenance_service, rpg_service, maint_workspace):
        """Edge type counts dict is non-empty after syncing edges."""
        await rpg_service.sync(
            workspace_id=maint_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

        stats = await maintenance_service.compute_statistics(maint_workspace)

        assert len(stats["edge_type_counts"]) > 0

    async def test_statistics_empty_workspace_returns_zeros(self, maintenance_service, maint_workspace):
        """Statistics for empty workspace shows zeros and empty dicts."""
        stats = await maintenance_service.compute_statistics(maint_workspace)

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["node_type_counts"] == {}
        assert stats["edge_type_counts"] == {}

    async def test_statistics_result_contains_required_keys(self, maintenance_service, maint_workspace):
        """Statistics result always contains all expected keys."""
        stats = await maintenance_service.compute_statistics(maint_workspace)

        required_keys = {
            "workspace_id",
            "total_nodes",
            "total_edges",
            "node_type_counts",
            "edge_type_counts",
        }
        assert required_keys.issubset(stats.keys())


# ---------------------------------------------------------------------------
# Tests: POST /v1/rpg/maintenance/{operation} API endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRpgMaintenanceAPI:
    """Test the RPG maintenance API endpoint for all three operations."""

    async def test_validate_operation_returns_200(self, async_client, maint_workspace):
        """POST /v1/rpg/maintenance/validate returns HTTP 200."""
        response = await async_client.post(
            "/v1/rpg/maintenance/validate",
            headers={"X-Workspace-ID": maint_workspace},
        )
        assert response.status_code == 200

    async def test_validate_operation_response_shape(self, async_client, maint_workspace):
        """validate response contains operation, workspace_id, and result fields."""
        response = await async_client.post(
            "/v1/rpg/maintenance/validate",
            headers={"X-Workspace-ID": maint_workspace},
        )
        data = response.json()

        assert data["operation"] == "validate"
        assert data["workspace_id"] == maint_workspace
        assert "result" in data
        assert "healthy" in data["result"]

    async def test_cleanup_operation_returns_200(self, async_client, maint_workspace):
        """POST /v1/rpg/maintenance/cleanup returns HTTP 200."""
        response = await async_client.post(
            "/v1/rpg/maintenance/cleanup",
            headers={"X-Workspace-ID": maint_workspace},
        )
        assert response.status_code == 200

    async def test_cleanup_operation_response_shape(self, async_client, maint_workspace):
        """cleanup response contains operation, workspace_id, and result with counts."""
        response = await async_client.post(
            "/v1/rpg/maintenance/cleanup",
            headers={"X-Workspace-ID": maint_workspace},
        )
        data = response.json()

        assert data["operation"] == "cleanup"
        assert data["workspace_id"] == maint_workspace
        assert "result" in data
        assert "deleted_count" in data["result"]
        assert "checked_count" in data["result"]

    async def test_statistics_operation_returns_200(self, async_client, maint_workspace):
        """POST /v1/rpg/maintenance/statistics returns HTTP 200."""
        response = await async_client.post(
            "/v1/rpg/maintenance/statistics",
            headers={"X-Workspace-ID": maint_workspace},
        )
        assert response.status_code == 200

    async def test_statistics_operation_response_shape(self, async_client, maint_workspace):
        """statistics response contains operation, workspace_id, and result with counts."""
        response = await async_client.post(
            "/v1/rpg/maintenance/statistics",
            headers={"X-Workspace-ID": maint_workspace},
        )
        data = response.json()

        assert data["operation"] == "statistics"
        assert data["workspace_id"] == maint_workspace
        assert "result" in data
        assert "total_nodes" in data["result"]
        assert "total_edges" in data["result"]

    async def test_invalid_operation_returns_400(self, async_client, maint_workspace):
        """An unrecognized operation name returns HTTP 400."""
        response = await async_client.post(
            "/v1/rpg/maintenance/invalid_op",
            headers={"X-Workspace-ID": maint_workspace},
        )
        assert response.status_code == 400
