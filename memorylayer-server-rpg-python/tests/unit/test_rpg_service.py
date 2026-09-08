# SPDX-License-Identifier: Apache-2.0
"""Tests for RPG (Repository Planning Graph) service.

Tests the full RPG lifecycle: sync nodes/edges, query subgraphs, search nodes, status.
"""

import pytest_asyncio
from memorylayer_server.models.association import KNOWN_RELATIONSHIP_TYPES

from memorylayer_server_rpg.api.schemas import RpgMergedSubgraphResponse, RpgSubgraphResponse
from memorylayer_server_rpg.services.ontology_contributor import (
    RPG_RELATIONSHIP_TYPES as RPG_REL_TYPES_DICT,
)
from memorylayer_server_rpg.services.ontology_contributor import (
    RPG_SUBTYPES as RPG_SUBTYPES_DICT,
)
from memorylayer_server_rpg.services.service import RPG_RELATIONSHIP_TYPES, RPG_SUBTYPES, RpgService

# ---------------------------------------------------------------------------
# Model / Ontology tests
# ---------------------------------------------------------------------------


class TestRpgModels:
    """Verify RPG types are correctly registered in the ontology."""

    def test_rpg_subtypes_exist(self):
        """All RPG subtypes are declared by the RPG ontology contributor."""
        expected = {
            "rpg_directory",
            "rpg_file",
            "rpg_class",
            "rpg_function",
            "rpg_method",
            "rpg_component",
            "rpg_module",
            "rpg_package",
            "rpg_interface",
        }
        actual = {s for subtypes in RPG_SUBTYPES_DICT.values() for s in subtypes}
        assert actual == expected

    def test_rpg_subtypes_match_service_constant(self):
        """RPG_SUBTYPES service constant is in sync with the contributor dict."""
        contributor_rpg = {s for subtypes in RPG_SUBTYPES_DICT.values() for s in subtypes}
        assert RPG_SUBTYPES == contributor_rpg

    def test_code_structure_category_exists(self):
        """code_structure relationship category is registered in RPG contributor."""
        # Verify the RPG contributor declares code_structure types
        categories = {v["category"] for v in RPG_REL_TYPES_DICT.values()}
        assert "code_structure" in categories

    def test_rpg_relationship_types_in_ontology(self):
        """All RPG relationship types are in the local contributor dict with correct category."""
        for rel in RPG_RELATIONSHIP_TYPES:
            assert rel in RPG_REL_TYPES_DICT, f"'{rel}' missing from RPG_RELATIONSHIP_TYPES dict"
            assert RPG_REL_TYPES_DICT[rel]["category"] == "code_structure"

    def test_rpg_relationship_types_in_contributor_dict(self):
        """All RPG relationship types are in the RPG contributor dict.

        RPG types are not in the OSS KNOWN_RELATIONSHIP_TYPES constant. They are
        contributed at startup via RpgOntologyContributorPlugin, making this
        contributor dict the authoritative source.
        """
        for rel in RPG_RELATIONSHIP_TYPES:
            assert rel in RPG_REL_TYPES_DICT, f"'{rel}' missing from RPG contributor dict"
        # Confirm RPG types are NOT in the OSS static constant (expected post-step-4)
        for rel in RPG_RELATIONSHIP_TYPES:
            assert rel not in KNOWN_RELATIONSHIP_TYPES, f"'{rel}' should be owned by the RPG ontology contributor"

    def test_rpg_relationships_have_inverses(self):
        """All RPG relationships have proper inverse mappings in the local contributor dict."""
        for rel in RPG_RELATIONSHIP_TYPES:
            info = RPG_REL_TYPES_DICT[rel]
            inverse = info.get("inverse")
            assert inverse is not None, f"'{rel}' has no inverse"
            assert inverse in RPG_REL_TYPES_DICT, f"'{rel}' inverse '{inverse}' not in contributor dict"
            assert RPG_REL_TYPES_DICT[inverse]["inverse"] == rel

    def test_subgraph_schemas_preserve_truncation(self):
        """FastAPI response validation must not discard the truncation signal."""
        assert RpgSubgraphResponse(truncated=True).model_dump()["truncated"] is True
        assert RpgMergedSubgraphResponse(truncated=True).model_dump()["truncated"] is True


# ---------------------------------------------------------------------------
# RPG Service integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rpg_service(storage_backend) -> RpgService:
    """Create an RPG service instance backed by the test storage."""
    return RpgService(storage_backend)


@pytest_asyncio.fixture
async def rpg_workspace(storage_backend) -> str:
    """Create a unique workspace for RPG tests."""
    import uuid
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace

    ws_id = f"rpg_test_{uuid.uuid4().hex[:8]}"
    ws = Workspace(
        id=ws_id,
        tenant_id="test_tenant",
        name="RPG Test Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await storage_backend.create_workspace(ws)
    return ws_id


# Sample RPG graph: a small Python project structure
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


class TestRpgSync:
    """Test RPG graph sync operations."""

    async def test_sync_creates_nodes(self, rpg_service, rpg_workspace):
        """Syncing nodes creates memory entities."""
        result = await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES,
            edges=[],
        )
        assert result["nodes_created"] == len(SAMPLE_NODES)
        assert result["nodes_updated"] == 0
        assert result["sync_time_ms"] >= 0

    async def test_sync_creates_edges(self, rpg_service, rpg_workspace):
        """Syncing edges creates association entities."""
        # Sync nodes first
        await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES,
            edges=[],
        )
        # Now sync edges
        result = await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=[],
            edges=SAMPLE_EDGES,
        )
        assert result["edges_created"] == len(SAMPLE_EDGES)

    async def test_sync_upsert_updates_existing(self, rpg_service, rpg_workspace):
        """Re-syncing same nodes updates rather than duplicates."""
        # First sync
        r1 = await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES[:2],
            edges=[],
        )
        assert r1["nodes_created"] == 2
        assert r1["nodes_updated"] == 0

        # Second sync with same nodes (updated description)
        updated_nodes = [
            {**SAMPLE_NODES[0], "description": "Updated source directory"},
            {**SAMPLE_NODES[1], "description": "Updated main entry point"},
        ]
        r2 = await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=updated_nodes,
            edges=[],
        )
        assert r2["nodes_created"] == 0
        assert r2["nodes_updated"] == 2

    async def test_sync_with_commit_sha(self, rpg_service, rpg_workspace):
        """Sync records source commit SHA."""
        result = await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES[:1],
            edges=[],
            source_commit="abc123def456",
        )
        assert result["source_commit"] == "abc123def456"

    async def test_sync_auto_creates_context_per_workspace(self, rpg_service, storage_backend):
        """Auto-create works for multiple workspaces with the same context_id.

        Regression: contexts.id is a global PRIMARY KEY, but the RPG context_id
        ("rpg") is a workspace-scoped logical handle. Naively using context_id
        as the row id collides on the second workspace. Ensure both syncs succeed
        and produce one context row per workspace.
        """
        import uuid
        from datetime import UTC, datetime

        from memorylayer_server.models.workspace import Workspace

        ws_ids = []
        for _ in range(2):
            ws_id = f"rpg_multi_{uuid.uuid4().hex[:8]}"
            await storage_backend.create_workspace(
                Workspace(
                    id=ws_id,
                    tenant_id="test_tenant",
                    name="RPG Multi Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            ws_ids.append(ws_id)

        # Both syncs must succeed (previously the second 500'd on UNIQUE(contexts.id))
        for ws_id in ws_ids:
            result = await rpg_service.sync(
                workspace_id=ws_id,
                nodes=SAMPLE_NODES[:1],
                edges=[],
            )
            assert result["nodes_created"] == 1

        # Each workspace has its own "rpg" context row, distinct ids
        ctx_a = await storage_backend.list_contexts(ws_ids[0])
        ctx_b = await storage_backend.list_contexts(ws_ids[1])
        rpg_a = [c for c in ctx_a if c.name == "rpg"]
        rpg_b = [c for c in ctx_b if c.name == "rpg"]
        assert len(rpg_a) == 1
        assert len(rpg_b) == 1
        assert rpg_a[0].id != rpg_b[0].id

        # Idempotency: re-sync on the first workspace must not create another row
        await rpg_service.sync(workspace_id=ws_ids[0], nodes=SAMPLE_NODES[:1], edges=[])
        ctx_a_again = await storage_backend.list_contexts(ws_ids[0])
        assert len([c for c in ctx_a_again if c.name == "rpg"]) == 1

    async def test_context_creation_recovers_from_concurrent_insert(self):
        """A uniqueness race resolves to the deterministic context row."""
        from memorylayer_server.models.workspace import Context

        row = Context(
            id="ws_1:rpg",
            workspace_id="ws_1",
            name="rpg",
            settings={"rpg": True},
        )

        class RacingStorage:
            async def get_context(self, workspace_id, context_id):
                return row if context_id == row.id else None

            async def list_contexts(self, workspace_id):
                return []

            async def create_context(self, workspace_id, context):
                raise RuntimeError("duplicate key")

        service = RpgService(RacingStorage())

        assert await service._resolve_context_row_id("ws_1", "rpg", create=True) == row.id


class TestRpgSubgraph:
    """Test RPG subgraph extraction."""

    @pytest_asyncio.fixture(autouse=True)
    async def _seed_graph(self, rpg_service, rpg_workspace):
        """Seed the workspace with sample RPG data before each test."""
        await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
        )

    async def test_subgraph_from_path(self, rpg_service, rpg_workspace):
        """Query subgraph from a file path."""
        result = await rpg_service.get_subgraph(
            workspace_id=rpg_workspace,
            root_path="src/main.py",
            depth=2,
        )
        assert result["total_nodes"] > 0
        node_ids = {n["node_id"] for n in result["nodes"]}
        assert "src/main.py" in node_ids

    async def test_subgraph_from_node_id(self, rpg_service, rpg_workspace):
        """Query subgraph from a direct node ID."""
        result = await rpg_service.get_subgraph(
            workspace_id=rpg_workspace,
            root_node_id="src",
            depth=1,
        )
        assert result["total_nodes"] > 0
        assert result["root_id"] == "src"

    async def test_subgraph_empty_for_unknown_path(self, rpg_service, rpg_workspace):
        """Unknown path returns empty subgraph."""
        result = await rpg_service.get_subgraph(
            workspace_id=rpg_workspace,
            root_path="nonexistent/path.py",
            depth=2,
        )
        assert result["total_nodes"] == 0
        assert result["total_edges"] == 0

    async def test_subgraph_with_node_type_filter(self, rpg_service, rpg_workspace):
        """Subgraph respects node_type filtering."""
        result = await rpg_service.get_subgraph(
            workspace_id=rpg_workspace,
            root_node_id="src",
            depth=5,
            node_types=["rpg_file"],
        )
        for node in result["nodes"]:
            assert node["node_type"] == "rpg_file"

    async def test_merged_subgraph_propagates_truncation(self, rpg_service, rpg_workspace, monkeypatch):
        """A capped base or overlay graph marks the merged response as truncated."""
        responses = iter(
            [
                {"nodes": [], "edges": [], "root_id": None, "truncated": False},
                {"nodes": [], "edges": [], "root_id": None, "truncated": True},
            ]
        )

        async def fake_subgraph(*args, **kwargs):
            return next(responses)

        monkeypatch.setattr(rpg_service, "get_subgraph", fake_subgraph)
        result = await rpg_service.get_merged_subgraph(
            workspace_id=rpg_workspace,
            overlay_context="rpg-task-test",
        )

        assert result["truncated"] is True


class TestRpgSearch:
    """Test RPG node search."""

    @pytest_asyncio.fixture(autouse=True)
    async def _seed_graph(self, rpg_service, rpg_workspace):
        """Seed the workspace with sample RPG data."""
        await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES,
            edges=[],
        )

    async def test_search_by_name(self, rpg_service, rpg_workspace):
        """Search finds nodes by name."""
        result = await rpg_service.search_nodes(
            workspace_id=rpg_workspace,
            query="User",
            limit=10,
        )
        assert result["total_count"] > 0
        names = {n["name"] for n in result["nodes"]}
        assert "User" in names

    async def test_search_by_description(self, rpg_service, rpg_workspace):
        """Search finds nodes by description content."""
        result = await rpg_service.search_nodes(
            workspace_id=rpg_workspace,
            query="authentication",
            limit=10,
        )
        assert result["total_count"] > 0

    async def test_search_with_type_filter(self, rpg_service, rpg_workspace):
        """Search respects node type filter."""
        result = await rpg_service.search_nodes(
            workspace_id=rpg_workspace,
            query="User",
            node_types=["rpg_class"],
            limit=10,
        )
        for node in result["nodes"]:
            assert node["node_type"] == "rpg_class"

    async def test_search_returns_empty_for_no_match(self, rpg_service, rpg_workspace):
        """Search returns empty for non-matching query."""
        result = await rpg_service.search_nodes(
            workspace_id=rpg_workspace,
            query="xyznonexistent123",
            limit=10,
        )
        assert result["total_count"] == 0


class TestRpgStatus:
    """Test RPG status reporting."""

    async def test_status_empty_workspace(self, rpg_service, rpg_workspace):
        """Status for empty workspace shows no RPG data."""
        result = await rpg_service.get_status(workspace_id=rpg_workspace)
        assert result["workspace_id"] == rpg_workspace
        assert result["has_rpg"] is False
        assert result["node_count"] == 0

    async def test_status_after_sync(self, rpg_service, rpg_workspace):
        """Status reflects synced data."""
        await rpg_service.sync(
            workspace_id=rpg_workspace,
            nodes=SAMPLE_NODES,
            edges=SAMPLE_EDGES,
            source_commit="test123",
        )

        result = await rpg_service.get_status(workspace_id=rpg_workspace)
        assert result["has_rpg"] is True
        assert result["node_count"] >= len(SAMPLE_NODES)
        assert result["edge_count"] >= len(SAMPLE_EDGES)
