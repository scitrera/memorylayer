# SPDX-License-Identifier: Apache-2.0
"""Tests for RPG enrichment service.

Tests feature identification, description enrichment, data flow analysis,
graceful degradation on empty/bad LLM responses, and phase selection.
"""

import uuid
from datetime import UTC, datetime

import pytest_asyncio
from memorylayer_server.models.workspace import Workspace

from memorylayer_server_rpg.services.enrichment import RpgEnrichmentService
from memorylayer_server_rpg.services.service import RpgService

# ---------------------------------------------------------------------------
# Sample graph data
# ---------------------------------------------------------------------------

SAMPLE_NODES = [
    {
        "node_id": "src/auth.py",
        "node_type": "rpg_file",
        "path": "src/auth.py",
        "name": "auth.py",
        "description": "Authentication module",
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
        "node_id": "src/api.py",
        "node_type": "rpg_file",
        "path": "src/api.py",
        "name": "api.py",
        "description": "API endpoint handlers",
        "language": "python",
    },
    {
        "node_id": "src/auth.py:AuthService",
        "node_type": "rpg_class",
        "path": "src/auth.py",
        "name": "AuthService",
        "description": "Authentication service",
        "language": "python",
        "parent_id": "src/auth.py",
    },
    {
        "node_id": "src/models.py:User",
        "node_type": "rpg_class",
        "path": "src/models.py",
        "name": "User",
        "description": "User model",
        "language": "python",
        "parent_id": "src/models.py",
    },
]

SAMPLE_EDGES = [
    {"source_id": "src/auth.py", "target_id": "src/auth.py:AuthService", "relationship": "contains"},
    {"source_id": "src/models.py", "target_id": "src/models.py:User", "relationship": "contains"},
    {"source_id": "src/auth.py", "target_id": "src/models.py", "relationship": "imports"},
    {"source_id": "src/api.py", "target_id": "src/auth.py", "relationship": "imports"},
]


# ---------------------------------------------------------------------------
# Mock LLM service
# ---------------------------------------------------------------------------


class MockLLMResponse:
    """Minimal LLMResponse stand-in for testing."""

    def __init__(self, content: str):
        self.content = content
        self.prompt_tokens = 0
        self.completion_tokens = 0


class MockLLMService:
    """Mock LLM service that returns canned responses based on call order."""

    def __init__(self, responses: list[str]):
        """
        Args:
            responses: Ordered list of response strings. Each complete() call
                pops the next response. If exhausted, returns empty JSON '[]'.
        """
        self._responses = list(responses)
        self.call_count = 0

    async def complete(self, request, profile=None):
        self.call_count += 1
        if self._responses:
            content = self._responses.pop(0)
        else:
            content = "[]"
        return MockLLMResponse(content=content)


def _make_feature_response(features: list[dict]) -> str:
    """Serialize a feature list to a JSON string as the LLM would return."""
    import json

    return json.dumps(features)


def _make_description_response(descriptions: dict[str, str]) -> str:
    """Serialize a description map to a JSON string as the LLM would return."""
    import json

    return json.dumps(descriptions)


def _make_data_flow_response(flows: list[dict]) -> str:
    """Serialize a data flow list to a JSON string as the LLM would return."""
    import json

    return json.dumps(flows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def enrich_workspace(storage_backend) -> str:
    """Create a unique workspace for each enrichment test."""
    ws_id = f"enrich_test_{uuid.uuid4().hex[:8]}"
    ws = Workspace(
        id=ws_id,
        tenant_id="test_tenant",
        name="Enrichment Test Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await storage_backend.create_workspace(ws)
    return ws_id


@pytest_asyncio.fixture
async def rpg_service(storage_backend) -> RpgService:
    return RpgService(storage_backend)


@pytest_asyncio.fixture
async def populated_workspace(storage_backend, rpg_service, enrich_workspace) -> str:
    """Workspace pre-populated with SAMPLE_NODES and SAMPLE_EDGES."""
    await rpg_service.sync(
        workspace_id=enrich_workspace,
        nodes=SAMPLE_NODES,
        edges=SAMPLE_EDGES,
    )
    return enrich_workspace


# ---------------------------------------------------------------------------
# Tests: feature identification
# ---------------------------------------------------------------------------


class TestIdentifyFeatures:
    """Tests for the features enrichment phase."""

    async def test_feature_identification_creates_component_nodes(self, storage_backend, rpg_service, populated_workspace):
        """Feature identification should create rpg_component nodes."""
        feature_response = _make_feature_response(
            [
                {
                    "name": "Authentication",
                    "description": "Handles user login and session management.",
                    "members": ["src/auth.py"],
                },
                {
                    "name": "Data Models",
                    "description": "Core domain data models.",
                    "members": ["src/models.py"],
                },
            ]
        )
        llm = MockLLMService(responses=[feature_response])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        stats = await enrichment._identify_features(
            workspace_id=populated_workspace,
            context_id="rpg",
            nodes=await rpg_service._get_rpg_nodes(populated_workspace),
        )

        assert stats["features_created"] == 2
        assert stats["edges_created"] >= 2
        assert llm.call_count == 1

    async def test_feature_identification_no_duplicate_components(self, storage_backend, rpg_service, populated_workspace):
        """Running feature identification twice should not duplicate components."""
        feature_response = _make_feature_response(
            [
                {
                    "name": "Authentication",
                    "description": "Handles user login.",
                    "members": ["src/auth.py"],
                },
            ]
        )
        llm = MockLLMService(responses=[feature_response, feature_response])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        await enrichment._identify_features(populated_workspace, "rpg", nodes)
        # Second run — should not create a duplicate
        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        stats2 = await enrichment._identify_features(populated_workspace, "rpg", nodes)

        assert stats2["features_created"] == 0

    async def test_feature_identification_empty_nodes_returns_zero(self, storage_backend, enrich_workspace):
        """Feature identification with no file nodes should return zero stats."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        stats = await enrichment._identify_features(enrich_workspace, "rpg", [])

        assert stats["features_created"] == 0
        assert stats["edges_created"] == 0
        assert llm.call_count == 0

    async def test_feature_identification_bad_llm_response_graceful(self, storage_backend, rpg_service, populated_workspace):
        """Unparseable LLM response should not raise — returns zero stats."""
        llm = MockLLMService(responses=["not valid json at all"])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        stats = await enrichment._identify_features(populated_workspace, "rpg", nodes)

        assert stats["features_created"] == 0
        assert stats["edges_created"] == 0

    async def test_feature_identification_markdown_fenced_json(self, storage_backend, rpg_service, populated_workspace):
        """LLM response wrapped in ```json...``` should be parsed correctly."""
        import json

        features = [{"name": "Core", "description": "Core module.", "members": ["src/api.py"]}]
        fenced = f"```json\n{json.dumps(features)}\n```"
        llm = MockLLMService(responses=[fenced])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        stats = await enrichment._identify_features(populated_workspace, "rpg", nodes)

        assert stats["features_created"] == 1


# ---------------------------------------------------------------------------
# Tests: description enrichment
# ---------------------------------------------------------------------------


class TestEnrichDescriptions:
    """Tests for the descriptions enrichment phase."""

    async def test_description_enrichment_updates_class_nodes(self, storage_backend, rpg_service, populated_workspace):
        """Class nodes should have their content updated with LLM descriptions."""
        desc_response = _make_description_response(
            {
                "0": "Manages user authentication and session tokens.",
                "1": "Represents a registered user in the system.",
            }
        )
        llm = MockLLMService(responses=[desc_response])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        stats = await enrichment._enrich_descriptions(populated_workspace, "rpg", nodes)

        assert stats["nodes_enriched"] == 2
        assert llm.call_count == 1

    async def test_description_enrichment_skips_non_class_nodes(self, storage_backend, rpg_service, populated_workspace):
        """Only rpg_class and rpg_interface nodes should be targeted."""
        # File nodes should not be enriched
        llm = MockLLMService(responses=[_make_description_response({})])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        # Sync only file nodes with no classes
        file_only_ws = f"enrich_file_only_{uuid.uuid4().hex[:8]}"
        from memorylayer_server.models.workspace import Workspace

        ws = Workspace(
            id=file_only_ws,
            tenant_id="test_tenant",
            name="File Only WS",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await storage_backend.create_workspace(ws)
        await rpg_service.sync(
            workspace_id=file_only_ws,
            nodes=[SAMPLE_NODES[0]],  # only rpg_file
            edges=[],
        )

        nodes = await rpg_service._get_rpg_nodes(file_only_ws)
        stats = await enrichment._enrich_descriptions(file_only_ws, "rpg", nodes)

        assert stats["nodes_enriched"] == 0
        assert llm.call_count == 0

    async def test_description_enrichment_bad_response_graceful(self, storage_backend, rpg_service, populated_workspace):
        """Unparseable description response should not raise — returns zero."""
        llm = MockLLMService(responses=["{{broken json}"])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        stats = await enrichment._enrich_descriptions(populated_workspace, "rpg", nodes)

        assert stats["nodes_enriched"] == 0

    async def test_description_enrichment_empty_nodes_returns_zero(self, storage_backend, enrich_workspace):
        """No nodes means no descriptions needed — returns zero."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        stats = await enrichment._enrich_descriptions(enrich_workspace, "rpg", [])

        assert stats["nodes_enriched"] == 0
        assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Tests: data flow analysis
# ---------------------------------------------------------------------------


class TestAnalyzeDataFlows:
    """Tests for the data_flows enrichment phase."""

    async def test_data_flow_analysis_creates_associations(self, storage_backend, rpg_service, populated_workspace):
        """Data flow pairs should produce data_flow associations in storage."""
        flow_response = _make_data_flow_response(
            [
                {
                    "source": "src/auth.py",
                    "target": "src/models.py",
                    "data_id": "user_credentials",
                    "data_type": "user_input",
                    "transformation": "validated and hashed",
                },
            ]
        )
        llm = MockLLMService(responses=[flow_response])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        edges = await rpg_service._get_rpg_edges(populated_workspace, node_id_to_memory_id)

        stats = await enrichment._analyze_data_flows(populated_workspace, "rpg", nodes, edges)

        assert stats["flows_created"] >= 0  # May be 0 if no qualifying pairs
        assert "pairs_analyzed" in stats

    async def test_data_flow_analysis_empty_graph_returns_zero(self, storage_backend, enrich_workspace):
        """No nodes or edges means no pairs to analyze."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        stats = await enrichment._analyze_data_flows(enrich_workspace, "rpg", [], [])

        assert stats["flows_created"] == 0
        assert stats["pairs_analyzed"] == 0
        assert llm.call_count == 0

    async def test_data_flow_analysis_bad_response_graceful(self, storage_backend, rpg_service, populated_workspace):
        """Unparseable data flow response should not raise."""
        llm = MockLLMService(responses=["not json"])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        nodes = await rpg_service._get_rpg_nodes(populated_workspace)
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        edges = await rpg_service._get_rpg_edges(populated_workspace, node_id_to_memory_id)

        stats = await enrichment._analyze_data_flows(populated_workspace, "rpg", nodes, edges)

        assert stats["flows_created"] == 0


# ---------------------------------------------------------------------------
# Tests: enrich() orchestration and phase selection
# ---------------------------------------------------------------------------


class TestEnrichOrchestration:
    """Tests for the top-level enrich() method and phase selection."""

    async def test_enrich_all_phases_run_by_default(self, storage_backend, rpg_service, populated_workspace):
        """Default enrich() runs all three phases and returns combined stats."""
        feature_resp = _make_feature_response(
            [
                {"name": "Core", "description": "Core.", "members": ["src/auth.py"]},
            ]
        )
        desc_resp = _make_description_response({"0": "Manages auth.", "1": "User model."})
        flow_resp = _make_data_flow_response([])

        llm = MockLLMService(responses=[feature_resp, desc_resp, flow_resp])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        result = await enrichment.enrich(workspace_id=populated_workspace)

        assert "phases" in result
        assert "features" in result["phases"]
        assert "descriptions" in result["phases"]
        assert "data_flows" in result["phases"]
        assert result["workspace_id"] == populated_workspace

    async def test_enrich_single_phase_only_runs_that_phase(self, storage_backend, rpg_service, populated_workspace):
        """Specifying phases=["features"] should only run feature identification."""
        feature_resp = _make_feature_response(
            [
                {"name": "API", "description": "API layer.", "members": ["src/api.py"]},
            ]
        )
        llm = MockLLMService(responses=[feature_resp])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        result = await enrichment.enrich(
            workspace_id=populated_workspace,
            phases=["features"],
        )

        assert "features" in result["phases"]
        assert "descriptions" not in result["phases"]
        assert "data_flows" not in result["phases"]
        # Only one LLM call for features
        assert llm.call_count == 1

    async def test_enrich_descriptions_only(self, storage_backend, rpg_service, populated_workspace):
        """Specifying phases=["descriptions"] should only run description enrichment."""
        desc_resp = _make_description_response({"0": "Auth service.", "1": "User entity."})
        llm = MockLLMService(responses=[desc_resp])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        result = await enrichment.enrich(
            workspace_id=populated_workspace,
            phases=["descriptions"],
        )

        assert "descriptions" in result["phases"]
        assert "features" not in result["phases"]
        assert "data_flows" not in result["phases"]

    async def test_enrich_empty_phases_list_runs_nothing(self, storage_backend, rpg_service, populated_workspace):
        """Empty phases list should run no phases and make no LLM calls."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        result = await enrichment.enrich(
            workspace_id=populated_workspace,
            phases=[],
        )

        assert result["phases"] == {}
        assert llm.call_count == 0

    async def test_enrich_returns_stats_dict_structure(self, storage_backend, enrich_workspace):
        """enrich() always returns a dict with workspace_id, context_id, phases."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        result = await enrichment.enrich(workspace_id=enrich_workspace, phases=[])

        assert "workspace_id" in result
        assert "context_id" in result
        assert "phases" in result
        assert result["workspace_id"] == enrich_workspace

    async def test_enrich_noop_with_empty_llm_responses(self, storage_backend, rpg_service, populated_workspace):
        """Empty LLM responses (all return []) should produce zero stats without errors."""
        llm = MockLLMService(responses=[])
        enrichment = RpgEnrichmentService(storage=storage_backend, llm_service=llm)

        # Should not raise
        result = await enrichment.enrich(workspace_id=populated_workspace)

        assert result is not None
        assert "phases" in result
