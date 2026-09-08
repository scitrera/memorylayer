"""Tests for the single-pass graph analysis refactor.

TDD contract:
  1. Golden parity: normalize_for_golden(analyze().model_dump(mode="json")) must be
     identical to the pre-refactor golden fixtures (stored in tests/fixtures/graph_analysis/).
     Normalization replaces all memory IDs with stable positional tokens (sorted) so that
     the comparison is invariant to DB-assigned UUIDs while catching any algorithmic change.
  2. Bulk-fetch assertion: after refactoring, analyze() must trigger exactly ONE call to
     get_associations_batch (or search_memories_by_filter at most 2 times) instead of N
     per-node get_associations calls / 5x search_memories_by_filter calls.

Golden fixtures are regenerated when UPDATE_GOLDEN=1 is set in the environment.
They are committed JSON under tests/fixtures/graph_analysis/ and serve as the
regression contract for the future AGE backend too.
"""

import json
import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from scitrera_app_framework import Variables, get_extension

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.graph_analysis.default import NetworkXGraphAnalysisService

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "graph_analysis"
_UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN", "0").strip() == "1"

# Fixed workspace IDs so fixture data is built once per session in the same DB.
_SMALL_WS_ID = "golden_graph_small_v1"
_MEDIUM_WS_ID = "golden_graph_medium_v1"
_LINEAR_WS_ID = "golden_graph_linear_v1"
_EMPTY_WS_ID = "golden_graph_empty_v1"


# ---------------------------------------------------------------------------
# Normalization: replace all memory IDs with stable positional tokens
# ---------------------------------------------------------------------------



def normalize_for_golden(data: dict) -> dict:
    """Return a structurally-normalized copy of the GraphAnalysis dict for stable golden comparison.

    Because Louvain with seed=42 is deterministic given the same internal NetworkX node ordering,
    but that ordering depends on DB-assigned UUIDs (which vary per test session), the community
    partition assignment is non-deterministic across runs on symmetric graphs.

    This normalization strips all node identity and retains only graph-invariant quantities:

    - snapshot: node_count, edge_count, context_id, includes_rpg (no workspace_id — varies)
    - stats: all numeric fields (all invariant)
    - communities: sorted list of {size, cohesion_score, central_node_count} — no IDs
    - central_nodes: sorted list of {degree, betweenness} — no IDs or community_ids
    - bridges: sorted list of {relationship_type, strength} — no IDs or community_ids

    This is sufficient to catch any algorithmic change (wrong sort, different Louvain call,
    different betweenness formula, wrong bridge detection) while being invariant to UUID ordering.
    """
    # --- snapshot ---
    snap = data.get("snapshot", {})
    snapshot = {
        "node_count": snap.get("node_count", 0),
        "edge_count": snap.get("edge_count", 0),
        "context_id": snap.get("context_id"),
        "includes_rpg": snap.get("includes_rpg", False),
    }

    # --- stats (all numeric) ---
    stats = dict(data.get("stats", {}))

    # --- communities: structural signature only ---
    communities = []
    for comm in data.get("communities", []):
        communities.append(
            {
                "size": comm["size"],
                "cohesion_score": comm["cohesion_score"],
                "central_node_count": len(comm.get("central_node_ids", [])),
                "label": comm.get("label"),
            }
        )
    # Sort by size desc, cohesion desc for stable ordering
    communities.sort(key=lambda c: (-c["size"], -c["cohesion_score"]))

    # --- central_nodes: metric signature only ---
    central_nodes = []
    for node in data.get("central_nodes", []):
        central_nodes.append(
            {
                "degree": node["degree"],
                "betweenness": node["betweenness"],
            }
        )
    # Sort by betweenness desc, degree desc for stable ordering
    central_nodes.sort(key=lambda n: (-n["betweenness"], -n["degree"]))

    # --- bridges: metric signature only ---
    bridges = []
    for bridge in data.get("bridges", []):
        bridges.append(
            {
                "relationship_type": bridge["relationship_type"],
                "strength": bridge["strength"],
            }
        )
    # Sort by strength desc, relationship_type asc
    bridges.sort(key=lambda b: (-b["strength"], b["relationship_type"]))

    return {
        "snapshot": snapshot,
        "communities": communities,
        "central_nodes": central_nodes,
        "bridges": bridges,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_storage(v):
    """Get storage backend from the session-scoped Variables instance."""
    from memorylayer_server.services._constants import EXT_STORAGE_BACKEND

    return get_extension(EXT_STORAGE_BACKEND, v)


async def _ensure_workspace(storage, workspace_id: str) -> None:
    """Create workspace if it doesn't already exist."""
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace

    existing = await storage.get_workspace(workspace_id)
    if not existing:
        ws = Workspace(
            id=workspace_id,
            tenant_id="default_tenant",
            name=f"Test Graph Workspace {workspace_id}",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await storage.create_workspace(ws)


async def _add_memory(storage, workspace_id: str, content: str) -> str:
    """Create a memory and return its ID."""
    mem = await storage.create_memory(workspace_id, RememberInput(content=content))
    return mem.id


async def _add_association(
    storage,
    workspace_id: str,
    source_id: str,
    target_id: str,
    relationship: str = "related_to",
    strength: float = 0.7,
) -> None:
    """Create an association between two memories."""
    await storage.create_association(
        workspace_id,
        AssociateInput(source_id=source_id, target_id=target_id, relationship=relationship, strength=strength),
    )


def _make_service(storage) -> NetworkXGraphAnalysisService:
    """Instantiate the service without going through the full DI plugin system."""
    v = Variables()
    return NetworkXGraphAnalysisService(storage=storage, v=v)


async def _build_small_workspace(storage, workspace_id: str) -> None:
    """Small graph: 4 nodes, 4 edges, 2 natural communities.

    Community A: m0-m1-m2 (triangle, dense)
    Community B: m3 (singleton bridged to m1)
    Bridge: m1 -> m3

    Also includes a similar_to edge between m0 and m2.
    """
    await _ensure_workspace(storage, workspace_id)

    # Guard: skip if already populated (re-used across test runs in same session)
    from memorylayer_server.models.memory import MemoryStatus

    existing = await storage.search_memories_by_filter(workspace_id, status="active", limit=1)
    if existing:
        return

    m0 = await _add_memory(storage, workspace_id, "Small graph node alpha")
    m1 = await _add_memory(storage, workspace_id, "Small graph node beta")
    m2 = await _add_memory(storage, workspace_id, "Small graph node gamma")
    m3 = await _add_memory(storage, workspace_id, "Small graph node delta")

    await _add_association(storage, workspace_id, m0, m1, "related_to", 0.9)
    await _add_association(storage, workspace_id, m1, m2, "related_to", 0.8)
    await _add_association(storage, workspace_id, m0, m2, "similar_to", 0.75)
    await _add_association(storage, workspace_id, m1, m3, "related_to", 0.3)


async def _build_medium_workspace(storage, workspace_id: str) -> None:
    """Medium graph: 9 nodes, two dense clusters + a bridge node.

    Cluster A: m0-m4 (fully connected 5-clique)
    Cluster B: m5-m8 (4-clique)
    Bridge: m2 -> m6 (cross-cluster)
    Typed edges: similar_to, causes, solves
    """
    await _ensure_workspace(storage, workspace_id)

    existing = await storage.search_memories_by_filter(workspace_id, status="active", limit=1)
    if existing:
        return

    nodes_a = []
    for i in range(5):
        mid = await _add_memory(storage, workspace_id, f"Medium cluster-A node {i}")
        nodes_a.append(mid)

    nodes_b = []
    for i in range(4):
        mid = await _add_memory(storage, workspace_id, f"Medium cluster-B node {i}")
        nodes_b.append(mid)

    # Intra-cluster A edges (clique)
    for i in range(len(nodes_a)):
        for j in range(i + 1, len(nodes_a)):
            await _add_association(storage, workspace_id, nodes_a[i], nodes_a[j], "similar_to", 0.85)

    # Intra-cluster B edges (clique)
    for i in range(len(nodes_b)):
        for j in range(i + 1, len(nodes_b)):
            await _add_association(storage, workspace_id, nodes_b[i], nodes_b[j], "causes", 0.8)

    # Bridge edge between clusters
    await _add_association(storage, workspace_id, nodes_a[2], nodes_b[1], "solves", 0.4)


async def _build_linear_workspace(storage, workspace_id: str) -> None:
    """Linear chain: 6 nodes in a line, no strong communities.

    m0 - m1 - m2 - m3 - m4 - m5
    """
    await _ensure_workspace(storage, workspace_id)

    existing = await storage.search_memories_by_filter(workspace_id, status="active", limit=1)
    if existing:
        return

    prev = None
    for i in range(6):
        mid = await _add_memory(storage, workspace_id, f"Linear chain node {i}")
        if prev is not None:
            await _add_association(storage, workspace_id, prev, mid, "leads_to", 0.6)
        prev = mid


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def session_storage(v):
    """Session-scoped access to the storage backend."""
    return _get_storage(v)


@pytest_asyncio.fixture(scope="session")
async def small_ws(session_storage):
    await _build_small_workspace(session_storage, _SMALL_WS_ID)
    return _SMALL_WS_ID


@pytest_asyncio.fixture(scope="session")
async def medium_ws(session_storage):
    await _build_medium_workspace(session_storage, _MEDIUM_WS_ID)
    return _MEDIUM_WS_ID


@pytest_asyncio.fixture(scope="session")
async def linear_ws(session_storage):
    await _build_linear_workspace(session_storage, _LINEAR_WS_ID)
    return _LINEAR_WS_ID


@pytest_asyncio.fixture(scope="session")
async def empty_ws(session_storage):
    await _ensure_workspace(session_storage, _EMPTY_WS_ID)
    return _EMPTY_WS_ID


@pytest.fixture(scope="session")
def svc(session_storage):
    return _make_service(session_storage)


# ---------------------------------------------------------------------------
# Golden parity helpers
# ---------------------------------------------------------------------------


def _golden_path(name: str) -> Path:
    return _FIXTURE_DIR / f"{name}.json"


def _load_golden(name: str) -> dict:
    p = _golden_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Golden fixture not found: {p}\n"
            "Run with UPDATE_GOLDEN=1 to generate it from the current implementation."
        )
    return json.loads(p.read_text())


def _save_golden(name: str, data: dict) -> None:
    p = _golden_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


async def _assert_golden(svc, workspace_id: str, name: str) -> None:
    """Run analyze() and either save or compare against the normalized golden fixture.

    Normalization replaces all memory IDs with stable positional tokens so that
    comparison is invariant to DB-assigned UUIDs while still catching algorithmic changes.
    """
    result = await svc.analyze(workspace_id)
    raw = result.model_dump(mode="json")
    actual = normalize_for_golden(raw)

    if _UPDATE_GOLDEN:
        _save_golden(name, actual)
        return

    expected = _load_golden(name)
    actual_str = json.dumps(actual, sort_keys=True)
    expected_str = json.dumps(expected, sort_keys=True)
    assert actual_str == expected_str, (
        f"Golden mismatch for '{name}'.\n"
        "The refactor changed the output — fix the implementation, not the golden.\n"
        f"Expected length: {len(expected_str)}, Actual length: {len(actual_str)}"
    )


# ---------------------------------------------------------------------------
# Test class: golden parity
# ---------------------------------------------------------------------------


class TestGoldenParity:
    """analyze() normalized output must match pre-refactor golden fixtures."""

    @pytest.mark.asyncio
    async def test_small_graph_golden(self, svc, small_ws):
        """Small 4-node graph with 2 communities and a bridge."""
        await _assert_golden(svc, small_ws, "small")

    @pytest.mark.asyncio
    async def test_medium_graph_golden(self, svc, medium_ws):
        """Medium 9-node graph with two dense clusters and a cross-cluster bridge."""
        await _assert_golden(svc, medium_ws, "medium")

    @pytest.mark.asyncio
    async def test_linear_graph_golden(self, svc, linear_ws):
        """Linear 6-node chain with weak community structure."""
        await _assert_golden(svc, linear_ws, "linear")

    @pytest.mark.asyncio
    async def test_empty_graph_golden(self, svc, empty_ws):
        """Empty workspace: all result fields should be empty/zero."""
        await _assert_golden(svc, empty_ws, "empty")


# ---------------------------------------------------------------------------
# Test class: single bulk-fetch assertion
# ---------------------------------------------------------------------------


class TestSinglePassBulkFetch:
    """After the refactor, analyze() must use ONE bulk edge fetch, not N per-node calls."""

    @pytest.mark.asyncio
    async def test_analyze_uses_bulk_fetch_not_per_node(self, session_storage, small_ws):
        """analyze() must call get_associations_batch exactly once.

        Spies on both get_associations (per-node) and get_associations_batch (bulk).
        After refactor: exactly 1 bulk call from _analyze_core.
        """
        svc = _make_service(session_storage)

        bulk_calls = []
        original_bulk = session_storage.get_associations_batch

        async def spy_bulk(*args, **kwargs):
            bulk_calls.append((args, kwargs))
            return await original_bulk(*args, **kwargs)

        session_storage.get_associations_batch = spy_bulk

        try:
            await svc.analyze(small_ws)
        finally:
            session_storage.get_associations_batch = original_bulk

        assert bulk_calls, (
            "analyze() did not call get_associations_batch at all. "
            "The single-pass refactor must use the bulk fetch path."
        )
        assert len(bulk_calls) == 1, (
            f"analyze() called get_associations_batch {len(bulk_calls)} times; expected exactly 1. "
            "The single-pass _analyze_core must build the graph once."
        )

    @pytest.mark.asyncio
    async def test_analyze_builds_graph_once(self, session_storage, medium_ws):
        """analyze() must call search_memories_by_filter at most 2 times.

        Old code: 5 calls (one per sub-method, each rebuilds the graph).
        New single-pass: 1 call (or 2 if include_rpg=True adds a second search).
        """
        svc = _make_service(session_storage)

        search_calls = []
        original_search = session_storage.search_memories_by_filter

        async def spy_search(*args, **kwargs):
            search_calls.append((args, kwargs))
            return await original_search(*args, **kwargs)

        session_storage.search_memories_by_filter = spy_search

        try:
            await svc.analyze(medium_ws)
        finally:
            session_storage.search_memories_by_filter = original_search

        assert len(search_calls) <= 2, (
            f"analyze() called search_memories_by_filter {len(search_calls)} times; "
            f"expected <= 2 (single-pass). The old code called it 5 times."
        )

    @pytest.mark.asyncio
    async def test_public_methods_still_work_independently(self, svc, small_ws):
        """The 6 public ABC methods must still function when called directly."""
        snapshot = await svc.build_workspace_graph(small_ws)
        assert snapshot.node_count == 4
        assert snapshot.edge_count == 4

        communities = await svc.detect_communities(small_ws)
        assert isinstance(communities, list)
        assert len(communities) >= 1

        central_nodes = await svc.compute_centrality(small_ws)
        assert isinstance(central_nodes, list)
        assert len(central_nodes) == 4

        bridges = await svc.get_bridges(small_ws)
        assert isinstance(bridges, list)

        stats = await svc.get_statistics(small_ws)
        assert stats.node_count == 4
        assert stats.edge_count == 4

        analysis = await svc.analyze(small_ws)
        assert analysis.snapshot.node_count == 4
        assert len(analysis.communities) >= 1
        assert len(analysis.central_nodes) == 4

    @pytest.mark.asyncio
    async def test_central_nodes_carry_memory_label(self, svc, small_ws):
        """Central nodes expose a human-readable label (memory abstract/content
        snippet) so the graph UI can show text instead of raw memory ids."""
        nodes = await svc.compute_centrality(small_ws)
        assert nodes
        # Every node has a non-empty label that is NOT just the memory id.
        assert all(n.label for n in nodes)
        assert all(n.label != n.memory_id for n in nodes)
        # Labels are the seeded memory contents.
        assert {n.label for n in nodes} == {
            "Small graph node alpha",
            "Small graph node beta",
            "Small graph node gamma",
            "Small graph node delta",
        }
        # analyze() carries the same enrichment.
        analysis = await svc.analyze(small_ws)
        assert all(n.label for n in analysis.central_nodes)

    @pytest.mark.asyncio
    async def test_communities_only_skips_central_nodes(self, svc, small_ws):
        """include_central_nodes=False returns the lighter communities-only view:
        communities + bridges + stats, but no per-memory central_nodes."""
        full = await svc.analyze(small_ws)
        light = await svc.analyze(small_ws, include_central_nodes=False)

        assert full.central_nodes            # full graph carries them
        assert light.central_nodes == []     # communities-only drops them
        # Communities + stats are still computed identically.
        assert len(light.communities) == len(full.communities)
        assert light.stats.node_count == full.stats.node_count
        assert light.stats.edge_count == full.stats.edge_count
