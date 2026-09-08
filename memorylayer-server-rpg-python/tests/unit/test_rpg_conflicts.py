# SPDX-License-Identifier: Apache-2.0
"""Tests for RPG conflict detection service.

Tests detect_file_conflicts() and detect_symbol_conflicts() across all
conflict severity scenarios: no conflict, medium, high, empty overlays,
self-exclusion, deduplication, and mixed-severity cases.
"""

import uuid
from datetime import UTC, datetime

import pytest_asyncio
from memorylayer_server.models.workspace import Workspace

from memorylayer_server_rpg.services.conflicts import RpgConflictService
from memorylayer_server_rpg.services.service import RpgService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rpg_service(storage_backend) -> RpgService:
    """Create an RPG service instance backed by the test storage."""
    return RpgService(storage_backend)


@pytest_asyncio.fixture
async def conflict_service(storage_backend, rpg_service) -> RpgConflictService:
    """Create a conflict service instance backed by the same storage and rpg service."""
    return RpgConflictService(storage_backend, rpg_service)


@pytest_asyncio.fixture
async def rpg_workspace(storage_backend) -> str:
    """Create a unique workspace for each conflict test."""
    ws_id = f"conflict_test_{uuid.uuid4().hex[:8]}"
    ws = Workspace(
        id=ws_id,
        tenant_id="test_tenant",
        name="RPG Conflict Test Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await storage_backend.create_workspace(ws)
    return ws_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent_nodes(task_id: str, file_paths: list[str]) -> list[dict]:
    """Build rpg_file nodes for an intent overlay (rpg-intent-{task_id})."""
    return [
        {
            "node_id": path,
            "node_type": "rpg_file",
            "path": path,
            "name": path.split("/")[-1],
            "description": f"File {path} claimed by task {task_id}",
        }
        for path in file_paths
    ]


def _task_symbol_nodes(task_id: str, symbols: list[dict]) -> list[dict]:
    """Build symbol nodes for a task overlay (rpg-task-{task_id}).

    Each symbol dict must have: node_id, node_type, path, name.
    """
    return [
        {
            "node_id": sym["node_id"],
            "node_type": sym["node_type"],
            "path": sym["path"],
            "name": sym["name"],
            "description": f"Symbol {sym['name']} modified by task {task_id}",
        }
        for sym in symbols
    ]


async def _sync_intent(rpg_service: RpgService, workspace_id: str, task_id: str, file_paths: list[str]) -> None:
    """Sync intent overlay nodes for a task."""
    nodes = _intent_nodes(task_id, file_paths)
    await rpg_service.sync(
        workspace_id=workspace_id,
        nodes=nodes,
        edges=[],
        context_id=f"rpg-intent-{task_id}",
    )


async def _sync_task_symbols(
    rpg_service: RpgService,
    workspace_id: str,
    task_id: str,
    symbols: list[dict],
) -> None:
    """Sync task overlay symbol nodes for a task."""
    nodes = _task_symbol_nodes(task_id, symbols)
    await rpg_service.sync(
        workspace_id=workspace_id,
        nodes=nodes,
        edges=[],
        context_id=f"rpg-task-{task_id}",
    )


# ---------------------------------------------------------------------------
# Tests: detect_file_conflicts()
# ---------------------------------------------------------------------------


class TestDetectFileConflicts:
    """Tests for detect_file_conflicts() method."""

    async def test_returns_no_conflicts_when_files_are_disjoint(self, conflict_service, rpg_service, rpg_workspace):
        """Returns empty conflicts list when task has no overlapping files with other intents."""
        task_a = "task-a"
        task_b = "task-b"
        await _sync_intent(rpg_service, rpg_workspace, task_a, ["src/auth.py"])
        await _sync_intent(rpg_service, rpg_workspace, task_b, ["src/models.py"])

        result = await conflict_service.detect_file_conflicts(rpg_workspace, task_a)

        assert result["task_id"] == task_a
        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_medium_severity_when_two_tasks_claim_same_file(self, conflict_service, rpg_service, rpg_workspace):
        """Reports medium severity when exactly one other task claims the same file."""
        task_a = "task-med-a"
        task_b = "task-med-b"
        shared_file = "src/shared.py"
        await _sync_intent(rpg_service, rpg_workspace, task_a, [shared_file])
        await _sync_intent(rpg_service, rpg_workspace, task_b, [shared_file])

        result = await conflict_service.detect_file_conflicts(rpg_workspace, task_a)

        assert result["total_conflicts"] == 1
        conflict = result["conflicts"][0]
        assert conflict["file_path"] == shared_file
        assert conflict["severity"] == "medium"
        assert task_b in conflict["task_ids"]

    async def test_high_severity_when_three_or_more_tasks_claim_same_file(self, conflict_service, rpg_service, rpg_workspace):
        """Reports high severity when 2+ other tasks also claim the same file."""
        task_a = "task-high-a"
        task_b = "task-high-b"
        task_c = "task-high-c"
        shared_file = "src/hotspot.py"
        await _sync_intent(rpg_service, rpg_workspace, task_a, [shared_file])
        await _sync_intent(rpg_service, rpg_workspace, task_b, [shared_file])
        await _sync_intent(rpg_service, rpg_workspace, task_c, [shared_file])

        result = await conflict_service.detect_file_conflicts(rpg_workspace, task_a)

        assert result["total_conflicts"] == 1
        conflict = result["conflicts"][0]
        assert conflict["file_path"] == shared_file
        assert conflict["severity"] == "high"
        assert len(conflict["task_ids"]) == 2
        assert task_b in conflict["task_ids"]
        assert task_c in conflict["task_ids"]

    async def test_returns_empty_when_no_intent_overlays_exist(self, conflict_service, rpg_workspace):
        """Returns empty result when no intent overlays have been registered."""
        result = await conflict_service.detect_file_conflicts(rpg_workspace, "orphan-task")

        assert result["task_id"] == "orphan-task"
        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_ignores_target_tasks_own_intent_overlay(self, conflict_service, rpg_service, rpg_workspace):
        """Does not report the target task's own overlay as a conflict."""
        task_a = "task-self"
        await _sync_intent(rpg_service, rpg_workspace, task_a, ["src/solo.py"])

        result = await conflict_service.detect_file_conflicts(rpg_workspace, task_a)

        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_mixed_severity_across_multiple_files(self, conflict_service, rpg_service, rpg_workspace):
        """Correctly reports high and medium severities for different files in one call."""
        task_a = "task-mix-a"
        task_b = "task-mix-b"
        task_c = "task-mix-c"
        high_file = "src/busy.py"
        medium_file = "src/moderate.py"
        # task_a claims both files
        await _sync_intent(rpg_service, rpg_workspace, task_a, [high_file, medium_file])
        # task_b claims both
        await _sync_intent(rpg_service, rpg_workspace, task_b, [high_file, medium_file])
        # task_c only claims the high_file (making it 2 other tasks = high)
        await _sync_intent(rpg_service, rpg_workspace, task_c, [high_file])

        result = await conflict_service.detect_file_conflicts(rpg_workspace, task_a)

        assert result["total_conflicts"] == 2
        by_path = {c["file_path"]: c for c in result["conflicts"]}
        assert by_path[high_file]["severity"] == "high"
        assert by_path[medium_file]["severity"] == "medium"
        # High conflicts should sort before medium
        assert result["conflicts"][0]["severity"] == "high"


# ---------------------------------------------------------------------------
# Tests: detect_symbol_conflicts()
# ---------------------------------------------------------------------------


class TestDetectSymbolConflicts:
    """Tests for detect_symbol_conflicts() method."""

    async def test_returns_no_conflicts_when_tasks_modify_different_symbols(self, conflict_service, rpg_service, rpg_workspace):
        """Returns empty conflicts when two tasks touch entirely different symbols."""
        task_a = "sym-no-a"
        task_b = "sym-no-b"
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_a,
            [
                {"node_id": "src/auth.py:login", "node_type": "rpg_function", "path": "src/auth.py", "name": "login"},
            ],
        )
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_b,
            [
                {"node_id": "src/models.py:User", "node_type": "rpg_class", "path": "src/models.py", "name": "User"},
            ],
        )

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, task_a, task_b)

        assert result["task_ids"] == [task_a, task_b]
        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_high_severity_for_exact_symbol_match(self, conflict_service, rpg_service, rpg_workspace):
        """Reports high severity when both tasks modify the same symbol (same node_id)."""
        task_a = "sym-exact-a"
        task_b = "sym-exact-b"
        shared_symbol = {
            "node_id": "src/auth.py:validate_token",
            "node_type": "rpg_function",
            "path": "src/auth.py",
            "name": "validate_token",
        }
        await _sync_task_symbols(rpg_service, rpg_workspace, task_a, [shared_symbol])
        await _sync_task_symbols(rpg_service, rpg_workspace, task_b, [shared_symbol])

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, task_a, task_b)

        assert result["total_conflicts"] == 1
        conflict = result["conflicts"][0]
        assert conflict["severity"] == "high"
        assert conflict["symbol_id"] == shared_symbol["node_id"]
        assert conflict["file_path"] == "src/auth.py"
        assert task_a in conflict["task_ids"]
        assert task_b in conflict["task_ids"]

    async def test_medium_severity_for_different_symbols_in_same_file(self, conflict_service, rpg_service, rpg_workspace):
        """Reports medium severity when tasks modify different symbols in the same file."""
        task_a = "sym-same-file-a"
        task_b = "sym-same-file-b"
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_a,
            [
                {"node_id": "src/utils.py:helper_one", "node_type": "rpg_function", "path": "src/utils.py", "name": "helper_one"},
            ],
        )
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_b,
            [
                {"node_id": "src/utils.py:helper_two", "node_type": "rpg_function", "path": "src/utils.py", "name": "helper_two"},
            ],
        )

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, task_a, task_b)

        assert result["total_conflicts"] == 1
        conflict = result["conflicts"][0]
        assert conflict["severity"] == "medium"
        assert conflict["file_path"] == "src/utils.py"

    async def test_returns_empty_when_task_a_overlay_does_not_exist(self, conflict_service, rpg_service, rpg_workspace):
        """Returns empty conflicts when task A has no task overlay at all."""
        task_b = "sym-exist-b"
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_b,
            [
                {"node_id": "src/main.py:run", "node_type": "rpg_function", "path": "src/main.py", "name": "run"},
            ],
        )

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, "sym-ghost-a", task_b)

        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_returns_empty_when_task_b_overlay_does_not_exist(self, conflict_service, rpg_service, rpg_workspace):
        """Returns empty conflicts when task B has no task overlay at all."""
        task_a = "sym-exist-a"
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_a,
            [
                {"node_id": "src/main.py:setup", "node_type": "rpg_function", "path": "src/main.py", "name": "setup"},
            ],
        )

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, task_a, "sym-ghost-b")

        assert result["conflicts"] == []
        assert result["total_conflicts"] == 0

    async def test_deduplicates_conflict_pairs(self, conflict_service, rpg_service, rpg_workspace):
        """Each conflicting symbol pair is reported exactly once."""
        task_a = "sym-dedup-a"
        task_b = "sym-dedup-b"
        # Both tasks touch the same two symbols in the same file — only one medium
        # conflict per pair should appear, not two.
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_a,
            [
                {"node_id": "src/core.py:FeatureA", "node_type": "rpg_class", "path": "src/core.py", "name": "FeatureA"},
                {"node_id": "src/core.py:FeatureB", "node_type": "rpg_class", "path": "src/core.py", "name": "FeatureB"},
            ],
        )
        await _sync_task_symbols(
            rpg_service,
            rpg_workspace,
            task_b,
            [
                {"node_id": "src/core.py:FeatureA", "node_type": "rpg_class", "path": "src/core.py", "name": "FeatureA"},
                {"node_id": "src/core.py:FeatureB", "node_type": "rpg_class", "path": "src/core.py", "name": "FeatureB"},
            ],
        )

        result = await conflict_service.detect_symbol_conflicts(rpg_workspace, task_a, task_b)

        # Both symbols match exactly → 2 high-severity conflicts, no duplicates
        symbol_ids = [c["symbol_id"] for c in result["conflicts"]]
        assert len(symbol_ids) == len(set(symbol_ids)), "Duplicate conflict pairs were reported"
        assert all(c["severity"] == "high" for c in result["conflicts"])
