"""Unit tests for AetherServiceConnection's POOL task-handler dispatch.

Covers the concurrency cap + bounded-shutdown behaviour added to stop a burst of
pool assignments (e.g. a ``doc_verify`` sweep resuming many ``document_render``
jobs) from running every handler at once, and to keep a handler that's blocked on
a (silently) dead channel from hanging termination past the k8s grace.

What is tested here (no live gateway — the handler + client are mocked):
- ``task_concurrency`` bounds how many handlers run concurrently.
- ``task_concurrency <= 0`` disables the cap (no semaphore).
- Dispatched handlers are tracked in ``_inflight_tasks`` and discarded on
  completion.
- ``disconnect()`` drains in-flight handlers and CANCELS the ones that don't
  finish within the drain grace (a hung handler cannot stall shutdown).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.aether_service import (
    AetherServiceConnection,
    TASK_LANE_DEFAULT,
    TASK_LANE_DOCUMENT,
    TASK_LANE_FANOUT,
    resolve_task_lane,
)


@pytest.fixture
def mock_variables():
    """Minimal Variables stub satisfying ``get_logger`` and ``environ``."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kwargs: default)
    return v


def _make(mock_variables, **overrides) -> AetherServiceConnection:
    kwargs = dict(
        gateway_addr="localhost:50051",
        workspace="test-workspace",
        specifier="main",
        liveness_enabled=False,  # watchdog irrelevant here
        shutdown_drain_timeout_s=0.05,
        shutdown_close_timeout_s=0.05,
    )
    kwargs.update(overrides)
    return AetherServiceConnection(mock_variables, **kwargs)


def _client_with_close() -> MagicMock:
    client = MagicMock()
    client.close = AsyncMock()
    return client


class TestConcurrencyCap:
    def test_no_semaphore_when_concurrency_zero(self, mock_variables):
        assert _make(mock_variables, task_concurrency=0)._task_semaphore is None

    def test_semaphore_present_when_capped(self, mock_variables):
        assert _make(mock_variables, task_concurrency=3)._task_semaphore is not None

    @pytest.mark.asyncio
    async def test_cap_bounds_concurrent_handlers(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=2)
        running = 0
        peak = 0
        gate = asyncio.Event()

        async def handler(_assignment):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await gate.wait()
            finally:
                running -= 1

        svc.set_task_assignment_handler(handler)
        for _ in range(5):
            await svc._on_task_assignment(object())

        await asyncio.sleep(0.05)  # let handlers acquire the semaphore
        assert peak == 2  # only two ran at once despite five assignments
        assert running == 2

        gate.set()
        await asyncio.gather(*list(svc._inflight_tasks), return_exceptions=True)
        assert running == 0


class TestInflightTracking:
    @pytest.mark.asyncio
    async def test_tracked_then_discarded_on_completion(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=0)

        async def handler(_assignment):
            return None

        svc.set_task_assignment_handler(handler)
        await svc._on_task_assignment(object())
        assert len(svc._inflight_tasks) == 1  # tracked before it runs

        await asyncio.sleep(0.02)  # let it run + fire the done-callback
        assert len(svc._inflight_tasks) == 0


class TestBoundedShutdown:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_hung_handler(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=0)
        started = asyncio.Event()
        cancelled = False

        async def handler(_assignment):
            nonlocal cancelled
            started.set()
            try:
                await asyncio.sleep(100)  # never completes on its own
            except asyncio.CancelledError:
                cancelled = True
                raise

        svc.set_task_assignment_handler(handler)
        client = _client_with_close()
        svc._client = client

        await svc._on_task_assignment(object())
        await started.wait()
        assert len(svc._inflight_tasks) == 1

        # Drain grace (0.05s) elapses -> the hung handler is cancelled, and the
        # bounded client.close() is awaited.
        await svc.disconnect()

        assert cancelled is True
        assert [t for t in svc._inflight_tasks if not t.done()] == []
        assert svc._client is None  # reset in disconnect's finally
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_awaits_finished_handler(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=0)

        async def handler(_assignment):
            await asyncio.sleep(0.001)

        svc.set_task_assignment_handler(handler)
        client = _client_with_close()
        svc._client = client

        await svc._on_task_assignment(object())
        await svc.disconnect()  # should drain the (quickly-finishing) handler

        client.close.assert_awaited_once()


class TestTaskMonitor:
    """The periodic monitor logs the in-flight count + per-type breakdown while
    tasks run, and stays quiet when idle."""

    @pytest.mark.asyncio
    async def test_monitor_logs_live_count_and_breakdown(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=2)
        svc.logger = MagicMock()
        svc._task_monitor_interval_s = 0.01

        async def _park():
            await asyncio.sleep(5)

        # Handler tasks are named by task_type (as _on_task_assignment does).
        tasks = [
            asyncio.create_task(_park(), name="document_render"),
            asyncio.create_task(_park(), name="document_render"),
            asyncio.create_task(_park(), name="decay_memories"),
        ]
        svc._inflight_tasks.update(tasks)

        mon = asyncio.create_task(svc._task_monitor_loop())
        await asyncio.sleep(0.05)  # let it tick a few times
        mon.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(mon, *tasks, return_exceptions=True)

        inflight_calls = [
            c.args for c in svc.logger.info.call_args_list
            if c.args and "in flight" in str(c.args[0])
        ]
        assert inflight_calls, "monitor never logged an in-flight line"
        args = inflight_calls[0]
        assert args[1] == 3  # total live count
        assert "document_render=2" in args[-1] and "decay_memories=1" in args[-1]

    @pytest.mark.asyncio
    async def test_monitor_silent_when_idle(self, mock_variables):
        svc = _make(mock_variables, task_concurrency=2)
        svc.logger = MagicMock()
        svc._task_monitor_interval_s = 0.01

        mon = asyncio.create_task(svc._task_monitor_loop())
        await asyncio.sleep(0.05)
        mon.cancel()
        await asyncio.gather(mon, return_exceptions=True)

        assert not [
            c for c in svc.logger.info.call_args_list
            if c.args and "in flight" in str(c.args[0])
        ]


def _assignment(task_type: str):
    return SimpleNamespace(task_type=task_type)


class TestTaskLanes:
    """Lanes exist so fan-out volume cannot starve the document pipeline.

    One decomposed page yields ~35-40 facts and each fact schedules its own
    generate_tiers + auto_enrich, so >1000 enrichment tasks routinely queue
    during an ingest. asyncio.Semaphore is FIFO, so under a single shared cap
    the next document task waits behind all of them.
    """

    @pytest.mark.parametrize(
        "task_type,expected",
        [
            ("document_transcribe", TASK_LANE_DOCUMENT),
            ("document_render", TASK_LANE_DOCUMENT),
            ("doc_added", TASK_LANE_DOCUMENT),
            ("doc_verify", TASK_LANE_DOCUMENT),
            ("auto_enrich", TASK_LANE_FANOUT),
            ("generate_tiers", TASK_LANE_FANOUT),
            ("decompose_facts", TASK_LANE_FANOUT),
            ("reindex_memory", TASK_LANE_FANOUT),
            ("kb_update", TASK_LANE_FANOUT),
            ("session_cleanup", TASK_LANE_DEFAULT),
            ("decay_memories", TASK_LANE_DEFAULT),
            # An unrecognised type must NOT land in the document lane and spend
            # its reserved slots.
            ("some_future_task", TASK_LANE_DEFAULT),
            (None, TASK_LANE_DEFAULT),
            # NAMESPACED wire form. Task types reach the handler qualified
            # ("memorylayer-task.auto_enrich"), not bare, and matching the raw
            # value routed every one of them to the default lane -- leaving the
            # lanes inert in production while every unit test still passed.
            ("memorylayer-task.document_transcribe", TASK_LANE_DOCUMENT),
            ("memorylayer-task.doc_verify", TASK_LANE_DOCUMENT),
            ("memorylayer-task.doc_added", TASK_LANE_DOCUMENT),
            ("memorylayer-task.auto_enrich", TASK_LANE_FANOUT),
            ("memorylayer-task.generate_tiers", TASK_LANE_FANOUT),
            ("memorylayer-task.decompose_facts", TASK_LANE_FANOUT),
            ("memorylayer-task.kb_update", TASK_LANE_FANOUT),
            ("memorylayer-task.reindex_memory", TASK_LANE_FANOUT),
            ("memorylayer-task.job_reconcile", TASK_LANE_DEFAULT),
            ("memorylayer-task.session_cleanup", TASK_LANE_DEFAULT),
        ],
    )
    def test_task_types_map_to_lanes(self, task_type, expected):
        assert resolve_task_lane(task_type) == expected

    def test_namespaced_and_bare_names_route_identically(self):
        """Whatever the wire format turns out to be, the two must not diverge."""
        for bare in ("document_render", "auto_enrich", "job_reconcile"):
            assert resolve_task_lane(bare) == resolve_task_lane(f"memorylayer-task.{bare}")

    @pytest.mark.asyncio
    async def test_namespaced_fanout_flood_does_not_block_namespaced_document_work(
        self, mock_variables,
    ):
        """The production failure, reproduced at the dispatch layer.

        The lanes existed and every bare-name test passed, yet a doc_verify sweep
        still queued behind ~8500 enrichment tasks because the real task types
        matched nothing.
        """
        svc = _make(mock_variables, fanout_task_concurrency=2, document_task_concurrency=2)
        gate = asyncio.Event()
        started: list[str] = []

        async def handler(assignment):
            started.append(assignment.task_type)
            await gate.wait()

        svc.set_task_assignment_handler(handler)
        for _ in range(10):
            await svc._on_task_assignment(_assignment("memorylayer-task.auto_enrich"))
        await asyncio.sleep(0.05)
        assert started.count("memorylayer-task.auto_enrich") == 2

        await svc._on_task_assignment(_assignment("memorylayer-task.doc_verify"))
        await asyncio.sleep(0.05)
        assert "memorylayer-task.doc_verify" in started

        gate.set()
        await asyncio.gather(*list(svc._inflight_tasks), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_saturated_fanout_lane_does_not_block_document_work(self, mock_variables):
        """The regression this whole mechanism exists for."""
        svc = _make(mock_variables, fanout_task_concurrency=2, document_task_concurrency=2)
        gate = asyncio.Event()
        started: list[str] = []

        async def handler(assignment):
            started.append(assignment.task_type)
            await gate.wait()

        svc.set_task_assignment_handler(handler)

        # Flood the fan-out lane well past its cap.
        for _ in range(10):
            await svc._on_task_assignment(_assignment("auto_enrich"))
        await asyncio.sleep(0.05)
        assert started.count("auto_enrich") == 2  # lane is saturated

        # A document task arriving AFTER the flood must still run immediately.
        await svc._on_task_assignment(_assignment("document_transcribe"))
        await asyncio.sleep(0.05)
        assert "document_transcribe" in started

        gate.set()
        await asyncio.gather(*list(svc._inflight_tasks), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_lane_caps_are_independent(self, mock_variables):
        svc = _make(mock_variables, fanout_task_concurrency=1, document_task_concurrency=3)
        gate = asyncio.Event()
        running: Counter = Counter()

        async def handler(assignment):
            running[assignment.task_type] += 1
            await gate.wait()

        svc.set_task_assignment_handler(handler)
        for _ in range(5):
            await svc._on_task_assignment(_assignment("generate_tiers"))
        for _ in range(5):
            await svc._on_task_assignment(_assignment("document_embed"))
        await asyncio.sleep(0.05)

        assert running["generate_tiers"] == 1
        assert running["document_embed"] == 3

        gate.set()
        await asyncio.gather(*list(svc._inflight_tasks), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_zero_lane_cap_disables_that_lane_only(self, mock_variables):
        svc = _make(mock_variables, document_task_concurrency=0, fanout_task_concurrency=1)
        assert svc._lane_semaphores[TASK_LANE_DOCUMENT] is None
        assert svc._lane_semaphores[TASK_LANE_FANOUT] is not None

    def test_default_lane_reuses_the_legacy_knob(self, mock_variables):
        """MEMORYLAYER_TASK_CONCURRENCY keeps its meaning for unlaned work."""
        svc = _make(mock_variables, task_concurrency=5)
        assert svc._lane_semaphores[TASK_LANE_DEFAULT] is svc._task_semaphore
        assert svc._lane_concurrency[TASK_LANE_DEFAULT] == 5

    @pytest.mark.asyncio
    async def test_monitor_reports_running_and_waiting_per_lane(self, mock_variables):
        # A global cap would report one number and hide a saturated fan-out lane
        # sitting next to an idle document lane.
        svc = _make(mock_variables, fanout_task_concurrency=1, document_task_concurrency=4)
        svc.logger = MagicMock()
        svc._task_monitor_interval_s = 0.01
        gate = asyncio.Event()

        async def handler(_assignment):
            await gate.wait()

        svc.set_task_assignment_handler(handler)
        for _ in range(3):
            await svc._on_task_assignment(_assignment("auto_enrich"))
        await svc._on_task_assignment(_assignment("document_render"))

        mon = asyncio.create_task(svc._task_monitor_loop())
        await asyncio.sleep(0.05)
        mon.cancel()
        await asyncio.gather(mon, return_exceptions=True)

        lines = [
            str(c.args[0]) % tuple(c.args[1:])
            for c in svc.logger.info.call_args_list
            if c.args and "in flight" in str(c.args[0])
        ]
        assert lines, "monitor never logged an in-flight line"
        assert "fanout ~1/1 running, 2 waiting" in lines[0]
        assert "document ~1/4 running, 0 waiting" in lines[0]

        gate.set()
        await asyncio.gather(*list(svc._inflight_tasks), return_exceptions=True)
