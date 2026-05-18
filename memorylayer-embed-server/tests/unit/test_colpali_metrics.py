"""Unit tests for ColPali metrics emission + load snapshot.

These cover the OTel-compatible instrumentation paths added so an
upstream LB can route to the least-loaded embed-server replica:
  - Counter / histogram emitted on slot acquire
  - Counter / histogram emitted on queue timeout
  - Gauge updated on acquire and release
  - ``get_load_snapshot()`` reflects current in-flight count
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from memorylayer_embed_server.services.embedding.colpali import (
    ColPaliEmbeddingProvider,
    ColPaliQueueTimeoutError,
)


class _StubMetrics:
    """Records every call so tests can assert exactly what was emitted."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, float, dict | None]] = []
        self.histograms: list[tuple[str, float, dict | None]] = []
        self.gauges: list[tuple[str, float, dict | None]] = []

    def counter(self, name, value=1, labels=None):
        self.counters.append((name, value, dict(labels) if labels else None))

    def histogram(self, name, value, labels=None):
        self.histograms.append((name, value, dict(labels) if labels else None))

    def gauge(self, name, value, labels=None):
        self.gauges.append((name, value, dict(labels) if labels else None))


def _make_provider(stub_metrics: _StubMetrics, *, max_concurrent=2, queue_timeout_sec=0.0) -> ColPaliEmbeddingProvider:
    p = ColPaliEmbeddingProvider(
        v=None, model_name="test/mock", device="cpu",
        max_concurrent=max_concurrent, queue_timeout_sec=queue_timeout_sec,
    )
    p._get_metrics = lambda: stub_metrics
    p._get_model = lambda: (object(), object())
    return p


async def _hold(provider: ColPaliEmbeddingProvider, hold_s: float):
    async with provider._gpu_slot():
        await asyncio.sleep(hold_s)


async def test_metrics_emit_on_acquire_and_release():
    m = _StubMetrics()
    p = _make_provider(m, max_concurrent=2)
    async with p._gpu_slot():
        pass

    # Counter: one acquire
    acquire_counters = [c for c in m.counters if c[0] == "embed_server_colpali_gpu_slot_total"
                        and c[2] == {"result": "acquired"}]
    assert len(acquire_counters) == 1

    # Histogram: one wait_seconds sample, labelled acquired, value >= 0
    acquire_hists = [h for h in m.histograms if h[0] == "embed_server_colpali_gpu_slot_wait_seconds"
                     and h[2] == {"result": "acquired"}]
    assert len(acquire_hists) == 1 and acquire_hists[0][1] >= 0.0

    # Gauges: in_flight = 1 at entry, = 0 at exit
    in_flight_gauges = [g for g in m.gauges if g[0] == "embed_server_colpali_gpu_in_flight"]
    assert in_flight_gauges[0][1] == 1.0
    assert in_flight_gauges[-1][1] == 0.0

    # Utilization gauges follow the same shape
    util_gauges = [g for g in m.gauges if g[0] == "embed_server_colpali_gpu_utilization"]
    assert util_gauges[0][1] == pytest.approx(0.5)
    assert util_gauges[-1][1] == 0.0


async def test_metrics_emit_on_queue_timeout():
    m = _StubMetrics()
    p = _make_provider(m, max_concurrent=1, queue_timeout_sec=0.05)

    holder = asyncio.create_task(_hold(p, 0.2))
    await asyncio.sleep(0.01)  # ensure holder grabs the slot first

    with pytest.raises(ColPaliQueueTimeoutError):
        async with p._gpu_slot():
            pass

    timeout_counters = [c for c in m.counters if c[0] == "embed_server_colpali_gpu_slot_total"
                        and c[2] == {"result": "timeout"}]
    timeout_hists = [h for h in m.histograms if h[0] == "embed_server_colpali_gpu_slot_wait_seconds"
                     and h[2] == {"result": "timeout"}]
    assert len(timeout_counters) == 1
    assert len(timeout_hists) == 1 and timeout_hists[0][1] >= 0.04

    await holder


def test_load_snapshot_initial_state():
    p = ColPaliEmbeddingProvider(v=None, max_concurrent=4)
    snap = p.get_load_snapshot()
    assert snap == {"in_flight": 0, "max_concurrent": 4, "utilization": 0.0}


async def test_load_snapshot_tracks_in_flight():
    p = ColPaliEmbeddingProvider(v=None, max_concurrent=4)
    p._get_metrics = lambda: None  # disable metrics emission

    snapshots: list[dict] = []

    async def worker():
        async with p._gpu_slot():
            snapshots.append(p.get_load_snapshot())
            await asyncio.sleep(0.02)

    await asyncio.gather(worker(), worker(), worker())
    # Inside the slot, in_flight was always >= 1 and <= max_concurrent
    for s in snapshots:
        assert 1 <= s["in_flight"] <= 4
        assert 0.25 <= s["utilization"] <= 1.0
    # After the workers finish, snapshot returns to zero
    final = p.get_load_snapshot()
    assert final["in_flight"] == 0
    assert final["utilization"] == 0.0


async def test_metrics_skipped_when_no_service_registered():
    """Missing MetricsService is a non-fatal path — slot still works."""
    p = ColPaliEmbeddingProvider(v=None, max_concurrent=2)
    p._get_metrics = lambda: None  # explicit no-op
    async with p._gpu_slot():
        pass
    # No exception means the metrics-free path is well-formed.
