"""Unit tests for ColPali GPU concurrency control.

Exercises the asyncio.Semaphore guard around the GPU-bound methods plus
the ``ColPaliQueueTimeoutError`` / 503 path. We don't need torch / a
real model — we monkey-patch ``_get_model`` to short-circuit the GPU
call and just yield control inside the slot so the semaphore is the
only thing being tested.

Skipped at module load if ``colpali_engine`` is not importable (the
module-level import would fail on ``_load_*`` helpers otherwise — but
those are lazy, so the file imports without colpali_engine).
"""
from __future__ import annotations

import asyncio

import pytest

from memorylayer_embed_server.services.embedding.colpali import (
    ColPaliEmbeddingProvider,
    ColPaliQueueTimeoutError,
)


def _make_provider(*, max_concurrent: int, queue_timeout_sec: float = 0.0) -> ColPaliEmbeddingProvider:
    """ColPali provider with a stubbed model — semaphore is the only real thing."""
    p = ColPaliEmbeddingProvider(
        v=None,
        model_name="test/mock",
        device="cpu",
        revision="main",
        max_concurrent=max_concurrent,
        queue_timeout_sec=queue_timeout_sec,
    )
    # Stub _get_model — _gpu_slot is the unit under test.
    p._get_model = lambda: (object(), object())
    return p


async def _hold_slot(provider: ColPaliEmbeddingProvider, hold_s: float) -> float:
    """Acquire a GPU slot and sleep ``hold_s`` seconds. Returns wall time spent inside."""
    import time

    start = time.monotonic()
    async with provider._gpu_slot():
        await asyncio.sleep(hold_s)
    return time.monotonic() - start


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


async def test_semaphore_caps_concurrent_holders():
    """With max_concurrent=2, the 3rd request must wait for one to release."""
    p = _make_provider(max_concurrent=2)
    durations = await asyncio.gather(*[_hold_slot(p, 0.05) for _ in range(3)])
    # First two finish at ~0.05s. Third had to wait for one to release,
    # so its measured time is ~0.10s.
    durations.sort()
    assert durations[0] < 0.07
    assert durations[1] < 0.07
    assert durations[2] > 0.09


async def test_semaphore_max_concurrent_one_serializes():
    """max_concurrent=1 fully serializes requests."""
    p = _make_provider(max_concurrent=1)
    durations = await asyncio.gather(*[_hold_slot(p, 0.04) for _ in range(3)])
    durations.sort()
    assert durations[0] < 0.06
    assert durations[1] > 0.07
    assert durations[2] > 0.11


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


async def test_queue_timeout_zero_means_wait_forever():
    """Default timeout=0 → no timeout; slow holders just queue."""
    p = _make_provider(max_concurrent=1, queue_timeout_sec=0.0)

    async def waiter():
        async with p._gpu_slot():
            await asyncio.sleep(0.15)

    # Run two waiters; second waits ~150ms for the first.
    await asyncio.gather(waiter(), waiter())  # no exception


async def test_queue_timeout_raises_when_exceeded():
    """Positive timeout → ColPaliQueueTimeoutError when wait exceeds it."""
    p = _make_provider(max_concurrent=1, queue_timeout_sec=0.05)

    async def hold(hold_s: float):
        async with p._gpu_slot():
            await asyncio.sleep(hold_s)

    async def attempt():
        async with p._gpu_slot():
            pass

    # First request holds the slot for 200ms; second tries to acquire
    # and should bail out after 50ms.
    holder = asyncio.create_task(hold(0.2))
    await asyncio.sleep(0.01)  # ensure holder grabs first
    with pytest.raises(ColPaliQueueTimeoutError) as exc:
        await attempt()
    assert exc.value.max_concurrent == 1
    assert exc.value.wait_seconds >= 0.04
    await holder


async def test_queue_timeout_passes_through_when_slot_immediately_available():
    """If a slot is free, the timeout never kicks in."""
    p = _make_provider(max_concurrent=2, queue_timeout_sec=0.01)
    # Two concurrent waiters with max_concurrent=2 — both acquire immediately.
    await asyncio.gather(_hold_slot(p, 0.0), _hold_slot(p, 0.0))


# ---------------------------------------------------------------------------
# Cap normalization
# ---------------------------------------------------------------------------


def test_max_concurrent_is_clamped_to_at_least_one():
    p = ColPaliEmbeddingProvider(v=None, max_concurrent=0)
    assert p.max_concurrent == 1
    p = ColPaliEmbeddingProvider(v=None, max_concurrent=-5)
    assert p.max_concurrent == 1


def test_queue_timeout_is_clamped_to_non_negative():
    p = ColPaliEmbeddingProvider(v=None, queue_timeout_sec=-3.0)
    assert p.queue_timeout_sec == 0.0
