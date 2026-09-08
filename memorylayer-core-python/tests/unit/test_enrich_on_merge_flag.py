"""Merge-triggered enrichment is opt-in.

When an incoming fact MERGES into an existing memory, ingest_fact used to
re-run the whole post-store pipeline on the survivor. That memory was already
enriched when it was first stored, so the work is overwhelmingly re-derivation
of what exists -- and merges are not rare: decomposing many similar pages
produces many similar facts, so it fired continuously.

Off by default. The flag exists because there IS an argument for running it
(the survivor's content changed, so its tiers and associations are arguably
stale) -- it just does not outweigh the cost today.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memorylayer_server.config import DEFAULT_MEMORYLAYER_ENRICH_ON_MERGE
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.deduplication.base import DeduplicationAction


def _force_merge(service, monkeypatch):
    """Make the next ingest_fact take the MERGE branch, and record enrichment."""
    merged = SimpleNamespace(id="mem_survivor", embedding=[0.1, 0.2, 0.3])

    # monkeypatch, not direct assignment: `memory_service` is a shared fixture,
    # and stubbing its collaborators permanently leaks a fake dedup/storage into
    # every later test in the session.
    monkeypatch.setattr(
        service.deduplication, "check_duplicate",
        AsyncMock(return_value=SimpleNamespace(
            action=DeduplicationAction.MERGE,
            existing_memory_id="mem_survivor",
            reason="Potential merge candidate (similarity: 0.857)",
        )),
    )
    monkeypatch.setattr(service.storage, "get_memory", AsyncMock(return_value=merged))
    monkeypatch.setattr(service, "_merge_memories", AsyncMock(return_value=merged))
    pipeline = AsyncMock()
    monkeypatch.setattr(service, "_post_store_pipeline", pipeline)
    return pipeline


def test_the_default_is_off():
    assert DEFAULT_MEMORYLAYER_ENRICH_ON_MERGE is False


@pytest.mark.asyncio
async def test_a_merge_does_not_re_enrich_by_default(memory_service, monkeypatch, workspace_id):
    pipeline = _force_merge(memory_service, monkeypatch)
    memory_service.enrich_on_merge = False

    result = await memory_service.ingest_fact(
        workspace_id=workspace_id, input=RememberInput(content="a merged fact"),
    )

    pipeline.assert_not_awaited()
    assert result.id == "mem_survivor"  # the merge itself still happens


@pytest.mark.asyncio
async def test_enabling_the_flag_restores_re_enrichment(memory_service, monkeypatch, workspace_id):
    pipeline = _force_merge(memory_service, monkeypatch)
    memory_service.enrich_on_merge = True

    await memory_service.ingest_fact(
        workspace_id=workspace_id, input=RememberInput(content="a merged fact"),
    )

    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_flag_does_not_affect_newly_created_memories(
    memory_service, monkeypatch, workspace_id,
):
    """Only the merge branch is gated -- a genuinely new fact must still be
    enriched, or decomposition would stop producing associations entirely."""
    monkeypatch.setattr(
        memory_service.deduplication, "check_duplicate",
        AsyncMock(return_value=SimpleNamespace(
            action=DeduplicationAction.CREATE, existing_memory_id=None, reason="new",
        )),
    )
    pipeline = AsyncMock()
    monkeypatch.setattr(memory_service, "_post_store_pipeline", pipeline)
    monkeypatch.setattr(memory_service, "enqueue_post_store", AsyncMock())
    memory_service.enrich_on_merge = False

    await memory_service.ingest_fact(
        workspace_id=workspace_id, input=RememberInput(content="a brand new fact"),
    )

    # Either seam is acceptable; what matters is that enrichment was dispatched.
    assert pipeline.await_count + memory_service.enqueue_post_store.await_count >= 1
