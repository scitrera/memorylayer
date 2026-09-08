"""PART_OF association writes from fact decomposition.

Decomposition routes every extracted fact through ``ingest_fact``, which MERGES
a fact into an existing memory when one is similar enough. Two distinct facts
from the same parent therefore routinely come back as the SAME memory id, and
writing one association per fact then inserts an identical
``(source_id, target_id, 'part_of')`` row twice -- a uq_association violation
logged once per collision.

The merge behaviour is correct; the association write has to account for it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.models.memory import Memory, MemoryStatus, MemoryType
from memorylayer_server.services._constants import (
    EXT_EXTRACTION_SERVICE,
    EXT_MEMORY_SERVICE,
    EXT_STORAGE_BACKEND,
)
from memorylayer_server.tasks.fact_decomposition_handler import FactDecompositionTaskHandler

PARENT_ID = "mem_parent"
WORKSPACE = "ws_test"


def _parent_memory() -> Memory:
    return Memory(
        id=PARENT_ID,
        workspace_id=WORKSPACE,
        tenant_id="_default",
        content="A composite paragraph with several facts in it.",
        content_hash="0" * 64,
        type=MemoryType.SEMANTIC,
        status=MemoryStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
    )


def _make_handler(ingested_ids: list[str]):
    """Handler wired to mocks, returning the ids ingest_fact will hand back."""
    storage = MagicMock()
    storage.get_memory = AsyncMock(return_value=_parent_memory())
    storage.create_association = AsyncMock()
    storage.update_memory = AsyncMock()

    extraction = MagicMock()
    extraction.decompose_to_facts = AsyncMock(
        return_value=[{"content": "fact %d" % i} for i in range(len(ingested_ids))]
    )

    memory_service = MagicMock()
    memory_service.ingest_fact = AsyncMock(
        side_effect=[SimpleNamespace(id=mid) for mid in ingested_ids]
    )

    ext_map = {
        EXT_STORAGE_BACKEND: storage,
        EXT_EXTRACTION_SERVICE: extraction,
        EXT_MEMORY_SERVICE: memory_service,
    }
    handler = FactDecompositionTaskHandler()
    handler.get_extension = lambda key, v: ext_map[key]
    return handler, storage


def _associated_pairs(storage) -> list[tuple[str, str]]:
    return [
        (call.args[1].source_id, call.args[1].target_id)
        for call in storage.create_association.await_args_list
    ]


@pytest.mark.asyncio
async def test_one_association_per_unique_fact():
    handler, storage = _make_handler(["mem_a", "mem_b", "mem_c"])

    await handler.handle(MagicMock(), {"memory_id": PARENT_ID, "workspace_id": WORKSPACE})

    assert _associated_pairs(storage) == [
        ("mem_a", PARENT_ID), ("mem_b", PARENT_ID), ("mem_c", PARENT_ID),
    ]


@pytest.mark.asyncio
async def test_facts_merging_into_the_same_memory_associate_once():
    """The observed uq_association violation.

    Two facts merge into mem_dup; without collapsing the ids the identical row
    is inserted twice and Postgres rejects the second.
    """
    handler, storage = _make_handler(["mem_dup", "mem_other", "mem_dup"])

    await handler.handle(MagicMock(), {"memory_id": PARENT_ID, "workspace_id": WORKSPACE})

    pairs = _associated_pairs(storage)
    assert pairs == [("mem_dup", PARENT_ID), ("mem_other", PARENT_ID)]
    assert len(pairs) == len(set(pairs))


@pytest.mark.asyncio
async def test_a_fact_merging_into_the_parent_does_not_self_associate():
    """A fact similar enough to merge into its OWN parent would otherwise write
    a part_of edge from the parent to itself."""
    handler, storage = _make_handler(["mem_a", PARENT_ID])

    await handler.handle(MagicMock(), {"memory_id": PARENT_ID, "workspace_id": WORKSPACE})

    pairs = _associated_pairs(storage)
    assert pairs == [("mem_a", PARENT_ID)]
    assert all(src != tgt for src, tgt in pairs)


@pytest.mark.asyncio
async def test_association_failures_do_not_prevent_archiving_the_parent():
    """Decomposition has already happened by then; leaving the parent ACTIVE
    would let it be decomposed again on the next sweep."""
    handler, storage = _make_handler(["mem_a", "mem_b"])
    storage.create_association.side_effect = RuntimeError("unique violation")

    await handler.handle(MagicMock(), {"memory_id": PARENT_ID, "workspace_id": WORKSPACE})

    storage.update_memory.assert_awaited_once()
    assert storage.update_memory.await_args.kwargs["status"] == MemoryStatus.ARCHIVED.value


@pytest.mark.asyncio
async def test_one_failing_fact_does_not_abandon_the_rest():
    """A lost CAS race on one fact must not take the batch down.

    Facts from concurrent decompositions routinely converge on the same merge
    target, so `ETag does not match current revision` is expected under load.
    Letting it propagate abandoned every remaining fact and skipped the archive,
    leaving the parent ACTIVE and factless for doc_verify to re-drive into the
    same collision.
    """
    handler, storage = _make_handler(["mem_a", "mem_b", "mem_c"])
    memory_service_ids = ["mem_a", "mem_b", "mem_c"]
    calls = {"n": 0}

    async def flaky_ingest(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        if i == 1:
            raise RuntimeError("ETag does not match current revision")
        return SimpleNamespace(id=memory_service_ids[i])

    ext = handler.get_extension(EXT_MEMORY_SERVICE, None)
    ext.ingest_fact = AsyncMock(side_effect=flaky_ingest)

    await handler.handle(MagicMock(), {"memory_id": PARENT_ID, "workspace_id": WORKSPACE})

    # All three facts attempted, the survivors associated, parent still archived.
    assert calls["n"] == 3
    assert _associated_pairs(storage) == [("mem_a", PARENT_ID), ("mem_c", PARENT_ID)]
    storage.update_memory.assert_awaited_once()
