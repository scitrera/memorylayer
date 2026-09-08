"""auto_associate must not re-do work it has already done.

Enrichment re-runs for a memory that was enriched before -- a merge folds a new
fact into it, or a task is redelivered. Every re-run used to pay for a full
batched relationship classification and then fail every insert on
uq_association, logging a warning per edge and creating nothing:

    Failed to auto-associate mem_X with mem_Y: duplicate key value violates
    unique constraint "uq_association"
    ...
    Created 0 auto-associations for memory: mem_X (4 LLM-classified)

The four LLM calls are the part that matters. Skipping already-linked
candidates BEFORE the classification pass is what makes the re-run cheap rather
than merely quiet.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.association.default import AssociationService


def _service(existing_targets=(), *, classify=None):
    storage = MagicMock()
    storage.get_associations = AsyncMock(
        return_value=[SimpleNamespace(target_id=t) for t in existing_targets]
    )
    storage.get_memory = AsyncMock(
        return_value=SimpleNamespace(id="c", content="candidate content")
    )
    storage.create_association = AsyncMock(
        side_effect=lambda ws, inp: SimpleNamespace(
            source_id=inp.source_id, target_id=inp.target_id,
        )
    )

    ontology = MagicMock()
    ontology.llm_service = MagicMock()
    ontology.classify_relationships_batch = AsyncMock(
        return_value=classify if classify is not None else {}
    )

    svc = AssociationService.__new__(AssociationService)
    svc.storage = storage
    svc.ontology_service = ontology
    svc.logger = logging.getLogger("test-assoc-idempotence")
    svc.auto_association_threshold = 0.5
    svc.duplicate_threshold = 0.97
    svc.llm_classify_enabled = True
    return svc


@pytest.mark.asyncio
async def test_an_already_linked_candidate_is_skipped_entirely():
    svc = _service(existing_targets=["mem_b"])

    created = await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78)], new_memory_content="content",
    )

    assert created == []
    svc.storage.create_association.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_llm_is_not_called_for_already_linked_candidates():
    """The expensive half. Skipping only at insert time would still burn the
    classification, which is what the production logs showed."""
    svc = _service(existing_targets=["mem_b", "mem_c"])

    await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78), ("mem_c", 0.66)], new_memory_content="content",
    )

    svc.ontology_service.classify_relationships_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_candidates_are_still_associated():
    svc = _service(existing_targets=["mem_b"])

    created = await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78), ("mem_new", 0.72)], new_memory_content="content",
    )

    assert [a.target_id for a in created] == ["mem_new"]


@pytest.mark.asyncio
async def test_a_mixed_batch_classifies_only_the_new_candidate():
    svc = _service(existing_targets=["mem_b"])

    await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78), ("mem_new", 0.72)], new_memory_content="content",
    )

    candidates = svc.ontology_service.classify_relationships_batch.await_args.kwargs
    assert [cid for cid, _ in candidates["candidates"]] == ["mem_new"]


@pytest.mark.asyncio
async def test_a_first_run_is_unaffected():
    svc = _service(existing_targets=[])

    created = await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78)], new_memory_content="content",
    )

    assert [a.target_id for a in created] == ["mem_b"]


@pytest.mark.asyncio
async def test_a_failure_loading_existing_edges_degrades_to_the_old_behaviour():
    # A miss costs a duplicate insert, not correctness -- it must not abort the
    # association pass.
    svc = _service(existing_targets=[])
    svc.storage.get_associations.side_effect = RuntimeError("db blip")

    created = await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78)], new_memory_content="content",
    )

    assert [a.target_id for a in created] == ["mem_b"]


@pytest.mark.asyncio
async def test_only_outgoing_edges_are_consulted():
    # uq_association is on (source, target, relationship); an INCOMING edge from
    # the candidate does not block an outgoing one, so treating it as "linked"
    # would silently drop real associations.
    svc = _service(existing_targets=[])

    await svc.auto_associate(
        "ws", "mem_a", [("mem_b", 0.78)], new_memory_content="content",
    )

    assert svc.storage.get_associations.await_args.kwargs["direction"] == "outgoing"
