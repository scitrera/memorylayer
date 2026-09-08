"""Unit tests for the batched + deterministic auto-association path.

These lock in the cost-reduction contract of ``AssociationService.auto_associate``:

1. All ambiguous candidates are classified in a SINGLE batched LLM call
   (``classify_relationships_batch``), never one call per candidate.
2. Near-duplicates (similarity >= duplicate_threshold) are labeled
   ``duplicate_of`` deterministically and are NOT sent to the LLM.
3. With the LLM classifier disabled, no classification call happens at all and
   every non-duplicate edge is ``similar_to``.
"""

import pytest

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.association.default import AssociationService
from memorylayer_server.services.memory import MemoryService


class _SpyOntology:
    """Minimal ontology stand-in that records how it was called.

    ``llm_service`` is a truthy sentinel so ``auto_associate`` treats the LLM
    path as available. ``classify_relationships_batch`` returns a caller-supplied
    mapping and counts its invocations.
    """

    def __init__(self, batch_result=None):
        self.llm_service = object()  # truthy -> enables use_llm
        self.batch_calls = 0
        self.single_calls = 0
        self.last_candidates = None
        self._batch_result = batch_result or {}

    async def classify_relationships_batch(self, content_a, candidates, tenant_id="_default", workspace_id=None):
        self.batch_calls += 1
        self.last_candidates = list(candidates)
        # Default anything not explicitly mapped to related_to (mirrors real impl).
        return {cand_id: self._batch_result.get(cand_id, "related_to") for cand_id, _ in candidates}

    async def classify_relationship(self, content_a, content_b, tenant_id="_default", workspace_id=None):
        self.single_calls += 1
        return "related_to"


async def _make_memories(memory_service: MemoryService, workspace_id: str, n: int, storage=None):
    ids = []
    for i in range(n):
        mem = await memory_service.remember(workspace_id, RememberInput(content=f"Batching test memory number {i}"))
        ids.append(mem.id)

    # remember() runs the post-store pipeline, which may auto-associate these
    # memories with each other before the test gets to call auto_associate
    # itself. auto_associate now SKIPS candidates it is already linked to (so a
    # re-enriched memory does not pay for a classification whose edges all
    # collide with uq_association), which would otherwise leave these tests
    # asserting on edges that were filtered out as already-present. Clear the
    # slate so each test exercises classification from a known state.
    if storage is not None:
        for mid in ids:
            for assoc in await storage.get_associations(workspace_id, mid, direction="outgoing"):
                await storage.delete_association(workspace_id, assoc.id)
    return ids


@pytest.mark.asyncio
async def test_single_batched_call_for_many_candidates(v, storage_backend, memory_service, workspace_id):
    """N ambiguous candidates -> exactly ONE batch call, zero per-pair calls."""
    ids = await _make_memories(memory_service, workspace_id, 4, storage=storage_backend)
    new_id, cand_ids = ids[0], ids[1:]

    spy = _SpyOntology(batch_result={cand_ids[0]: "causes", cand_ids[1]: "supports"})
    svc = AssociationService(storage=storage_backend, ontology_service=spy, v=v)

    # All candidates ambiguous (below duplicate threshold, above assoc threshold).
    similar = [(cid, 0.90) for cid in cand_ids]
    assocs = await svc.auto_associate(
        workspace_id=workspace_id,
        new_memory_id=new_id,
        similar_memories=similar,
        threshold=0.85,
        new_memory_content="anchor content",
    )

    assert spy.batch_calls == 1, "must classify the whole batch in a single call"
    assert spy.single_calls == 0, "must not fall back to per-pair classification"
    assert len(spy.last_candidates) == 3
    by_target = {a.target_id: a.relationship for a in assocs}
    assert by_target[cand_ids[0]] == "causes"
    assert by_target[cand_ids[1]] == "supports"
    assert by_target[cand_ids[2]] == "related_to"  # unmapped -> default


@pytest.mark.asyncio
async def test_near_duplicate_skips_llm(v, storage_backend, memory_service, workspace_id):
    """A candidate above the duplicate threshold is duplicate_of, not classified."""
    ids = await _make_memories(memory_service, workspace_id, 3, storage=storage_backend)
    new_id, dup_id, amb_id = ids

    spy = _SpyOntology(batch_result={amb_id: "refines"})
    svc = AssociationService(storage=storage_backend, ontology_service=spy, v=v)

    similar = [(dup_id, 0.99), (amb_id, 0.88)]  # one near-dup, one ambiguous
    assocs = await svc.auto_associate(
        workspace_id=workspace_id,
        new_memory_id=new_id,
        similar_memories=similar,
        threshold=0.85,
        new_memory_content="anchor content",
    )

    by_target = {a.target_id: a.relationship for a in assocs}
    assert by_target[dup_id] == "duplicate_of"
    assert by_target[amb_id] == "refines"
    # The near-duplicate must NOT have been sent to the classifier.
    assert spy.batch_calls == 1
    assert [cid for cid, _ in spy.last_candidates] == [amb_id]


@pytest.mark.asyncio
async def test_llm_disabled_defaults_to_similar_to(v, storage_backend, memory_service, workspace_id):
    """With classification disabled, no LLM call and every edge is similar_to."""
    from memorylayer_server.config import MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED

    v.set(MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED, False)

    ids = await _make_memories(memory_service, workspace_id, 3, storage=storage_backend)
    new_id, cand_ids = ids[0], ids[1:]

    spy = _SpyOntology(batch_result={cand_ids[0]: "causes"})
    svc = AssociationService(storage=storage_backend, ontology_service=spy, v=v)

    similar = [(cid, 0.90) for cid in cand_ids]
    assocs = await svc.auto_associate(
        workspace_id=workspace_id,
        new_memory_id=new_id,
        similar_memories=similar,
        threshold=0.85,
        new_memory_content="anchor content",
    )

    assert spy.batch_calls == 0
    assert spy.single_calls == 0
    assert all(a.relationship == "similar_to" for a in assocs)
