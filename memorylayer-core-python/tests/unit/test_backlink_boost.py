"""Deterministic tests for backlink-salience (graph in-degree) recall boosting.

Memories are created directly via the storage backend (bypassing the write-path
auto-enrichment) and associations are added explicitly, so in-degree is exact
and the resulting score multipliers can be checked to the digit. Each test runs
in its own isolated workspace so it cannot perturb other tests' recall results.
"""

import math
import uuid
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import MemoryType, RememberInput
from memorylayer_server.models.workspace import Workspace


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    """A freshly-created, unique workspace for full test isolation."""
    ws = f"backlink_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="backlink_tenant", name="Backlink Test", created_at=now, updated_at=now)
    )
    return ws


async def _make(storage_backend, workspace_id: str, content: str):
    return await storage_backend.create_memory(workspace_id, RememberInput(content=content, type=MemoryType.SEMANTIC))


@pytest.mark.asyncio
async def test_backlink_boost_promotes_hub(memory_service, storage_backend, iso_ws):
    ws = iso_ws
    hub = await _make(storage_backend, ws, "backlink hub memory alpha")
    spoke1 = await _make(storage_backend, ws, "backlink spoke one beta")
    spoke2 = await _make(storage_backend, ws, "backlink spoke two gamma")

    # Two incoming edges to the hub; spokes have in-degree 0.
    await storage_backend.create_association(ws, AssociateInput(source_id=spoke1.id, target_id=hub.id, relationship="related_to"))
    await storage_backend.create_association(ws, AssociateInput(source_id=spoke2.id, target_id=hub.id, relationship="related_to"))

    # Equal base scores, hub deliberately last so ordering must come from the boost.
    for m in (hub, spoke1, spoke2):
        m.boosted_score = 0.5
    result = await memory_service.apply_backlink_boost(ws, [spoke1, spoke2, hub], weight=0.2)

    assert result[0].id == hub.id
    by_id = {m.id: m for m in result}
    assert abs(by_id[hub.id].boosted_score - 0.5 * (1.0 + 0.2 * math.log1p(2))) < 1e-9
    # Spokes (in-degree 0) are unchanged.
    assert by_id[spoke1.id].boosted_score == 0.5
    assert by_id[spoke2.id].boosted_score == 0.5


@pytest.mark.asyncio
async def test_backlink_boost_noop_without_edges(memory_service, storage_backend, iso_ws):
    ws = iso_ws
    a = await _make(storage_backend, ws, "lonely memory one no links")
    b = await _make(storage_backend, ws, "lonely memory two no links")
    a.boosted_score, b.boosted_score = 0.7, 0.3
    result = await memory_service.apply_backlink_boost(ws, [a, b], weight=0.2)
    assert [m.boosted_score for m in result] == [0.7, 0.3]


@pytest.mark.asyncio
async def test_backlink_boost_disabled_when_weight_zero(memory_service, storage_backend, iso_ws):
    ws = iso_ws
    a = await _make(storage_backend, ws, "zero weight hub delta")
    b = await _make(storage_backend, ws, "zero weight spoke epsilon")
    await storage_backend.create_association(ws, AssociateInput(source_id=b.id, target_id=a.id, relationship="related_to"))

    a.boosted_score, b.boosted_score = 0.4, 0.6
    result = await memory_service.apply_backlink_boost(ws, [a, b], weight=0.0)
    # weight=0 short-circuits: order and scores untouched.
    assert [m.id for m in result] == [a.id, b.id]
    assert [m.boosted_score for m in result] == [0.4, 0.6]


# ---------------------------------------------------------------------------
# similar_to exclusion tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_similar_to_edges_excluded_from_indegree(memory_service, storage_backend, iso_ws):
    """similar_to in-edges must NOT contribute to in-degree.

    The 'hub' here is generic (many similar_to back-pointers) but the 'gold'
    memory is the specific, relevant result.  With the old behaviour (similar_to
    counted), hub would be over-boosted and displace gold at rank 0.  The fix
    excludes similar_to so hub's effective in-degree stays 0 and gold keeps its
    higher base score.
    """
    ws = iso_ws

    gold = await _make(storage_backend, ws, "gold specific answer zeta")
    hub = await _make(storage_backend, ws, "generic hub many similar eta")
    # Simulate KNN auto-association: three 'similar_to' edges pointing at hub.
    for _ in range(3):
        spoke = await _make(storage_backend, ws, f"similar spoke {uuid.uuid4().hex[:4]}")
        await storage_backend.create_association(
            ws, AssociateInput(source_id=spoke.id, target_id=hub.id, relationship="similar_to")
        )

    # Gold has a slightly higher base score than hub.
    gold.boosted_score = 0.6
    hub.boosted_score = 0.5

    result = await memory_service.apply_backlink_boost(ws, [gold, hub], weight=0.3)

    # similar_to edges are excluded, so hub's in-degree == 0 and its score is
    # unchanged; gold must remain rank-0.
    by_id = {m.id: m for m in result}
    assert result[0].id == gold.id, "gold should still rank first when similar_to edges are excluded"
    assert abs(by_id[hub.id].boosted_score - 0.5) < 1e-9, "hub score must be unchanged (in-degree==0 after similar_to exclusion)"
    assert abs(by_id[gold.id].boosted_score - 0.6) < 1e-9, "gold score must be unchanged (no incoming edges)"


@pytest.mark.asyncio
async def test_meaningful_edges_still_boost_with_similar_to_present(memory_service, storage_backend, iso_ws):
    """Non-similar_to edges still count even when similar_to edges are also present.

    A hub has both similar_to and related_to incoming edges.  Only the
    related_to edge should be counted, yielding in-degree == 1.
    """
    ws = iso_ws

    hub = await _make(storage_backend, ws, "mixed edge hub theta")
    spoke_similar = await _make(storage_backend, ws, "similar spoke iota")
    spoke_related = await _make(storage_backend, ws, "related spoke kappa")

    await storage_backend.create_association(
        ws, AssociateInput(source_id=spoke_similar.id, target_id=hub.id, relationship="similar_to")
    )
    await storage_backend.create_association(
        ws, AssociateInput(source_id=spoke_related.id, target_id=hub.id, relationship="related_to")
    )

    hub.boosted_score = 0.5
    result = await memory_service.apply_backlink_boost(ws, [hub], weight=0.2)

    # Effective in-degree == 1 (only related_to counted).
    expected = 0.5 * (1.0 + 0.2 * math.log1p(1))
    assert abs(result[0].boosted_score - expected) < 1e-9, (
        f"Expected boost from 1 meaningful edge ({expected:.6f}), got {result[0].boosted_score:.6f}"
    )


@pytest.mark.asyncio
async def test_default_backlink_weight_is_zero(memory_service, storage_backend, iso_ws):
    """DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT must be 0.0 (boost off by default).

    Verifies that the service-level attribute initialised from the default
    config is 0.0 so that similar_to-dense graphs cannot demote specific results
    out of the box.
    """
    from memorylayer_server.services.memory.base import DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT

    assert DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT == 0.0, (
        "Default backlink boost weight must be 0.0 to avoid anti-relevance boosting "
        "on similar_to-dense KNN graphs; set MEMORYLAYER_BACKLINK_BOOST_WEIGHT env var to opt in."
    )
    # The memory_service fixture uses the default config, so its attribute must
    # also be 0.0.
    assert memory_service.backlink_boost_weight == 0.0
