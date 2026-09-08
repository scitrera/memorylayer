"""Tests for the timeline/temporal index: event_time persistence, get_timeline
ordering/windowing, temporal_neighbors, and recall temporal filter/order.

Isolated workspace per test; event times are timezone-aware UTC so round-trips
compare exactly.
"""

import uuid
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import RecallInput, RecallMode, RememberInput
from memorylayer_server.models.workspace import Workspace

T1 = datetime(2020, 1, 1, tzinfo=UTC)
T2 = datetime(2021, 1, 1, tzinfo=UTC)
T3 = datetime(2022, 1, 1, tzinfo=UTC)


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"temporal_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="temporal_tenant", name="Temporal Test", created_at=now, updated_at=now)
    )
    return ws


@pytest.mark.asyncio
async def test_event_time_persists_round_trip(memory_service, iso_ws):
    m = await memory_service.remember(iso_ws, RememberInput(content="quarterly board meeting notes", event_time=T2))
    fetched = await memory_service.get(iso_ws, m.id)
    assert fetched.event_time == T2


@pytest.mark.asyncio
async def test_get_timeline_orders_and_windows(memory_service, storage_backend, iso_ws):
    m1 = await memory_service.remember(iso_ws, RememberInput(content="event alpha one", event_time=T1))
    m2 = await memory_service.remember(iso_ws, RememberInput(content="event alpha two", event_time=T2))
    m3 = await memory_service.remember(iso_ws, RememberInput(content="event alpha three", event_time=T3))

    asc = await memory_service.get_timeline(iso_ws, ascending=True)
    assert [m.id for m in asc] == [m1.id, m2.id, m3.id]

    desc = await memory_service.get_timeline(iso_ws, ascending=False)
    assert [m.id for m in desc] == [m3.id, m2.id, m1.id]

    # Inclusive window bounds on effective event time.
    after_t2 = await memory_service.get_timeline(iso_ws, event_after=T2, ascending=True)
    assert [m.id for m in after_t2] == [m2.id, m3.id]

    before_t2 = await memory_service.get_timeline(iso_ws, event_before=T2, ascending=True)
    assert [m.id for m in before_t2] == [m1.id, m2.id]


@pytest.mark.asyncio
async def test_undated_memory_falls_back_to_created_at(memory_service, iso_ws):
    # A dated past memory and an undated one (effective = created_at = ~now).
    past = await memory_service.remember(iso_ws, RememberInput(content="old dated event", event_time=T1))
    undated = await memory_service.remember(iso_ws, RememberInput(content="undated recent note"))

    asc = await memory_service.get_timeline(iso_ws, ascending=True)
    # Past dated memory sorts before the just-created undated one.
    assert [m.id for m in asc] == [past.id, undated.id]


@pytest.mark.asyncio
async def test_temporal_neighbors(memory_service, iso_ws):
    m1 = await memory_service.remember(iso_ws, RememberInput(content="neighbor one", event_time=T1))
    m2 = await memory_service.remember(iso_ws, RememberInput(content="neighbor two", event_time=T2))
    m3 = await memory_service.remember(iso_ws, RememberInput(content="neighbor three", event_time=T3))

    result = await memory_service.temporal_neighbors(iso_ws, m2.id, limit=5)
    assert result["anchor"].id == m2.id
    assert [m.id for m in result["before"]] == [m1.id]  # closest earlier
    assert [m.id for m in result["after"]] == [m3.id]  # closest later
    # Anchor itself is excluded from both lists.
    assert m2.id not in {m.id for m in result["before"] + result["after"]}


@pytest.mark.asyncio
async def test_recall_temporal_filter_and_order(memory_service, iso_ws):
    await memory_service.remember(iso_ws, RememberInput(content="timeline alpha first", event_time=T1))
    m2 = await memory_service.remember(iso_ws, RememberInput(content="timeline alpha second", event_time=T2))
    m3 = await memory_service.remember(iso_ws, RememberInput(content="timeline alpha third", event_time=T3))

    result = await memory_service.recall(
        iso_ws,
        RecallInput(
            query="timeline alpha",
            mode=RecallMode.RAG,
            min_relevance=0.0,
            include_associations=False,
            include_global=False,  # isolate from _global memories created by other tests
            include_global_user=False,
            event_after=T2,
            time_order="asc",
        ),
    )

    ids = [m.id for m in result.memories]
    # Only T2/T3 survive the window, ordered ascending by event time.
    assert ids == [m2.id, m3.id]
