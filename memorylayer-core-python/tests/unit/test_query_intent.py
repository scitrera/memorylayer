"""Tests for rule-based query intent classification and recall routing."""

import uuid
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import RecallInput, RecallMode, RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.memory.query_intent import ENTITY, EVENT, GENERAL, TEMPORAL, classify_query_intent

# --- Pure classifier tests (deterministic, no framework) ---


def test_temporal_relative_cue():
    intent = classify_query_intent("what did we discuss recently")
    assert intent.has(TEMPORAL)
    assert intent.event_after is None and intent.event_before is None  # fuzzy -> no window


def test_temporal_explicit_year_window():
    intent = classify_query_intent("decisions in 2021")
    assert intent.has(TEMPORAL)
    assert intent.event_after == datetime(2021, 1, 1, tzinfo=UTC)
    assert datetime(2021, 12, 31, tzinfo=UTC) < intent.event_before < datetime(2022, 1, 1, tzinfo=UTC)


def test_temporal_month_name_window():
    intent = classify_query_intent("the launch in March 2021")
    assert intent.event_after == datetime(2021, 3, 1, tzinfo=UTC)
    assert datetime(2021, 3, 31, tzinfo=UTC) < intent.event_before < datetime(2021, 4, 1, tzinfo=UTC)


def test_event_intent():
    intent = classify_query_intent("notes from the standup meeting")
    assert intent.has(EVENT)


def test_entity_quoted_and_proper_noun():
    assert classify_query_intent('what about "Project Aurora"').has(ENTITY)
    assert classify_query_intent("tell me about Project Aurora").has(ENTITY)


def test_general_default_and_no_temporal_false_positive():
    intent = classify_query_intent("how to use the time module")
    assert intent.labels == {GENERAL}  # "time" must not trip temporal


# --- Routing integration tests ---


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"intent_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(Workspace(id=ws, tenant_id="intent_tenant", name="Intent Test", created_at=now, updated_at=now))
    return ws


async def _seed(memory_service, ws):
    # Distinct content so content-hash dedup does not collapse them into one.
    m2020 = await memory_service.remember(
        ws, RememberInput(content="project alpha milestone phase one", event_time=datetime(2020, 6, 1, tzinfo=UTC))
    )
    m2021 = await memory_service.remember(
        ws, RememberInput(content="project alpha milestone phase two", event_time=datetime(2021, 6, 1, tzinfo=UTC))
    )
    m2022 = await memory_service.remember(
        ws, RememberInput(content="project alpha milestone phase three", event_time=datetime(2022, 6, 1, tzinfo=UTC))
    )
    return m2020, m2021, m2022


@pytest.mark.asyncio
async def test_explicit_date_intent_filters_window(memory_service, iso_ws):
    _, m2021, _ = await _seed(memory_service, iso_ws)

    result = await memory_service.recall(
        iso_ws,
        RecallInput(
            query="what happened with project alpha in 2021",
            mode=RecallMode.RAG,
            min_relevance=0.0,
            include_associations=False,
            include_global=False,
            include_global_user=False,
        ),
    )

    assert [m.id for m in result.memories] == [m2021.id]  # explicit-year window applied
    assert result.query_intent is not None and TEMPORAL in result.query_intent


@pytest.mark.asyncio
async def test_soft_intent_without_date_does_not_filter(memory_service, iso_ws):
    m2020, m2021, m2022 = await _seed(memory_service, iso_ws)

    result = await memory_service.recall(
        iso_ws,
        RecallInput(
            query="what happened with project alpha",  # event cue, no explicit date
            mode=RecallMode.RAG,
            min_relevance=0.0,
            include_associations=False,
            include_global=False,
            include_global_user=False,
        ),
    )

    # No date -> no window -> all three retained (soft routing never drops results).
    assert {m.id for m in result.memories} == {m2020.id, m2021.id, m2022.id}
    assert EVENT in (result.query_intent or [])
