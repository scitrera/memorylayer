"""Unit tests for query-intent-driven recall channel selection (P4.1).

The intent-channel selector (`_select_channels` + `_INTENT_CHANNEL_MATRIX`) gates
WHICH of the already-enabled fuse arms fire for a given query's intent, instead of
every enabled arm firing unconditionally. It is flag-gated
(`MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED`, default OFF) and is a strict no-op
when off.

Two layers are tested:

1. `_select_channels` in isolation (pure, deterministic over intent labels):
   - flag OFF  -> always the full default set (no-op)
   - flag ON   -> exactly the matrix's set per intent; UNION across multi-label
   - safe-degrade: None / empty / unknown label -> full default set

2. End-to-end through `recall(...)`, spying on the fuse methods to capture the
   actual arm-firing set. This proves the kill-switch precedence (a channel whose
   own flag is OFF never fires even if the matrix selects it) and that the
   flag-OFF path is byte-identical to today across intents.

No LoCoMo / benchmark run is involved — these are deterministic unit tests.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import (
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemorySubtype,
    MemoryType,
    RecallInput,
    RecallMode,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory.base import (
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
    MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_QUERY_INTENT_ENABLED,
)
from memorylayer_server.services.memory.default import (
    _CHANNEL_DENSE,
    _CHANNEL_ENTITY,
    _CHANNEL_FACT,
    _CHANNEL_FULL_DEFAULT,
    _CHANNEL_KEYWORD,
    MemoryService,
)
from memorylayer_server.services.memory.query_intent import (
    ENTITY,
    EVENT,
    GENERAL,
    TEMPORAL,
    QueryIntent,
)

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _make_v(**overrides) -> Variables:
    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    # Default posture: every fuse arm ENABLED so we can observe which ones FIRE.
    # Query intent routing ON so `recall` classifies and threads the intent down.
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, True)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, True)
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, True)
    v.set(MEMORYLAYER_QUERY_INTENT_ENABLED, True)
    v.set(MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED, False)
    for k, val in overrides.items():
        v.set(k, val)
    return v


class _FakeStorage:
    """Minimal storage: a single primary vector pool; entity/fact arms empty."""

    def __init__(self):
        self.primary: list[tuple[Memory, float]] = []

    async def search_memories(
        self,
        workspace_id: str,
        query_embedding,
        limit: int = 10,
        offset: int = 0,
        min_relevance: float = 0.5,
        types=None,
        subtypes=None,
        tags=None,
        include_archived: bool = False,
        observer_id=None,
        subject_id=None,
        created_after=None,
        created_before=None,
        user_id=None,
    ):
        if subtypes and MemorySubtype.FACT.value in subtypes:
            return []
        out = [(m, s) for m, s in self.primary if s >= min_relevance]
        return out[offset : offset + limit]

    async def search_memories_by_entities(self, *args, **kwargs):
        return []

    async def full_text_search(self, *args, **kwargs):
        return []


def _make_service(v: Variables) -> MemoryService:
    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[0.1] * 8)
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )
    storage = _FakeStorage()
    svc = MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        v=v,
    )
    svc.reranker_service = None
    svc.cache = None
    return svc


def _mem(mid: str) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content="turn content",
        content_hash=f"h_{mid}",
        type=MemoryType.SEMANTIC,
        subtype=None,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
    )


def _recall_input(query: str, limit: int = 3) -> RecallInput:
    return RecallInput(
        query=query,
        mode=RecallMode.RAG,
        limit=limit,
        min_relevance=0.0,
        include_associations=False,
        include_global=False,
        include_global_user=False,
    )


def _spy_arms(svc: MemoryService) -> dict[str, bool]:
    """Wrap the fuse methods so a recall records which arms actually fired.

    The dense arm (A) is unconditional in `_recall_rag` (the primary
    `search_memories` call), so it is treated as always-fired for the firing-set;
    B/D/E are the wrapped fuse arms. Each wrapper preserves the original
    pass-through (returns the vector_results unchanged since the fake pools for
    keyword/entity/fact are empty).
    """
    fired: dict[str, bool] = {_CHANNEL_DENSE: True, _CHANNEL_KEYWORD: False, _CHANNEL_ENTITY: False, _CHANNEL_FACT: False}

    real_keyword = svc._fuse_keyword_results
    real_entity = svc._fuse_entity_results
    real_fact = svc._fuse_fact_results

    async def keyword_spy(*args, **kwargs):
        fired[_CHANNEL_KEYWORD] = True
        return await real_keyword(*args, **kwargs)

    async def entity_spy(*args, **kwargs):
        fired[_CHANNEL_ENTITY] = True
        return await real_entity(*args, **kwargs)

    async def fact_spy(*args, **kwargs):
        fired[_CHANNEL_FACT] = True
        return await real_fact(*args, **kwargs)

    svc._fuse_keyword_results = keyword_spy
    svc._fuse_entity_results = entity_spy
    svc._fuse_fact_results = fact_spy
    return fired


def _fired_set(fired: dict[str, bool]) -> frozenset[str]:
    return frozenset(ch for ch, on in fired.items() if on)


# Real queries chosen so the rule-based classifier yields a definite intent.
# (verified against classify_query_intent)
_QUERY_BY_INTENT = {
    ENTITY: "Caroline Smith",  # proper-noun span -> entity
    TEMPORAL: "What did we do yesterday",  # temporal token -> temporal
    EVENT: "what happened at the party",  # event phrase -> event
    GENERAL: "tell me about cooking",  # no signal -> general
}


# ===========================================================================
# Layer 1: _select_channels in isolation (pure, deterministic over labels)
# ===========================================================================


def test_select_channels_flag_off_is_full_default_for_every_intent():
    """Flag OFF -> _select_channels always returns the full default set (no-op)."""
    svc = _make_service(_make_v())  # flag OFF
    for label in (ENTITY, TEMPORAL, EVENT, GENERAL):
        assert svc._select_channels(QueryIntent(labels={label})) == _CHANNEL_FULL_DEFAULT
    # Even None / empty degrade to full default while OFF.
    assert svc._select_channels(None) == _CHANNEL_FULL_DEFAULT
    assert svc._select_channels(QueryIntent(labels=set())) == _CHANNEL_FULL_DEFAULT


def test_select_channels_flag_on_matches_matrix_per_intent():
    """Flag ON -> each single intent returns exactly its matrix arm set."""
    svc = _make_service(_make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True}))
    assert svc._select_channels(QueryIntent(labels={ENTITY})) == frozenset(
        {_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT}
    )
    assert svc._select_channels(QueryIntent(labels={EVENT})) == frozenset(
        {_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT}
    )
    assert svc._select_channels(QueryIntent(labels={TEMPORAL})) == frozenset(
        {_CHANNEL_DENSE, _CHANNEL_KEYWORD}
    )
    # GENERAL keeps the entity-anchor (D): it's the rule-based classifier's
    # catch-all, and the P4.1 flip-gate showed dropping D from `general` craters
    # multi_hop recall (the classifier dumps multi_hop queries into `general`).
    assert svc._select_channels(QueryIntent(labels={GENERAL})) == frozenset(
        {_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY}
    )


def test_select_channels_flag_on_dense_and_keyword_always_present():
    """A and B are the proven backbone: present in every selected set."""
    svc = _make_service(_make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True}))
    for label in (ENTITY, TEMPORAL, EVENT, GENERAL):
        sel = svc._select_channels(QueryIntent(labels={label}))
        assert _CHANNEL_DENSE in sel
        assert _CHANNEL_KEYWORD in sel


def test_select_channels_flag_on_multilabel_is_union():
    """A multi-label intent gets the UNION of each label's matrix set."""
    svc = _make_service(_make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True}))
    # temporal({A,B}) ∪ event({A,B,D,E}) = {A,B,D,E}
    sel = svc._select_channels(QueryIntent(labels={TEMPORAL, EVENT}))
    assert sel == frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT})


def test_select_channels_safe_degrade_on_uncertainty():
    """None / empty / unknown label all degrade to the FULL default set (flag ON)."""
    svc = _make_service(_make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True}))
    # Missing intent (e.g. query-intent routing disabled).
    assert svc._select_channels(None) == _CHANNEL_FULL_DEFAULT
    # Empty label set.
    assert svc._select_channels(QueryIntent(labels=set())) == _CHANNEL_FULL_DEFAULT
    # Unknown / future / low-confidence label.
    assert svc._select_channels(QueryIntent(labels={"some_unknown_label"})) == _CHANNEL_FULL_DEFAULT
    # A recognized label mixed with an unknown one still degrades (any unknown -> full).
    assert (
        svc._select_channels(QueryIntent(labels={GENERAL, "some_unknown_label"})) == _CHANNEL_FULL_DEFAULT
    )


# ===========================================================================
# Layer 2: end-to-end arm firing through recall(...) (spies on fuse methods)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_label", [ENTITY, TEMPORAL, EVENT, GENERAL])
async def test_flag_off_fires_all_enabled_arms_for_every_intent(intent_label):
    """No-op guarantee: flag OFF -> every enabled arm fires for every intent.

    This is the byte-identical-to-today baseline: with all channel flags ON and
    the intent-select flag OFF, the firing set is the full default set regardless
    of the classified intent.
    """
    svc = _make_service(_make_v())  # intent-select flag OFF, all channels ON
    svc.storage.primary = [(_mem("m0"), 0.9)]
    fired = _spy_arms(svc)

    await svc.recall("ws", _recall_input(_QUERY_BY_INTENT[intent_label]))

    assert _fired_set(fired) == _CHANNEL_FULL_DEFAULT, (
        f"intent {intent_label!r}: flag-off firing set must equal today's full default"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent_label,expected",
    [
        (ENTITY, frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT})),
        (EVENT, frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT})),
        (TEMPORAL, frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD})),
        (GENERAL, frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY})),
    ],
)
async def test_flag_on_fires_exactly_matrix_set_per_intent(intent_label, expected):
    """Flag ON -> each intent fires exactly its matrix arm set, end to end."""
    svc = _make_service(_make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True}))
    svc.storage.primary = [(_mem("m0"), 0.9)]
    fired = _spy_arms(svc)

    await svc.recall("ws", _recall_input(_QUERY_BY_INTENT[intent_label]))

    assert _fired_set(fired) == expected, f"intent {intent_label!r}: wrong arm-firing set"


@pytest.mark.asyncio
async def test_flag_on_missing_intent_safe_degrades_to_full_default():
    """Flag ON but intent routing OFF -> intent is None -> full default arm set.

    This proves the safe-degrade-on-uncertainty path end to end: with no
    classified intent the recall must still fire A+B+D+E (today's behavior),
    never fewer arms.
    """
    v = _make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True, MEMORYLAYER_QUERY_INTENT_ENABLED: False})
    svc = _make_service(v)
    svc.storage.primary = [(_mem("m0"), 0.9)]
    fired = _spy_arms(svc)

    # Even a query that WOULD classify as general/temporal: intent routing off -> None.
    await svc.recall("ws", _recall_input(_QUERY_BY_INTENT[TEMPORAL]))

    assert _fired_set(fired) == _CHANNEL_FULL_DEFAULT


@pytest.mark.asyncio
async def test_kill_switch_precedence_channel_flag_off_never_fires():
    """Channel's own flag OFF -> that channel never fires even if the matrix selects it.

    Entity intent's matrix selects D (entity-anchor), but with
    entity_anchor_enabled=False the global kill-switch wins and D must not fire.
    """
    v = _make_v(
        **{
            MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True,
            MEMORYLAYER_ENTITY_ANCHOR_ENABLED: False,  # global kill-switch for D
        }
    )
    svc = _make_service(v)
    svc.storage.primary = [(_mem("m0"), 0.9)]
    fired = _spy_arms(svc)

    # Entity intent -> matrix selects {A,B,D,E}, but D is killed -> {A,B,E}.
    await svc.recall("ws", _recall_input(_QUERY_BY_INTENT[ENTITY]))

    assert fired[_CHANNEL_ENTITY] is False, "entity arm fired despite its flag being OFF"
    assert _fired_set(fired) == frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_FACT})


@pytest.mark.asyncio
async def test_flag_off_with_a_channel_flag_off_matches_today():
    """Flag OFF baseline holds when a channel flag is off too (kill-switch unchanged).

    With intent-select OFF and the fact channel OFF, the firing set is {A,B,D} for
    every intent — exactly today's behavior (the fact arm is simply absent).
    """
    v = _make_v(**{MEMORYLAYER_FACT_CHANNEL_ENABLED: False})  # intent-select stays OFF
    svc = _make_service(v)
    svc.storage.primary = [(_mem("m0"), 0.9)]
    fired = _spy_arms(svc)

    await svc.recall("ws", _recall_input(_QUERY_BY_INTENT[ENTITY]))

    assert _fired_set(fired) == frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY})


@pytest.mark.asyncio
async def test_wildcard_query_path_unchanged_under_flag_on():
    """Wildcard '*' -> browse path; no fuse arms fire, flag on or off (unchanged)."""
    v = _make_v(**{MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: True})
    svc = _make_service(v)
    svc.storage.primary = [(_mem("m0"), 0.9)]

    # get_recent_memories / get_memory are used by the browse path; stub them.
    svc.storage.get_recent_memories = AsyncMock(return_value=[])
    svc.storage.get_memory = AsyncMock(return_value=None)

    fired = _spy_arms(svc)
    # Dense is "always" in the firing-set sentinel, but for wildcard the RAG body
    # is skipped entirely (browse mode), so NO fuse arm should fire.
    fired[_CHANNEL_DENSE] = False

    result = await svc.recall("ws", _recall_input("*"))

    assert fired[_CHANNEL_KEYWORD] is False
    assert fired[_CHANNEL_ENTITY] is False
    assert fired[_CHANNEL_FACT] is False
    assert result.memories == []
