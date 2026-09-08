"""Unit tests for the flag-gated cue-anchor retrieval channel.

The cue channel (Memora-inspired abstraction+cue indexing) generates short
"[entity] + [aspect]" semantic keys per memory, embeds them, and stores them
separately; at recall the cue anchors are searched by vector similarity and
dereferenced back to their primary memories, then RRF-fused as an additional
retrieval arm. It is a BOLT-ON arm (content embedding is untouched),
enterprise-backed, and ships DARK (default OFF).

OSS storage/extraction cue methods are no-ops (no LLM, no cue table), so the
fusion — the real deliverable — is tested directly by injecting a fake storage
whose ``search_cue_anchors`` returns primary memories.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType,
    RecallInput,
    RecallMode,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.extraction.base import ExtractionService
from memorylayer_server.services.extraction.default import DefaultExtractionService
from memorylayer_server.services.memory.base import (
    MEMORYLAYER_CUE_CHANNEL_ENABLED,
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
    MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_QUERY_INTENT_ENABLED,
)
from memorylayer_server.config import (
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.models.llm import LLMResponse, LLMRole
from memorylayer_server.services.memory.default import MemoryService
from memorylayer_server.services.storage.base import StorageBackend


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _make_v(**overrides) -> Variables:
    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    # Isolate the cue channel: keep other fusion channels off unless a test opts in.
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, False)
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
    v.set(MEMORYLAYER_QUERY_INTENT_ENABLED, False)
    for k, val in overrides.items():
        v.set(k, val)
    return v


class _FakeStorage:
    """In-test storage: a primary vector arm plus a cue-anchor arm.

    ``search_memories`` returns the primary arm; ``search_cue_anchors`` returns
    the cue arm (primary memories already dereferenced, ranked best-first). A
    call counter proves the arm is/ isn't consulted.
    """

    def __init__(self):
        self.primary: list[tuple[Memory, float]] = []
        self.cues: list[tuple[Memory, float]] = []
        self.cue_calls = 0
        self.stored_cues: list[dict] = []

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
        out = [(m, s) for m, s in self.primary if s >= min_relevance]
        return out[offset : offset + limit]

    async def search_cue_anchors(self, workspace_id: str, query_embedding, limit: int):
        self.cue_calls += 1
        return self.cues[:limit]

    async def store_cue_anchors(self, workspace_id: str, memory_id: str, cues: list[dict]):
        self.stored_cues = list(cues)
        return None


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
    # No reranker -> _apply_reranking is a plain top-limit slice (deterministic).
    svc.reranker_service = None
    svc.cache = None
    return svc


def _mem(mid: str, content: str = "turn content") -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content=content,
        content_hash=f"h_{mid}",
        type=MemoryType.SEMANTIC,
        subtype=None,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
    )


def _recall_input(limit: int = 3) -> RecallInput:
    return RecallInput(
        query="hiking trip",
        mode=RecallMode.RAG,
        limit=limit,
        min_relevance=0.0,
        include_associations=False,
        include_global=False,
        include_global_user=False,
    )


# ===========================================================================
# Recall arm behaviour
# ===========================================================================


@pytest.mark.asyncio
async def test_cue_arm_off_by_default_not_consulted():
    """Default config: the cue arm never calls search_cue_anchors, results unchanged."""
    v = _make_v()  # cue channel defaults OFF
    svc = _make_service(v)
    assert svc.cue_channel_enabled is False
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(3)]
    svc.storage.cues = [(_mem("cue_gold"), 0.99)]

    result = await svc.recall("ws", _recall_input(limit=5))
    ids = [m.id for m in result.memories]
    assert svc.storage.cue_calls == 0, "cue arm was consulted while OFF"
    assert "cue_gold" not in ids
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("cue" not in (m.match_signals or []) for m in result.memories)


@pytest.mark.asyncio
async def test_low_sim_memory_promoted_via_cue_arm():
    """A memory low in the vector arm is promoted into top-k via the fused cue arm."""
    v = _make_v()
    v.set(MEMORYLAYER_CUE_CHANNEL_ENABLED, True)
    svc = _make_service(v)

    # Primary arm: 5 turns, the gold memory is NOT here.
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(5)]
    # Cue arm: gold memory ranked #1 in the cue-restricted set (low raw sim).
    gold = _mem("cue_gold", content="Jane went hiking in the Alps")
    svc.storage.cues = [(gold, 0.20), (_mem("c2"), 0.10)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert svc.storage.cue_calls == 1
    assert "cue_gold" in ids, f"cue-anchored memory not promoted into top-3: {ids}"
    gold_mem = next(m for m in result.memories if m.id == "cue_gold")
    assert "cue" in (gold_mem.match_signals or [])


@pytest.mark.asyncio
async def test_noop_when_no_cues_present():
    """Flag on but empty cue arm: recall is unchanged (turns only, no 'cue' signal)."""
    v = _make_v()
    v.set(MEMORYLAYER_CUE_CHANNEL_ENABLED, True)
    svc = _make_service(v)
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(3)]
    svc.storage.cues = []

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("cue" not in (m.match_signals or []) for m in result.memories)


# ===========================================================================
# Cue ingest wiring (_inline_auto_enrich)
# ===========================================================================


class _FakeCueExtraction:
    """Extraction stub returning structured cue dicts for the ingest test."""

    def __init__(self, cues: list[dict]):
        self._cues = cues

    async def generate_cue_anchors(self, content: str) -> list[dict]:
        return self._cues


@pytest.mark.asyncio
async def test_inline_auto_enrich_stores_cue_dicts_entity_id_none_without_registry():
    """With the cue channel on but NO entity registry, cues are stored as dicts
    with cue/embedding/entity_id keys and entity_id=None (no resolution)."""
    v = _make_v()
    v.set(MEMORYLAYER_CUE_CHANNEL_ENABLED, True)
    svc = _make_service(v)
    # No registry wired -> entity resolution must yield None.
    assert svc.entity_registry_enabled is False
    assert svc.entity_registry_service is None

    svc.extraction_service = _FakeCueExtraction(
        [
            {"cue": "Jane hiking trip", "entity": "Jane", "aspect": "hiking trip"},
            {"cue": "Alps travel", "entity": "Alps", "aspect": "travel"},
        ]
    )
    svc.association_service = None
    svc.embedding.embed_batch = AsyncMock(return_value=[[0.1] * 8, [0.2] * 8])

    memory = _mem("m1", content="Jane went hiking in the Alps")
    await svc._inline_auto_enrich("ws", memory, [0.3] * 8, classify_type=False)

    assert len(svc.storage.stored_cues) == 2
    for row in svc.storage.stored_cues:
        assert set(row) == {"cue", "embedding", "entity_id"}
        assert row["entity_id"] is None
    assert [r["cue"] for r in svc.storage.stored_cues] == ["Jane hiking trip", "Alps travel"]


# ===========================================================================
# generate_cue_anchors (extraction)
# ===========================================================================


class _FakeLLM:
    """Minimal LLM stub returning a canned response for the cue prompt."""

    def __init__(self, response_content: str):
        self._response_content = response_content
        self.last_system_prompt = None

    async def complete(self, request, profile: str = "default", **_generation_metadata) -> LLMResponse:
        for message in request.messages:
            if message.role == LLMRole.SYSTEM:
                self.last_system_prompt = message.content
        return LLMResponse(
            content=self._response_content,
            model="fake",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason="stop",
        )


def _extraction(llm=None) -> DefaultExtractionService:
    return DefaultExtractionService(
        llm_service=llm,
        storage=None,
        deduplication_service=None,
        embedding_service=None,
    )


@pytest.mark.asyncio
async def test_generate_cue_anchors_no_llm_returns_empty():
    """Without an LLM, cue generation is a no-op ([])."""
    svc = _extraction(llm=None)
    assert await svc.generate_cue_anchors("Jane went hiking in the Alps") == []


@pytest.mark.asyncio
async def test_generate_cue_anchors_parses_structured_objects():
    """A fake LLM returning JSON objects yields structured dicts, capped at 3."""
    llm = _FakeLLM(
        '[{"entity": "Jane", "aspect": "hiking trip"}, '
        '{"entity": "Alps", "aspect": "travel"}, '
        '{"entity": "Jane", "aspect": "outdoor hobby"}, '
        '{"entity": "extra", "aspect": "cue"}]'
    )
    svc = _extraction(llm=llm)
    cues = await svc.generate_cue_anchors("Jane went hiking in the Alps")
    assert cues == [
        {"cue": "Jane hiking trip", "entity": "Jane", "aspect": "hiking trip"},
        {"cue": "Alps travel", "entity": "Alps", "aspect": "travel"},
        {"cue": "Jane outdoor hobby", "entity": "Jane", "aspect": "outdoor hobby"},
    ]
    assert len(cues) <= 3
    # Each dict carries the parsed components plus the reconstructed cue string.
    for c in cues:
        assert set(c) == {"cue", "entity", "aspect"}
        assert c["cue"] == f"{c['entity']} {c['aspect']}"


@pytest.mark.asyncio
async def test_generate_cue_anchors_markdown_wrapped():
    """A markdown-fenced JSON array of objects is stripped and parsed."""
    llm = _FakeLLM(
        '```json\n[{"entity": "Project Orion", "aspect": "timeline"}, '
        '{"entity": "Project Orion", "aspect": "budget"}]\n```'
    )
    svc = _extraction(llm=llm)
    cues = await svc.generate_cue_anchors("Project Orion is behind schedule and over budget")
    assert cues == [
        {"cue": "Project Orion timeline", "entity": "Project Orion", "aspect": "timeline"},
        {"cue": "Project Orion budget", "entity": "Project Orion", "aspect": "budget"},
    ]


@pytest.mark.asyncio
async def test_generate_cue_anchors_defensive_parse_skips_bad_items():
    """Non-object items and empty entity+aspect pairs are skipped defensively."""
    llm = _FakeLLM(
        '["a bare string", {"entity": "", "aspect": ""}, '
        '{"entity": "Nova", "aspect": "launch"}]'
    )
    svc = _extraction(llm=llm)
    cues = await svc.generate_cue_anchors("Nova launches next week")
    assert cues == [{"cue": "Nova launch", "entity": "Nova", "aspect": "launch"}]


@pytest.mark.asyncio
async def test_generate_cue_anchors_garbage_returns_empty():
    """Unparseable model output degrades gracefully to []."""
    llm = _FakeLLM("not json at all")
    svc = _extraction(llm=llm)
    assert await svc.generate_cue_anchors("anything") == []


# ===========================================================================
# OSS storage base no-ops
# ===========================================================================


class _BareBackend(StorageBackend):
    """Concrete backend that implements nothing beyond the abstract requirements
    only enough to instantiate — exercises inherited base cue no-ops."""

    async def connect(self): ...
    async def disconnect(self): ...
    async def health_check(self): return True
    async def create_memory(self, *a, **k): ...
    async def get_memory(self, *a, **k): ...
    async def update_memory(self, *a, **k): ...
    async def delete_memory(self, *a, **k): ...
    async def search_memories(self, *a, **k): return []
    async def full_text_search(self, *a, **k): return []
    async def get_memory_by_hash(self, *a, **k): ...
    async def get_recent_memories(self, *a, **k): return []
    async def create_association(self, *a, **k): ...
    async def get_associations(self, *a, **k): return []
    async def traverse_graph(self, *a, **k): ...
    async def create_workspace(self, *a, **k): ...
    async def get_workspace(self, *a, **k): ...
    async def create_context(self, *a, **k): ...
    async def get_context(self, *a, **k): ...
    async def list_contexts(self, *a, **k): return []
    async def list_workspaces(self, *a, **k): return []
    async def get_workspace_stats(self, *a, **k): return {}
    async def create_session(self, *a, **k): ...
    async def get_session(self, *a, **k): ...
    async def get_session_by_id(self, *a, **k): ...
    async def delete_session(self, *a, **k): ...
    async def set_working_memory(self, *a, **k): ...
    async def get_working_memory(self, *a, **k): ...
    async def get_all_working_memory(self, *a, **k): return []
    async def cleanup_expired_sessions(self, *a, **k): return 0


@pytest.mark.asyncio
async def test_storage_base_cue_methods_are_noops():
    """The OSS storage base cue methods return []/None so OSS is a clean no-op."""
    backend = _BareBackend()
    assert await backend.search_cue_anchors("ws", [0.1] * 4, 10) == []
    assert (
        await backend.store_cue_anchors(
            "ws",
            "mem1",
            [{"cue": "cue", "embedding": [0.1] * 4, "entity_id": None}],
        )
        is None
    )
    assert await backend.expand_via_cues("ws", ["mem1"], limit=10, threshold=0.85) == []


def test_extraction_base_generate_cue_anchors_is_noop_method():
    """The ExtractionService ABC declares generate_cue_anchors as a concrete no-op."""
    # Concrete (not abstract): a subclass need not override it.
    assert "generate_cue_anchors" not in ExtractionService.__abstractmethods__
