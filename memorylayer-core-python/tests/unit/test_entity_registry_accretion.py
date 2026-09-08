"""Integration tests for flag-gated entity registry accretion at ingest time.

These exercise the wiring in ``_inline_auto_enrich`` -> ``_accrete_entities``:
  * flag ON: ingesting two memories that name the same entity yields ONE
    canonical entity with TWO members.
  * flag OFF (default): the registry is never touched (zero entities) and
    ingest behavior is unchanged.
  * role semantics: the speaker entity gets ``role="self"`` and mentioned
    entities get ``role="mention"``. The registry-recall channel requests
    ``role="mention"`` only, so self-authored turns are automatically excluded.

The accretion path uses the speaker/entities folded onto ``metadata`` by the
existing entity-anchor extraction (on by default in tests), so memories are
phrased with a ``[ts] Speaker:`` prefix and proper-noun bodies.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from memorylayer_server.models.entity_registry import Entity, EntityResolution, EntityType, Member
from memorylayer_server.models.memory import Memory, MemoryStatus, MemoryType, RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.entity_registry.default import DefaultEntityRegistryService
from memorylayer_server.services.memory.default import MemoryService


# ---------------------------------------------------------------------------
# Fake registry that records add_member calls (role-semantic unit tests)
# ---------------------------------------------------------------------------


class _RoleRecordingRegistry:
    """Minimal entity-registry stand-in that records add_member role arguments.

    ``resolve`` always succeeds (creates a stub entity so accretion never
    raises). ``add_member`` records every (entity_canonical_name, role) pair
    so tests can assert role assignment without needing real storage.
    """

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self.add_member_calls: list[dict] = []  # {name, entity_id, memory_id, role}

    def _get_or_create(self, name: str, entity_type: EntityType) -> Entity:
        key = name.lower()
        if key not in self._entities:
            now = datetime.now(UTC)
            self._entities[key] = Entity(
                id=f"ent_{key}",
                workspace_id="ws",
                entity_type=entity_type,
                canonical_name=name,
                normalized_name=key,
                created_at=now,
                updated_at=now,
            )
        return self._entities[key]

    async def resolve(self, workspace_id, name, entity_type, *, allow_create=True, **kwargs):
        ent = self._get_or_create(name, entity_type)
        return EntityResolution(entity=ent, matched_via="exact", score=1.0)

    async def add_member(self, workspace_id, entity_id, memory_id, *, role="mention"):
        # Find canonical name for readability in assertions.
        name = next(
            (e.canonical_name for e in self._entities.values() if e.id == entity_id),
            entity_id,
        )
        self.add_member_calls.append(
            {"name": name, "entity_id": entity_id, "memory_id": memory_id, "role": role}
        )

    async def list_members(self, workspace_id, entity_id, *, role=None, limit=100):
        return []


def _make_memory_service_with_fake_registry(registry) -> MemoryService:
    """Build a MemoryService wired to a fake storage + fake registry."""
    from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
    from memorylayer_server.config import (
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    )
    from memorylayer_server.services.memory.base import (
        MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
        MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    )
    from scitrera_app_framework import Variables

    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, True)

    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[1.0, 0.0])
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )

    # Minimal storage: remember() needs create_memory to persist + return an id.
    storage = AsyncMock()
    now = datetime.now(UTC)

    async def _create_memory(workspace_id, memory):
        memory.id = memory.id or f"mem_{uuid.uuid4().hex[:8]}"
        return memory

    storage.create_memory = AsyncMock(side_effect=_create_memory)
    storage.get_workspace = AsyncMock(return_value=AsyncMock(id="ws"))
    storage.update_memory = AsyncMock(side_effect=lambda ws, m: m)
    storage.search_memories = AsyncMock(return_value=[])
    storage.get_memory = AsyncMock(return_value=None)

    svc = MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        entity_registry_service=registry,
        v=v,
    )
    svc.entity_registry_enabled = True
    svc.reranker_service = None
    svc.cache = None
    return svc


async def _new_workspace(storage_backend) -> str:
    ws_id = f"ws-accretion-{uuid.uuid4().hex[:8]}"
    await storage_backend.create_workspace(
        Workspace(id=ws_id, tenant_id="_default", name="Accretion WS")
    )
    return ws_id


@pytest.mark.asyncio
async def test_accretion_flag_on_one_entity_two_members(memory_service, storage_backend, v):
    ws_id = await _new_workspace(storage_backend)

    # Enable accretion on this service instance only (ships DARK globally).
    registry = DefaultEntityRegistryService(storage=storage_backend, v=v)
    original_enabled = memory_service.entity_registry_enabled
    original_service = memory_service.entity_registry_service
    memory_service.entity_registry_enabled = True
    memory_service.entity_registry_service = registry
    try:
        await memory_service.remember(
            ws_id, RememberInput(content="[2026-06-01 10:00] Alice: I shipped the Orion release"), inline=True
        )
        await memory_service.remember(
            ws_id, RememberInput(content="[2026-06-01 11:00] Alice: I reviewed the Orion PR"), inline=True
        )
    finally:
        memory_service.entity_registry_enabled = original_enabled
        memory_service.entity_registry_service = original_service

    # "Alice" should be a single canonical PERSON entity with two members.
    res = await registry.resolve(ws_id, "Alice", EntityType.PERSON, allow_create=False)
    assert res.matched_via == "exact"
    members = await registry.list_members(ws_id, res.entity.id)
    assert len(members) == 2

    # "Orion" appears in both turns too -> one CONCEPT entity, two members.
    orion = await registry.resolve(ws_id, "Orion", EntityType.CONCEPT, allow_create=False)
    orion_members = await registry.list_members(ws_id, orion.entity.id)
    assert len(orion_members) == 2


@pytest.mark.asyncio
async def test_accretion_flag_off_registry_untouched(memory_service, storage_backend, v):
    ws_id = await _new_workspace(storage_backend)

    # Default config: flag OFF. Confirm the service reflects that.
    assert memory_service.entity_registry_enabled is False

    await memory_service.remember(
        ws_id, RememberInput(content="[2026-06-01 10:00] Bob: I shipped the Orion release"), inline=True
    )

    # Registry was never touched: no entity exists for the named spans.
    registry = DefaultEntityRegistryService(storage=storage_backend, v=v)
    for name, etype in (("Bob", EntityType.PERSON), ("Orion", EntityType.CONCEPT)):
        hit = await storage_backend.find_entity_by_normalized_name(
            ws_id, etype.value, name.casefold()
        )
        assert hit is None
        with pytest.raises(LookupError):
            await registry.resolve(ws_id, name, etype, allow_create=False)


# ===========================================================================
# Role-semantic unit tests (use _RoleRecordingRegistry + fake storage)
# ===========================================================================


@pytest.mark.asyncio
async def test_speaker_entity_gets_self_role_mention_entity_gets_mention_role():
    """Speaker entity accretes with role='self'; mentioned entity with role='mention'.

    Memory authored by "Mel" that also mentions "Caroline": after accretion the
    add_member call for Mel must use role='self' and the call for Caroline must
    use role='mention'.
    """
    registry = _RoleRecordingRegistry()
    svc = _make_memory_service_with_fake_registry(registry)

    # _accrete_entities reads speaker/entities from memory.metadata directly.
    # We call it with a pre-built Memory that already has the metadata folded on.
    now = datetime.now(UTC)
    memory = Memory(
        id="mem-mel-001",
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content="[2026-06-01 10:00] Mel: Caroline and I went to the market",
        content_hash="h1",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        metadata={
            "speaker": "Mel",
            "entities": ["Mel", "Caroline"],
            "entity_types": {},
        },
    )

    await svc._accrete_entities("ws", memory)

    roles_by_name = {c["name"]: c["role"] for c in registry.add_member_calls}
    assert "Mel" in roles_by_name, f"Mel not accreted: {registry.add_member_calls}"
    assert "Caroline" in roles_by_name, f"Caroline not accreted: {registry.add_member_calls}"
    assert roles_by_name["Mel"] == "self", (
        f"Speaker entity must get role='self', got {roles_by_name['Mel']!r}"
    )
    assert roles_by_name["Caroline"] == "mention", (
        f"Mentioned entity must get role='mention', got {roles_by_name['Caroline']!r}"
    )


@pytest.mark.asyncio
async def test_no_speaker_all_entities_get_mention_role():
    """When there is no speaker, every extracted entity gets role='mention'."""
    registry = _RoleRecordingRegistry()
    svc = _make_memory_service_with_fake_registry(registry)

    now = datetime.now(UTC)
    memory = Memory(
        id="mem-nospeaker-001",
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content="Alice and Bob discussed the Nimbus project",
        content_hash="h2",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        metadata={
            "speaker": None,
            "entities": ["Alice", "Bob", "Nimbus"],
            "entity_types": {},
        },
    )

    await svc._accrete_entities("ws", memory)

    assert registry.add_member_calls, "No add_member calls made"
    for call in registry.add_member_calls:
        assert call["role"] == "mention", (
            f"Entity {call['name']!r} got role={call['role']!r}; expected 'mention' "
            f"(no speaker present)"
        )


@pytest.mark.asyncio
async def test_observer_id_field_anchors_self_without_content_prefix():
    """observer_id is the AUTHORITATIVE self-anchor, independent of the regex.

    A prefix-less memory (real email/doc shape) whose ``observer_id`` is set to
    "Carol" but whose content carries NO ``[ts] Carol:`` dialogue prefix and does
    NOT mention "Carol" at all must still accrete the Carol entity as a
    ``role="self"`` member. This proves perspective is driven by the observer_id
    FIELD, not by the content speaker-prefix regex.
    """
    registry = _RoleRecordingRegistry()
    svc = _make_memory_service_with_fake_registry(registry)

    now = datetime.now(UTC)
    memory = Memory(
        id="mem-carol-email-001",
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        # Prefix-less body that never names Carol — the email/doc case.
        content="Phoenix will go live on the fourteenth of March per the roadmap.",
        content_hash="h-obs",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        observer_id="Carol",
        # No speaker, and Carol is NOT in the extracted entities — only the
        # proper nouns in the body would be (e.g. Phoenix). The self-anchor must
        # come from observer_id alone.
        metadata={
            "speaker": None,
            "entities": ["Phoenix"],
            "entity_types": {},
        },
    )

    await svc._accrete_entities("ws", memory)

    carol_calls = [c for c in registry.add_member_calls if c["name"] == "Carol"]
    assert carol_calls, (
        f"Carol (observer_id) not accreted from the FIELD: {registry.add_member_calls}"
    )
    assert all(c["role"] == "self" for c in carol_calls), (
        f"observer_id anchor must be role='self', got {carol_calls}"
    )
    # The Carol entity must have THIS memory attached.
    assert any(c["memory_id"] == "mem-carol-email-001" for c in carol_calls)


@pytest.mark.asyncio
async def test_subject_id_field_anchors_mention_member():
    """subject_id accretes the subject entity as a ``role="mention"`` member.

    A memory whose ``subject_id`` is set to "Bob" must attach this memory to the
    Bob entity as a mention (the memory is *about* Bob; Bob is observed, not the
    author), even when "Bob" is not in the content/extracted entities.
    """
    registry = _RoleRecordingRegistry()
    svc = _make_memory_service_with_fake_registry(registry)

    now = datetime.now(UTC)
    memory = Memory(
        id="mem-about-bob-001",
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content="The backend caching layer shipped this week.",
        content_hash="h-subj",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        observer_id="Alice",
        subject_id="Bob",
        metadata={
            "speaker": None,
            "entities": [],
            "entity_types": {},
        },
    )

    await svc._accrete_entities("ws", memory)

    bob_calls = [c for c in registry.add_member_calls if c["name"] == "Bob"]
    assert bob_calls, (
        f"Bob (subject_id) not accreted from the FIELD: {registry.add_member_calls}"
    )
    assert all(c["role"] == "mention" for c in bob_calls), (
        f"subject_id anchor must be role='mention', got {bob_calls}"
    )
    assert any(c["memory_id"] == "mem-about-bob-001" for c in bob_calls)

    # The observer (Alice) must also accrete as self from its field.
    alice_calls = [c for c in registry.add_member_calls if c["name"] == "Alice"]
    assert alice_calls and all(c["role"] == "self" for c in alice_calls), (
        f"observer_id Alice must accrete as role='self', got {alice_calls}"
    )


@pytest.mark.asyncio
async def test_observer_id_anchor_dedups_against_regex_self_member():
    """observer_id anchor does not double-add a (entity, memory, role) the regex
    loop already produced.

    A dialogue memory where the speaker == observer_id ("Mel") already accretes
    Mel as role='self' via the regex loop. The explicit observer_id anchor must
    NOT add a second Mel/self member for the same memory (dedup), so exactly one
    Mel 'self' add_member call is recorded.
    """
    registry = _RoleRecordingRegistry()
    svc = _make_memory_service_with_fake_registry(registry)

    now = datetime.now(UTC)
    memory = Memory(
        id="mem-mel-dedup-001",
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content="[2026-06-01 10:00] Mel: I finished the report",
        content_hash="h-dedup",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        observer_id="Mel",
        metadata={
            "speaker": "Mel",
            "entities": ["Mel"],
            "entity_types": {},
        },
    )

    await svc._accrete_entities("ws", memory)

    mel_self = [
        c for c in registry.add_member_calls if c["name"] == "Mel" and c["role"] == "self"
    ]
    assert len(mel_self) == 1, (
        f"observer_id anchor double-added Mel/self: {registry.add_member_calls}"
    )


@pytest.mark.asyncio
async def test_registry_recall_channel_requests_mention_role_only():
    """Confirm _fuse_registry_results passes role='mention' to list_members.

    This is a contract assertion: the recall expansion channel must explicitly
    request only 'mention' members so self-authored turns (role='self') are
    excluded from expansion even after the role distinction is introduced by
    accretion. Mirrors the existing test_list_members_called_with_mention_role
    in test_entity_registry_recall.py but is co-located here to document the
    accretion<->recall role contract in one place.
    """
    from memorylayer_server.config import MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED
    from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
    from memorylayer_server.config import (
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    )
    from memorylayer_server.services.memory.base import (
        MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
        MEMORYLAYER_FACT_CHANNEL_ENABLED,
        MEMORYLAYER_HYBRID_SEARCH_ENABLED,
        MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
        MEMORYLAYER_QUERY_INTENT_ENABLED,
    )
    from memorylayer_server.models.memory import RecallInput, RecallMode
    from scitrera_app_framework import Variables

    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, False)
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
    v.set(MEMORYLAYER_QUERY_INTENT_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)

    # Build a registry that records list_members calls.
    registry = _RoleRecordingRegistry()
    # Patch list_members to record role kwarg.
    list_members_calls: list[dict] = []

    async def _recording_list_members(workspace_id, entity_id, *, role=None, limit=100):
        list_members_calls.append({"entity_id": entity_id, "role": role})
        return []

    registry.list_members = _recording_list_members  # type: ignore[method-assign]

    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[1.0, 0.0])
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )
    storage = AsyncMock()
    storage.search_memories = AsyncMock(return_value=[])
    storage.get_memory = AsyncMock(return_value=None)

    svc = MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        entity_registry_service=registry,
        v=v,
    )
    svc.reranker_service = None
    svc.cache = None

    recall_input = RecallInput(
        query="[2026-06-01 10:00] Mel: What did Caroline say",
        mode=RecallMode.RAG,
        limit=3,
        min_relevance=0.0,
        include_associations=False,
        include_global=False,
        include_global_user=False,
    )
    await svc.recall("ws", recall_input)

    # list_members may not have been called if no entities resolved (empty registry),
    # but IF it was called it must request role='mention'.
    for call in list_members_calls:
        assert call["role"] == "mention", (
            f"Registry recall channel called list_members with role={call['role']!r}; "
            f"expected 'mention' to exclude self-authored turns"
        )
