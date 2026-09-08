"""Unit tests for the DefaultRepresentationService (P3 perspective slice 1).

Covers the deterministic (observer, subject) assembly contract against in-memory
FAKES for the entity registry + storage (no SQLite, no recall):

  * self-report (O == S): the subject's role="self" members become observations.
  * other-observation INTERSECTION: a turn O authored mentioning S is included;
    a turn mentioning S authored by a DIFFERENT observer is EXCLUDED (the
    leakage guard — the key test); a turn O authored NOT mentioning S is
    excluded.
  * unknown observer/subject: resolve raises LookupError -> empty Representation
    with provenance, no crash.
  * profile: a PROFILE-subtype memory -> profile.derived=False from it; else a
    synthesized-from-observations fallback.
  * limit/truncation + provenance.truncated.

The whole surface is leakage-safe by construction: it NEVER calls recall() and
NEVER injects members into recall — these fakes have no recall surface at all,
which is itself evidence of the no-recall property.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memorylayer_server.models.entity_registry import (
    Entity,
    EntityResolution,
    EntityType,
    Member,
)
from memorylayer_server.models.memory import Memory, MemoryStatus, MemorySubtype, MemoryType
from memorylayer_server.services.representation.default import DefaultRepresentationService

WS = "ws-rep-1"


# --------------------------------------------------------------------------- #
# Fakes (registry + storage). Deliberately minimal — only the methods the
# DefaultRepresentationService actually calls. No recall surface exists.
# --------------------------------------------------------------------------- #
class FakeRegistry:
    """In-memory entity registry fake.

    ``entities`` maps surface name (lowercased) -> Entity. ``members`` maps
    entity_id -> list[Member]. ``resolve`` honours allow_create=False by raising
    LookupError on a miss (never creates).
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.members: dict[str, list[Member]] = {}

    def add_entity(self, name: str, entity_id: str, entity_type=EntityType.PERSON) -> Entity:
        now = datetime.now(UTC)
        ent = Entity(
            id=entity_id,
            workspace_id=WS,
            entity_type=entity_type,
            canonical_name=name,
            normalized_name=name.lower(),
            created_at=now,
            updated_at=now,
        )
        self.entities[name.lower()] = ent
        return ent

    def add_member(self, entity_id: str, memory_id: str, role: str):
        self.members.setdefault(entity_id, []).append(
            Member(entity_id=entity_id, memory_id=memory_id, role=role)
        )

    async def resolve(self, workspace_id, name, entity_type, *, allow_create=True, **kw):
        ent = self.entities.get(name.lower())
        if ent is None:
            if not allow_create:
                raise LookupError(f"no entity {name!r}")
            raise AssertionError("representation service must never create entities")
        return EntityResolution(entity=ent, matched_via="exact", score=1.0)

    async def list_members(self, workspace_id, entity_id, *, role=None, limit=100):
        rows = self.members.get(entity_id, [])
        if role is not None:
            rows = [m for m in rows if m.role == role]
        return rows[:limit]


class FakeStorage:
    """In-memory storage fake exposing only ``get_memory``.

    Records ``track_access`` per call so a test can assert decay is not
    perturbed.
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.track_access_calls: list[bool] = []

    def add_memory(self, mem: Memory):
        self.memories[mem.id] = mem

    async def get_memory(self, workspace_id, memory_id, track_access=True):
        self.track_access_calls.append(track_access)
        return self.memories.get(memory_id)


def _mem(
    mem_id: str,
    content: str,
    *,
    subtype: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    event_offset_days: int | None = None,
) -> Memory:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    event_time = None
    if event_offset_days is not None:
        event_time = base + timedelta(days=event_offset_days)
    return Memory(
        id=mem_id,
        workspace_id=WS,
        tenant_id="_default",
        content=content,
        content_hash=f"hash-{mem_id}",
        type=MemoryType.SEMANTIC,
        subtype=subtype,
        status=status,
        event_time=event_time,
        created_at=base,
        updated_at=base,
    )


@pytest.fixture
def registry():
    return FakeRegistry()


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def service(registry, storage):
    return DefaultRepresentationService(registry=registry, storage=storage, v=None)


@pytest.mark.asyncio
class TestSelfReport:
    async def test_self_report_returns_self_members(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "m1", "self")
        registry.add_member(alice.id, "m2", "self")
        storage.add_memory(_mem("m1", "I shipped the release", event_offset_days=2))
        storage.add_memory(_mem("m2", "I reviewed the PR", event_offset_days=1))

        rep = await service.get_representation(WS, "Alice", "Alice")

        assert rep.is_self is True
        assert rep.provenance["scoping_mode"] == "self"
        assert {o.memory_id for o in rep.observations} == {"m1", "m2"}
        assert all(o.role == "self" for o in rep.observations)
        # event_time desc: m1 (day 2) before m2 (day 1).
        assert [o.memory_id for o in rep.observations] == ["m1", "m2"]
        # Decay must not be perturbed.
        assert storage.track_access_calls and all(ta is False for ta in storage.track_access_calls)


@pytest.mark.asyncio
class TestOtherObservationIntersection:
    async def test_intersection_includes_only_observer_authored_mentions(
        self, service, registry, storage
    ):
        """THE leakage guard: only memories the OBSERVER authored that MENTION
        the subject are in scope.

        M1: authored by O, mentions S            -> INCLUDED
        M2: mentions S, authored by OTHER observer -> EXCLUDED (leakage guard)
        M3: authored by O, does NOT mention S    -> EXCLUDED
        """
        observer = registry.add_entity("Observer", "ent-o")
        subject = registry.add_entity("Subject", "ent-s")
        other = registry.add_entity("Other", "ent-other")

        # M1: O authored (self) + S mentioned -> in the intersection.
        registry.add_member(observer.id, "M1", "self")
        registry.add_member(subject.id, "M1", "mention")
        # M2: S mentioned, but authored by a DIFFERENT observer.
        registry.add_member(subject.id, "M2", "mention")
        registry.add_member(other.id, "M2", "self")
        # M3: O authored, but does NOT mention S.
        registry.add_member(observer.id, "M3", "self")

        storage.add_memory(_mem("M1", "Subject closed the deal", event_offset_days=1))
        storage.add_memory(_mem("M2", "Subject was late", event_offset_days=1))
        storage.add_memory(_mem("M3", "Observer had lunch", event_offset_days=1))

        rep = await service.get_representation(WS, "Observer", "Subject")

        assert rep.is_self is False
        assert rep.provenance["scoping_mode"] == "intersection"
        ids = {o.memory_id for o in rep.observations}
        assert ids == {"M1"}, f"leakage guard failed: expected only M1, got {ids}"
        assert "M2" not in ids  # authored by a different observer
        assert "M3" not in ids  # does not mention the subject


@pytest.mark.asyncio
class TestUnresolved:
    async def test_unknown_subject_returns_empty(self, service, registry):
        registry.add_entity("Observer", "ent-o")
        rep = await service.get_representation(WS, "Observer", "GhostSubject")
        assert rep.observations == []
        assert rep.profile is None
        assert "subject" in rep.provenance["unresolved"]
        assert rep.subject.provenance.get("unresolved") is True
        # No crash; observer side still resolved best-effort.
        assert rep.observer.id == "ent-o"

    async def test_unknown_observer_returns_empty(self, service, registry):
        registry.add_entity("Subject", "ent-s")
        rep = await service.get_representation(WS, "GhostObserver", "Subject")
        assert rep.observations == []
        assert "observer" in rep.provenance["unresolved"]

    async def test_no_registry_degrades_to_empty(self, storage):
        svc = DefaultRepresentationService(registry=None, storage=storage, v=None)
        rep = await svc.get_representation(WS, "A", "B")
        assert rep.observations == []
        assert rep.provenance["error"] == "no_registry"


@pytest.mark.asyncio
class TestProfile:
    async def test_profile_from_profile_subtype_memory(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "p1", "self")
        registry.add_member(alice.id, "m2", "self")
        storage.add_memory(
            _mem("p1", "Alice is a staff engineer", subtype=MemorySubtype.PROFILE.value, event_offset_days=5)
        )
        storage.add_memory(_mem("m2", "Alice shipped X", event_offset_days=1))

        rep = await service.get_representation(WS, "Alice", "Alice")
        assert rep.profile is not None
        assert rep.profile.derived is False
        assert rep.profile.summary == "Alice is a staff engineer"
        assert rep.profile.source_memory_ids == ["p1"]

    async def test_profile_synthesized_fallback(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "m1", "self")
        registry.add_member(alice.id, "m2", "self")
        storage.add_memory(_mem("m1", "shipped X", event_offset_days=2))
        storage.add_memory(_mem("m2", "reviewed Y", event_offset_days=1))

        rep = await service.get_representation(WS, "Alice", "Alice")
        assert rep.profile is not None
        assert rep.profile.derived is False
        # Synthesized from top observation contents, in order.
        assert rep.profile.summary == "shipped X reviewed Y"
        assert rep.profile.source_memory_ids == ["m1", "m2"]

    async def test_include_profile_false_skips_profile(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "m1", "self")
        storage.add_memory(_mem("m1", "shipped X"))
        rep = await service.get_representation(WS, "Alice", "Alice", include_profile=False)
        assert rep.profile is None


@pytest.mark.asyncio
class TestFilteringAndTruncation:
    async def test_archived_and_deleted_dropped(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        for mid, status in [
            ("m1", MemoryStatus.ACTIVE),
            ("m2", MemoryStatus.ARCHIVED),
            ("m3", MemoryStatus.DELETED),
        ]:
            registry.add_member(alice.id, mid, "self")
            storage.add_memory(_mem(mid, f"content {mid}", status=status, event_offset_days=1))
        rep = await service.get_representation(WS, "Alice", "Alice")
        assert {o.memory_id for o in rep.observations} == {"m1"}

    async def test_limit_truncation(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        for i in range(5):
            mid = f"m{i}"
            registry.add_member(alice.id, mid, "self")
            storage.add_memory(_mem(mid, f"content {i}", event_offset_days=i))
        rep = await service.get_representation(WS, "Alice", "Alice", limit=3)
        assert len(rep.observations) == 3
        assert rep.provenance["truncated"] is True
        # Most-recent-first: highest event offsets win.
        assert [o.memory_id for o in rep.observations] == ["m4", "m3", "m2"]

    async def test_no_truncation_flag_when_under_limit(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "m1", "self")
        storage.add_memory(_mem("m1", "x"))
        rep = await service.get_representation(WS, "Alice", "Alice", limit=20)
        assert rep.provenance["truncated"] is False


@pytest.mark.asyncio
class TestContractStability:
    async def test_derived_beliefs_always_empty_in_slice1(self, service, registry, storage):
        alice = registry.add_entity("Alice", "ent-alice")
        registry.add_member(alice.id, "m1", "self")
        storage.add_memory(_mem("m1", "x"))
        rep = await service.get_representation(WS, "Alice", "Alice")
        assert rep.derived_beliefs == []
