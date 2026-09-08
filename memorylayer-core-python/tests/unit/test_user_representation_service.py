"""Unit tests for the user-scope self-representation (cross-workspace).

``DefaultRepresentationService.get_user_representation(user_id)`` assembles a
user's preferences/traits from their USER-scope (``_global_user``) memories,
filtered by a FORCED ``user_id`` filter. It is the cross-workspace analogue of
the (observer, subject) self-report with observer == subject == the user.

Covered here against an in-memory storage FAKE (no SQLite, no recall):

  * a user's prefs written from >=2 workspaces assemble into ONE cross-workspace
    self-representation (observations span the origin workspaces).
  * CARDINAL cross-user isolation: user B's prefs NEVER appear in user A's
    user-representation (the forced user_id filter — the leakage test).
  * graceful empty: no user_id / no _global_user rows -> empty, no raise.
  * profile + ordering + truncation parity with the (observer, subject) path.

The surface NEVER calls recall() and NEVER injects members into recall — this
fake has no recall surface at all, which is itself evidence of the property.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memorylayer_server.config import GLOBAL_USER_WORKSPACE_ID
from memorylayer_server.models.memory import Memory, MemoryStatus, MemorySubtype, MemoryType
from memorylayer_server.services.representation.default import DefaultRepresentationService


# --------------------------------------------------------------------------- #
# Storage fake: only ``search_memories_by_filter`` (the user-scope read seam).
# It honours the workspace + FORCED user_id + status filter exactly like a real
# backend so the cross-user isolation is exercised end-to-end.
# --------------------------------------------------------------------------- #
class FakeStorage:
    def __init__(self):
        self.memories: list[Memory] = []
        self.filter_calls: list[dict] = []

    def add_memory(self, mem: Memory):
        self.memories.append(mem)

    async def search_memories_by_filter(
        self,
        workspace_id,
        *,
        subtypes=None,
        tags=None,
        metadata_filter=None,
        status="active",
        context_id=None,
        user_id=None,
        limit=100,
        offset=0,
    ):
        self.filter_calls.append({"workspace_id": workspace_id, "user_id": user_id, "status": status})
        rows = [
            m
            for m in self.memories
            if m.workspace_id == workspace_id
            and (user_id is None or m.user_id == user_id)
            and (status is None or (m.status.value if hasattr(m.status, "value") else m.status) == status)
        ]
        return rows[offset : offset + limit]


def _user_mem(
    mem_id: str,
    content: str,
    *,
    user_id: str,
    origin_workspace_id: str,
    subtype: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    event_offset_days: int | None = None,
) -> Memory:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    event_time = base + timedelta(days=event_offset_days) if event_offset_days is not None else None
    return Memory(
        id=mem_id,
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        tenant_id="_default",
        user_id=user_id,
        content=content,
        content_hash=f"hash-{mem_id}",
        type=MemoryType.SEMANTIC,
        subtype=subtype,
        status=status,
        event_time=event_time,
        metadata={"origin_workspace_id": origin_workspace_id},
        created_at=base,
        updated_at=base,
    )


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def service(storage):
    return DefaultRepresentationService(registry=None, storage=storage, v=None)


@pytest.mark.asyncio
class TestUserSelfRepresentation:
    async def test_cross_workspace_self_representation(self, service, storage):
        """User A's prefs from TWO workspaces -> one cross-workspace self-rep."""
        storage.add_memory(
            _user_mem("a1", "Prefers dark mode", user_id="A", origin_workspace_id="ws-1", event_offset_days=2)
        )
        storage.add_memory(
            _user_mem("a2", "Likes concise answers", user_id="A", origin_workspace_id="ws-2", event_offset_days=1)
        )

        rep = await service.get_user_representation("A")

        assert rep.user_id == "A"
        assert {o.memory_id for o in rep.observations} == {"a1", "a2"}
        # observations span the origin workspaces (cross-workspace span).
        assert rep.provenance["origin_workspace_ids"] == ["ws-1", "ws-2"]
        assert rep.provenance["scoping_mode"] == "user"
        # event_time desc: a1 (day 2) before a2 (day 1).
        assert [o.memory_id for o in rep.observations] == ["a1", "a2"]
        # OSS deterministic path: no derived beliefs.
        assert rep.derived_beliefs == []
        # The read used the _global_user workspace + the FORCED user_id filter.
        assert storage.filter_calls[0]["workspace_id"] == GLOBAL_USER_WORKSPACE_ID
        assert storage.filter_calls[0]["user_id"] == "A"


@pytest.mark.asyncio
class TestCrossUserIsolation:
    async def test_user_b_prefs_never_in_user_a_representation(self, service, storage):
        """THE cardinal leakage test: the forced user_id filter keeps B out of A."""
        storage.add_memory(
            _user_mem("a1", "A likes vim", user_id="A", origin_workspace_id="ws-1", event_offset_days=1)
        )
        # User B's prefs live in the SAME _global_user workspace, different user_id.
        storage.add_memory(
            _user_mem("b1", "B_SECRET_PREFERENCE_TOKEN", user_id="B", origin_workspace_id="ws-1", event_offset_days=1)
        )

        rep = await service.get_user_representation("A")

        ids = {o.memory_id for o in rep.observations}
        assert ids == {"a1"}, f"cross-user leakage: expected only A's, got {ids}"
        # B's content must NEVER appear anywhere in A's representation.
        contents = " ".join(o.content for o in rep.observations)
        assert "B_SECRET_PREFERENCE_TOKEN" not in contents
        if rep.profile is not None:
            assert "B_SECRET_PREFERENCE_TOKEN" not in (rep.profile.summary or "")

    async def test_defense_in_depth_inprocess_filter(self, service, storage):
        """Even if a backend ignored the filter, the in-process re-assert drops
        foreign rows (belt-and-suspenders on the cardinal property)."""

        class LeakyStorage(FakeStorage):
            async def search_memories_by_filter(self, workspace_id, *, user_id=None, **kw):
                # Deliberately IGNORE the user_id filter (simulate a buggy backend).
                return list(self.memories)

        leaky = LeakyStorage()
        leaky.add_memory(_user_mem("a1", "A pref", user_id="A", origin_workspace_id="ws-1", event_offset_days=1))
        leaky.add_memory(_user_mem("b1", "B pref", user_id="B", origin_workspace_id="ws-1", event_offset_days=1))
        svc = DefaultRepresentationService(registry=None, storage=leaky, v=None)

        rep = await svc.get_user_representation("A")
        assert {o.memory_id for o in rep.observations} == {"a1"}


@pytest.mark.asyncio
class TestGracefulEmpty:
    async def test_no_user_id_returns_empty_no_raise(self, service, storage):
        rep = await service.get_user_representation("")
        assert rep.observations == []
        assert rep.profile is None
        assert rep.derived_beliefs == []
        assert rep.provenance["error"] == "no_user_id"

    async def test_no_global_user_rows_returns_empty(self, service, storage):
        rep = await service.get_user_representation("A")
        assert rep.observations == []
        assert rep.profile is None
        assert rep.provenance["scoping_mode"] == "user"
        assert rep.provenance["observation_count"] == 0

    async def test_storage_error_returns_empty_no_raise(self):
        class BrokenStorage:
            async def search_memories_by_filter(self, *a, **k):
                raise RuntimeError("storage down")

        svc = DefaultRepresentationService(registry=None, storage=BrokenStorage(), v=None)
        rep = await svc.get_user_representation("A")
        assert rep.observations == []
        assert rep.provenance["error"] == "storage_error"


@pytest.mark.asyncio
class TestProfileAndTruncation:
    async def test_profile_from_profile_subtype(self, service, storage):
        storage.add_memory(
            _user_mem(
                "p1", "A is a senior dev who prefers tabs", user_id="A", origin_workspace_id="ws-1",
                subtype=MemorySubtype.PROFILE.value, event_offset_days=5,
            )
        )
        storage.add_memory(_user_mem("m2", "shipped X", user_id="A", origin_workspace_id="ws-1", event_offset_days=1))
        rep = await service.get_user_representation("A")
        assert rep.profile is not None
        assert rep.profile.summary == "A is a senior dev who prefers tabs"
        assert rep.profile.derived is False
        assert rep.profile.source_memory_ids == ["p1"]

    async def test_archived_dropped(self, service, storage):
        storage.add_memory(_user_mem("m1", "active", user_id="A", origin_workspace_id="ws-1", event_offset_days=1))
        storage.add_memory(
            _user_mem("m2", "archived", user_id="A", origin_workspace_id="ws-1", status=MemoryStatus.ARCHIVED, event_offset_days=1)
        )
        rep = await service.get_user_representation("A")
        assert {o.memory_id for o in rep.observations} == {"m1"}

    async def test_limit_truncation(self, service, storage):
        for i in range(5):
            storage.add_memory(
                _user_mem(f"m{i}", f"pref {i}", user_id="A", origin_workspace_id="ws-1", event_offset_days=i)
            )
        rep = await service.get_user_representation("A", limit=3)
        assert len(rep.observations) == 3
        assert rep.provenance["truncated"] is True
        assert [o.memory_id for o in rep.observations] == ["m4", "m3", "m2"]

    async def test_include_profile_false(self, service, storage):
        storage.add_memory(_user_mem("m1", "x", user_id="A", origin_workspace_id="ws-1", event_offset_days=1))
        rep = await service.get_user_representation("A", include_profile=False)
        assert rep.profile is None
