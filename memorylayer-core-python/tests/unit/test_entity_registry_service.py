"""Unit tests for the DefaultEntityRegistryService over SQLite storage.

Covers the slice-1 contract end-to-end against the relational backend:
exact / alias / create resolution, allow_create=False, member accretion across
two memories, find_by_alias, merge (member + alias reassignment + tombstoning),
and workspace isolation.
"""

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.entity_registry.default import DefaultEntityRegistryService
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend


@pytest_asyncio.fixture
async def storage(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "test_entity_registry.db"))
    await backend.connect()
    yield backend
    await backend.disconnect()


async def _make_workspace(storage, ws_id: str) -> str:
    await storage.create_workspace(
        Workspace(
            id=ws_id,
            tenant_id="_default",
            name=f"WS {ws_id}",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    return ws_id


async def _make_memory(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


@pytest_asyncio.fixture
async def workspace_id(storage) -> str:
    return await _make_workspace(storage, "ws-entity-1")


@pytest.fixture
def registry(storage):
    # Service only uses ``v`` for logging; None is fine for unit tests.
    return DefaultEntityRegistryService(storage=storage, v=None)


@pytest.mark.asyncio
class TestResolve:
    async def test_create_on_miss(self, registry, workspace_id):
        res = await registry.resolve(workspace_id, "Alice", EntityType.PERSON)
        assert res.matched_via == "created"
        assert res.entity.canonical_name == "Alice"
        assert res.entity.normalized_name == "alice"
        assert res.entity.entity_type == EntityType.PERSON
        assert res.entity.status == "active"
        assert res.entity.provenance.get("matched_via") == "created"

    async def test_exact_hit_on_second_resolve(self, registry, workspace_id):
        first = await registry.resolve(workspace_id, "Alice", EntityType.PERSON)
        # Different surface form, same normalized key -> exact match, same id.
        second = await registry.resolve(workspace_id, "alice", EntityType.PERSON)
        assert second.matched_via == "exact"
        assert second.score == 1.0
        assert second.entity.id == first.entity.id

    async def test_possessive_resolves_exact(self, registry, workspace_id):
        first = await registry.resolve(workspace_id, "Alice", EntityType.PERSON)
        again = await registry.resolve(workspace_id, "Alice's", EntityType.PERSON)
        assert again.matched_via == "exact"
        assert again.entity.id == first.entity.id

    async def test_alias_hit(self, registry, workspace_id):
        created = await registry.upsert(
            workspace_id, "Robert", EntityType.PERSON, aliases=["Bob"]
        )
        res = await registry.resolve(workspace_id, "Bob", EntityType.PERSON)
        assert res.matched_via == "alias"
        assert res.entity.id == created.id

    async def test_allow_create_false_raises(self, registry, workspace_id):
        with pytest.raises(LookupError):
            await registry.resolve(
                workspace_id, "Nonexistent", EntityType.PERSON, allow_create=False
            )

    async def test_type_separates_entities(self, registry, workspace_id):
        person = await registry.resolve(workspace_id, "Mercury", EntityType.PERSON)
        place = await registry.resolve(workspace_id, "Mercury", EntityType.PLACE)
        assert person.entity.id != place.entity.id
        assert person.matched_via == "created"
        assert place.matched_via == "created"


@pytest.mark.asyncio
class TestNameFirstPromotion:
    """``resolve(promote=True)`` name-first resolution + PERSON promotion.

    Guards the fix for the entity-type-split bug: a person who is both a speaker
    (PERSON) and a mention (CONCEPT) must collapse to ONE PERSON node so the
    default ``get_representation`` path resolves. Only PERSON⟷CONCEPT for the same
    normalized name is unified; the strict typed default (``promote=False``) and
    unrelated types (PLACE/etc.) are untouched.
    """

    async def test_mention_promotes_existing_concept_to_person(self, registry, workspace_id):
        # A name first seen as a CONCEPT mention...
        concept = await registry.resolve(workspace_id, "Bob", EntityType.CONCEPT)
        assert concept.entity.entity_type == EntityType.CONCEPT
        # ...then seen as a speaker (PERSON, promote=True) PROMOTES the SAME node.
        person = await registry.resolve(workspace_id, "Bob", EntityType.PERSON, promote=True)
        assert person.entity.id == concept.entity.id, "promotion must keep the entity id"
        assert person.entity.entity_type == EntityType.PERSON
        # No CONCEPT node remains for the name.
        with pytest.raises(LookupError):
            await registry.resolve(workspace_id, "Bob", EntityType.CONCEPT, allow_create=False)

    async def test_promotion_carries_members(self, registry, storage, workspace_id):
        # Attach a mention member to the CONCEPT node, then promote.
        concept = await registry.resolve(workspace_id, "Dana", EntityType.CONCEPT)
        mem = await _make_memory(storage, workspace_id, "[ts] X: Dana shipped it")
        await registry.add_member(workspace_id, concept.entity.id, mem, role="mention")
        promoted = await registry.resolve(workspace_id, "Dana", EntityType.PERSON, promote=True)
        assert promoted.entity.id == concept.entity.id
        members = await registry.list_members(workspace_id, promoted.entity.id, role="mention")
        assert [m.memory_id for m in members] == [mem], "mention member must survive promotion"

    async def test_mention_of_known_person_resolves_to_person(self, registry, workspace_id):
        # A name first seen as a speaker PERSON...
        person = await registry.resolve(workspace_id, "Eve", EntityType.PERSON, promote=True)
        # ...later mentioned (intended CONCEPT, promote=True) resolves to the PERSON.
        mention = await registry.resolve(workspace_id, "Eve", EntityType.CONCEPT, promote=True)
        assert mention.entity.id == person.entity.id
        assert mention.entity.entity_type == EntityType.PERSON

    async def test_existing_person_and_concept_merge_on_promote(self, registry, storage, workspace_id):
        # Force the (rare) state where BOTH a PERSON and a CONCEPT node exist for
        # the same name (e.g. created via strict typed resolve), each with a
        # member. Promoting must MERGE the CONCEPT into the PERSON (one survivor).
        person = await registry.resolve(workspace_id, "Frank", EntityType.PERSON)
        concept = await registry.resolve(workspace_id, "Frank", EntityType.CONCEPT)
        assert person.entity.id != concept.entity.id
        m_self = await _make_memory(storage, workspace_id, "[ts] Frank: I did it")
        m_mention = await _make_memory(storage, workspace_id, "[ts] X: Frank did it")
        await registry.add_member(workspace_id, person.entity.id, m_self, role="self")
        await registry.add_member(workspace_id, concept.entity.id, m_mention, role="mention")

        resolved = await registry.resolve(workspace_id, "Frank", EntityType.PERSON, promote=True)
        assert resolved.entity.id == person.entity.id, "PERSON node must survive the merge"
        # The CONCEPT is tombstoned; its mention member moved onto the PERSON.
        self_members = await registry.list_members(workspace_id, person.entity.id, role="self")
        mention_members = await registry.list_members(workspace_id, person.entity.id, role="mention")
        assert {m.memory_id for m in self_members} == {m_self}
        assert {m.memory_id for m in mention_members} == {m_mention}

    async def test_promote_does_not_touch_unrelated_type(self, registry, workspace_id):
        # A PLACE node with the same name must NOT be unified by promotion.
        place = await registry.resolve(workspace_id, "Jordan", EntityType.PLACE)
        person = await registry.resolve(workspace_id, "Jordan", EntityType.PERSON, promote=True)
        assert person.entity.id != place.entity.id
        assert person.entity.entity_type == EntityType.PERSON
        assert place.entity.entity_type == EntityType.PLACE

    async def test_default_resolve_still_type_separated(self, registry, workspace_id):
        # Without promote=True, strict typed semantics are preserved (no unify).
        concept = await registry.resolve(workspace_id, "Kim", EntityType.CONCEPT)
        person = await registry.resolve(workspace_id, "Kim", EntityType.PERSON)
        assert person.entity.id != concept.entity.id


@pytest.mark.asyncio
class TestMembers:
    async def test_accretion_two_memories_one_entity(self, registry, storage, workspace_id):
        mem_a = await _make_memory(storage, workspace_id, "Alice shipped the release")
        mem_b = await _make_memory(storage, workspace_id, "Alice reviewed the PR")

        res_a = await registry.resolve(
            workspace_id, "Alice", EntityType.PERSON, source_memory_id=mem_a
        )
        await registry.add_member(workspace_id, res_a.entity.id, mem_a)

        res_b = await registry.resolve(
            workspace_id, "Alice", EntityType.PERSON, source_memory_id=mem_b
        )
        # Same canonical entity for both memories.
        assert res_b.entity.id == res_a.entity.id
        assert res_b.matched_via == "exact"
        await registry.add_member(workspace_id, res_b.entity.id, mem_b)

        members = await registry.list_members(workspace_id, res_a.entity.id)
        assert len(members) == 2
        assert {m.memory_id for m in members} == {mem_a, mem_b}

    async def test_add_member_idempotent(self, registry, storage, workspace_id):
        mem = await _make_memory(storage, workspace_id, "Carol")
        ent = await registry.resolve(workspace_id, "Carol", EntityType.PERSON)
        await registry.add_member(workspace_id, ent.entity.id, mem)
        await registry.add_member(workspace_id, ent.entity.id, mem)  # dupe
        members = await registry.list_members(workspace_id, ent.entity.id)
        assert len(members) == 1


@pytest.mark.asyncio
class TestFindByAlias:
    async def test_find_by_alias(self, registry, workspace_id):
        created = await registry.upsert(
            workspace_id, "International Business Machines", EntityType.ORG, aliases=["IBM"]
        )
        hits = await registry.find_by_alias(workspace_id, "IBM")
        assert len(hits) == 1
        assert hits[0].id == created.id
        # Normalized lookup is case/diacritic insensitive (same normalized key).
        assert len(await registry.find_by_alias(workspace_id, "ibm")) == 1
        assert len(await registry.find_by_alias(workspace_id, "  Ibm  ")) == 1

    async def test_find_by_alias_type_filter(self, registry, workspace_id):
        await registry.upsert(workspace_id, "Apple Inc", EntityType.ORG, aliases=["Apple"])
        assert len(await registry.find_by_alias(workspace_id, "Apple", entity_type=EntityType.ORG)) == 1
        assert len(await registry.find_by_alias(workspace_id, "Apple", entity_type=EntityType.PERSON)) == 0


@pytest.mark.asyncio
class TestMerge:
    async def test_merge_reassigns_members_and_aliases(self, registry, storage, workspace_id):
        mem_a = await _make_memory(storage, workspace_id, "JS is great")
        mem_b = await _make_memory(storage, workspace_id, "JavaScript is great")

        source = await registry.upsert(
            workspace_id, "JS", EntityType.CONCEPT, aliases=["js-lang"]
        )
        target = await registry.upsert(workspace_id, "JavaScript", EntityType.CONCEPT)
        await registry.add_member(workspace_id, source.id, mem_a)
        await registry.add_member(workspace_id, target.id, mem_b)

        merged_target = await registry.merge(
            workspace_id, source.id, target.id, reason="duplicate"
        )

        # Target survives and absorbed the source's members.
        members = await registry.list_members(workspace_id, target.id)
        assert {m.memory_id for m in members} == {mem_a, mem_b}

        # Source canonical name + aliases carried forward as target aliases.
        assert "JS" in merged_target.aliases
        assert "js-lang" in merged_target.aliases

        # Source tombstoned.
        source_after = await registry.get(workspace_id, source.id)
        assert source_after.status == "merged"
        assert source_after.merged_into == target.id
        assert source_after.provenance.get("merged_reason") == "duplicate"

        # The source's old name now resolves (via alias) to the target.
        res = await registry.resolve(workspace_id, "JS", EntityType.CONCEPT)
        assert res.entity.id == target.id
        assert res.matched_via == "alias"

    async def test_merge_missing_raises(self, registry, workspace_id):
        target = await registry.upsert(workspace_id, "Target", EntityType.CONCEPT)
        with pytest.raises(LookupError):
            await registry.merge(workspace_id, "ent_missing", target.id, reason="x")

    async def test_merge_already_merged_source_is_noop(self, registry, storage, workspace_id):
        """FIX 1: merge() with an already-merged source must be a safe no-op.

        This is the TOCTOU double-merge scenario: first merge is legitimate;
        second call (same source_id) must not re-run destructive steps or corrupt
        the target's merged_into/aliases state.
        """
        source = await registry.upsert(workspace_id, "JS", EntityType.CONCEPT)
        target = await registry.upsert(workspace_id, "JavaScript", EntityType.CONCEPT)
        mem = await _make_memory(storage, workspace_id, "JS is great")
        await registry.add_member(workspace_id, source.id, mem)

        # First merge — legitimate.
        await registry.merge(workspace_id, source.id, target.id, reason="first")
        source_after_first = await registry.get(workspace_id, source.id)
        assert source_after_first.status == "merged"

        # Second merge with the same (already-merged) source — must be a no-op.
        # No exception, no corruption of the target.
        result = await registry.merge(workspace_id, source.id, target.id, reason="second")
        assert result.status == "active"
        assert result.id == target.id
        # Source is still tombstoned with the first merge's reason, not overwritten.
        source_final = await registry.get(workspace_id, source.id)
        assert source_final.status == "merged"
        assert source_final.merged_into == target.id
        assert source_final.provenance.get("merged_reason") == "first"

    async def test_merge_self_is_noop(self, registry, workspace_id):
        """FIX 1: merging an entity into itself must be a safe no-op."""
        ent = await registry.upsert(workspace_id, "SelfEntity", EntityType.CONCEPT)
        result = await registry.merge(workspace_id, ent.id, ent.id, reason="self")
        # Entity is still active and unchanged.
        assert result.status == "active"
        assert result.id == ent.id


@pytest.mark.asyncio
class TestWorkspaceIsolation:
    async def test_same_name_two_workspaces(self, registry, storage):
        ws1 = await _make_workspace(storage, "ws-iso-1")
        ws2 = await _make_workspace(storage, "ws-iso-2")

        e1 = await registry.resolve(ws1, "Acme", EntityType.ORG)
        e2 = await registry.resolve(ws2, "Acme", EntityType.ORG)
        assert e1.entity.id != e2.entity.id
        assert e1.matched_via == "created"
        assert e2.matched_via == "created"

        # ws1 lookup never sees ws2's entity.
        assert await registry.get(ws1, e2.entity.id) is None


@pytest.mark.asyncio
class TestDuplicateCreate:
    """MAJOR-1 fix: duplicate store_entity and duplicate resolve calls must not
    raise and must return the same canonical entity.

    These tests are the failing-before / passing-after evidence for the race
    fix in SQLiteStorageBackend.store_entity (IntegrityError → re-fetch).
    """

    async def test_store_entity_duplicate_returns_existing(self, storage, workspace_id):
        """Calling store_entity twice for the same active (ws, type, norm_name)
        must return the pre-existing row rather than raising IntegrityError."""
        entity_dict = {
            "workspace_id": workspace_id,
            "entity_type": "concept",
            "canonical_name": "RaceWidget",
            "normalized_name": "racewidget",
            "confidence": 1.0,
            "provenance": {"matched_via": "created"},
            "status": "active",
        }
        first = await storage.store_entity(entity_dict)
        # Second call with the same normalized_name must not raise.
        second = await storage.store_entity(dict(entity_dict))
        # Both calls return the same active entity (no duplicate active rows).
        assert first["id"] == second["id"]
        assert second["status"] == "active"
        # Only one active row with this normalized_name.
        hit = await storage.find_entity_by_normalized_name(
            workspace_id, "concept", "racewidget"
        )
        assert hit is not None
        assert hit["id"] == first["id"]

    async def test_resolve_twice_allow_create_returns_same_entity(self, registry, workspace_id):
        """Two consecutive resolve(allow_create=True) calls for the same name
        must return the same entity id (no duplicate active rows, no error)."""
        res1 = await registry.resolve(workspace_id, "DupliceName", EntityType.CONCEPT)
        res2 = await registry.resolve(workspace_id, "DupliceName", EntityType.CONCEPT)
        assert res1.entity.id == res2.entity.id
        assert res2.matched_via == "exact"

    async def test_store_entity_duplicate_after_member_add(self, storage, workspace_id):
        """After a member is added, a duplicate store_entity must still return
        the same entity and the member must be preserved."""
        mem_id = await _make_memory(storage, workspace_id, "race-content")
        entity_dict = {
            "workspace_id": workspace_id,
            "entity_type": "person",
            "canonical_name": "RacePerson",
            "normalized_name": "raceperson",
            "status": "active",
        }
        first = await storage.store_entity(entity_dict)
        await storage.add_entity_member(workspace_id, first["id"], mem_id)

        # Duplicate store — must not raise, must not touch existing member.
        second = await storage.store_entity(dict(entity_dict))
        assert second["id"] == first["id"]

        members = await storage.list_entity_members(workspace_id, first["id"])
        assert len(members) == 1
        assert members[0]["memory_id"] == mem_id
