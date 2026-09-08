"""Tests for recall-side supersession.

Contradiction detection has always run on every store and recorded WHICH of the two
memories is current. Recall never looked, so a fact positively identified as stale was
still returned at full score. These pin the read half:

  * the supersession direction actually round-trips through storage (it did not before —
    `newer_memory_id` was computed and then dropped on write)
  * `demote` lowers the stale memory's score; `exclude` removes it; `off` changes nothing
  * a contradiction with NO recorded direction supersedes nothing, rather than guessing
  * a RESOLVED contradiction stops penalising
"""

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.contradiction.base import ContradictionRecord


@pytest_asyncio.fixture
async def ws(storage_backend, unique_workspace_id):
    from memorylayer_server.models.workspace import Workspace

    if not await storage_backend.get_workspace(unique_workspace_id):
        await storage_backend.create_workspace(
            Workspace(
                id=unique_workspace_id,
                tenant_id="default_tenant",
                name="supersession test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    return unique_workspace_id


async def _mem(storage_backend, workspace_id, content):
    return await storage_backend.create_memory(workspace_id, RememberInput(content=content, importance=0.5))


@pytest.mark.asyncio
class TestSupersessionPersistence:
    async def test_newer_memory_id_round_trips(self, storage_backend, ws):
        """Regression: the direction was computed on every detection and dropped on write.

        Without it a contradiction says two memories conflict but not which is current,
        so recall-side supersession has nothing to act on.
        """
        old = await _mem(storage_backend, ws, "I live in Boston.")
        new = await _mem(storage_backend, ws, "I moved to Seattle.")

        stored = await storage_backend.create_contradiction(
            ContradictionRecord(
                workspace_id=ws,
                memory_a_id=new.id,
                memory_b_id=old.id,
                contradiction_type="temporal_supersession",
                confidence=0.9,
                detection_method="llm_fused",
                newer_memory_id=new.id,
            )
        )

        fetched = await storage_backend.get_contradiction(ws, stored.id)
        assert fetched is not None
        assert fetched.newer_memory_id == new.id

    async def test_superseded_lookup_returns_only_the_stale_side(self, storage_backend, ws):
        old = await _mem(storage_backend, ws, "My phone is 555-0100.")
        new = await _mem(storage_backend, ws, "My new phone is 555-0199.")
        await storage_backend.create_contradiction(
            ContradictionRecord(
                workspace_id=ws,
                memory_a_id=new.id,
                memory_b_id=old.id,
                detection_method="llm_fused",
                newer_memory_id=new.id,
            )
        )

        superseded = await storage_backend.get_superseded_memory_ids(ws, [old.id, new.id])

        assert superseded == {old.id}, "only the older memory is stale"

    async def test_contradiction_without_direction_supersedes_nothing(self, storage_backend, ws):
        """No recorded direction means we know they conflict, not which one is current."""
        a = await _mem(storage_backend, ws, "The build is broken.")
        b = await _mem(storage_backend, ws, "The build is fine.")
        await storage_backend.create_contradiction(
            ContradictionRecord(
                workspace_id=ws,
                memory_a_id=a.id,
                memory_b_id=b.id,
                detection_method="negation_pattern",
                newer_memory_id=None,
            )
        )

        assert await storage_backend.get_superseded_memory_ids(ws, [a.id, b.id]) == set()

    async def test_resolved_contradiction_stops_superseding(self, storage_backend, ws):
        """Resolution is the operator saying it is dealt with; penalising on would hide that."""
        old = await _mem(storage_backend, ws, "I have two cats.")
        new = await _mem(storage_backend, ws, "I now have three cats.")
        rec = await storage_backend.create_contradiction(
            ContradictionRecord(
                workspace_id=ws,
                memory_a_id=new.id,
                memory_b_id=old.id,
                detection_method="llm_fused",
                newer_memory_id=new.id,
            )
        )
        assert await storage_backend.get_superseded_memory_ids(ws, [old.id, new.id]) == {old.id}

        await storage_backend.resolve_contradiction(ws, rec.id, "keep_a")

        assert await storage_backend.get_superseded_memory_ids(ws, [old.id, new.id]) == set()

    async def test_empty_input_is_a_no_op(self, storage_backend, ws):
        assert await storage_backend.get_superseded_memory_ids(ws, []) == set()


@pytest.mark.asyncio
class TestApplySupersession:
    """The recall-side policy, exercised directly on the memory service."""

    @staticmethod
    def _scored(*pairs):
        """Attach recall scores to real stored Memory objects.

        Constructed rather than fetched would need every required field; using the real
        objects also keeps this honest about what recall actually operates on.
        """
        out = []
        for memory, score in pairs:
            memory.boosted_score = score
            out.append(memory)
        return out

    async def _seed(self, storage_backend, ws):
        old = await _mem(storage_backend, ws, "I live in Boston.")
        new = await _mem(storage_backend, ws, "I moved to Seattle.")
        await storage_backend.create_contradiction(
            ContradictionRecord(
                workspace_id=ws,
                memory_a_id=new.id,
                memory_b_id=old.id,
                detection_method="llm_fused",
                newer_memory_id=new.id,
            )
        )
        return old, new

    async def test_off_changes_nothing(self, memory_service, storage_backend, ws):
        old, new = await self._seed(storage_backend, ws)
        memory_service._supersession_mode = "off"
        mems = self._scored((old, 1.0), (new, 0.9))

        out = await memory_service.apply_supersession(ws, mems)

        assert [m.boosted_score for m in out] == [1.0, 0.9]

    async def test_demote_lowers_the_stale_score_only(self, memory_service, storage_backend, ws):
        old, new = await self._seed(storage_backend, ws)
        memory_service._supersession_mode = "demote"
        memory_service._supersession_penalty = 0.5
        mems = self._scored((old, 1.0), (new, 0.9))

        out = await memory_service.apply_supersession(ws, mems)
        by_id = {m.id: m.boosted_score for m in out}

        assert by_id[old.id] == 0.5, "stale memory demoted"
        assert by_id[new.id] == 0.9, "current memory untouched"

    async def test_demote_can_flip_the_ordering(self, memory_service, storage_backend, ws):
        """The point of demoting over the full pool before truncation."""
        old, new = await self._seed(storage_backend, ws)
        memory_service._supersession_mode = "demote"
        memory_service._supersession_penalty = 0.5
        mems = self._scored((old, 1.0), (new, 0.9))

        out = sorted(await memory_service.apply_supersession(ws, mems), key=lambda m: m.boosted_score, reverse=True)

        assert out[0].id == new.id, "current memory should now outrank the stale one"

    async def test_exclude_drops_the_stale_memory(self, memory_service, storage_backend, ws):
        old, new = await self._seed(storage_backend, ws)
        memory_service._supersession_mode = "exclude"
        mems = self._scored((old, 1.0), (new, 0.9))

        out = await memory_service.apply_supersession(ws, mems)

        assert [m.id for m in out] == [new.id]

    async def test_demote_records_the_evidence_signal(self, memory_service, storage_backend, ws):
        old, _ = await self._seed(storage_backend, ws)
        memory_service._supersession_mode = "demote"
        signals: dict[str, set[str]] = {}
        mems = self._scored((old, 1.0))

        await memory_service.apply_supersession(ws, mems, signals)

        assert "superseded" in signals.get(old.id, set())

    async def test_lookup_failure_degrades_to_no_op(self, memory_service, ws):
        """A stale result beats losing recall entirely."""

        class Boom:
            async def get_superseded_memory_ids(self, *a, **k):
                raise RuntimeError("storage down")

        memory_service._supersession_mode = "demote"
        original, memory_service.storage = memory_service.storage, Boom()
        try:
            mems = self._scored((await _mem(original, ws, "anything"), 1.0))
            out = await memory_service.apply_supersession(ws, mems)
            assert [m.boosted_score for m in out] == [1.0]
        finally:
            memory_service.storage = original
