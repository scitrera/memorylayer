"""Tests for the fused-LLM contradiction service.

The detector replaces a pairwise regex test with ONE call per stored memory that judges
the whole candidate neighbourhood. The behaviours worth pinning are the ones that were
measured or that failed during evaluation:

  * a superseding update is detected where the regex detector sees nothing
  * a TRUNCATED reply must not read as "no contradictions" — that failure mode is
    indistinguishable from a clean negative and silently scored 0% during the eval
  * an exact restatement short-circuits before spending a call
  * a missing/failing LLM degrades to no-detection, never blocking ingest
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.services.contradiction.default import DefaultContradictionService
from memorylayer_server.services.contradiction.llm import (
    DETECTION_METHOD,
    LLM_PROFILE_CONTRADICTION,
    LLMContradictionService,
    _parse_contradicted,
)
from memorylayer_server.services.llm import LLMNotConfiguredError


def _llm(content: str = '{"duplicate_facts": [], "contradicted_facts": [0]}', finish_reason: str = "stop"):
    svc = AsyncMock()
    resp = MagicMock()
    resp.content = content
    resp.finish_reason = finish_reason
    svc.complete.return_value = resp
    return svc


@pytest_asyncio.fixture
async def ws(storage_backend, unique_workspace_id):
    """Isolated workspace per test.

    The shared `workspace_id` fixture is literally "default" over a shared backend, so
    memories accumulate across tests in a session. That matters more here than usual: this
    provider deliberately runs a LOWER relevance floor (0.5), which widens the candidate
    neighbourhood enough for another test's memories to be picked up as candidates.
    """
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace

    if not await storage_backend.get_workspace(unique_workspace_id):
        await storage_backend.create_workspace(
            Workspace(
                id=unique_workspace_id,
                tenant_id="default_tenant",
                name="llm contradiction test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    return unique_workspace_id


async def _mem(storage_backend, embedding_service, workspace_id, content, embedding_text=None):
    embedding = (await embedding_service.embed_batch([embedding_text or content]))[0]
    mem = await storage_backend.create_memory(workspace_id, RememberInput(content=content, importance=0.8))
    return await storage_backend.update_memory(workspace_id, mem.id, embedding=embedding)


class TestParseContradicted:
    def test_plain_json(self):
        assert _parse_contradicted('{"duplicate_facts":[0],"contradicted_facts":[1,2]}') == [1, 2]

    def test_fenced_json(self):
        assert _parse_contradicted('thinking...\n```json\n{"contradicted_facts":[3]}\n```') == [3]

    def test_prose_prefixed_json(self):
        assert _parse_contradicted('Let me reason. Final answer: {"contradicted_facts": [0]}') == [0]

    def test_truncated_reasoning_yields_nothing(self):
        """A reasoning model cut off mid-chain emits no JSON at all."""
        assert _parse_contradicted("The user wants me to compare the NEW FACT with") == []

    def test_garbage_and_empty(self):
        assert _parse_contradicted("no json here") == []
        assert _parse_contradicted("") == []

    def test_non_integer_indices_skipped(self):
        assert _parse_contradicted('{"contradicted_facts": [0, "x", null, 2]}') == [0, 2]


class TestDefaults:
    def test_llm_provider_uses_a_lower_floor_than_regex(self, storage_backend):
        """The floor moves WITH the better detector, never independently.

        51.3% of supersession pairs are reachable at 0.7 vs 93.6% at 0.5, but a lower
        floor also shows a weak detector more pairs to be wrong about — so only the
        provider that can absorb it gets the lower default.
        """
        regex = DefaultContradictionService(storage=storage_backend)
        fused = LLMContradictionService(storage=storage_backend, llm_service=_llm())

        assert regex.min_relevance == 0.7
        assert fused.min_relevance == 0.5
        assert fused.candidate_limit == regex.candidate_limit == 20


@pytest.mark.asyncio
class TestFusedDetection:
    async def test_detects_supersession_the_regex_detector_misses(self, storage_backend, embedding_service, ws):
        """A value update carries no negation words, so regex sees nothing here."""
        shared = "personal best 5k time"
        old = await _mem(storage_backend, embedding_service, ws, "My personal best 5K time is 27:12.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I set a new 5K personal best of 25:50.", shared)

        assert not DefaultContradictionService._has_negation_pattern(old.content, new.content)

        service = LLMContradictionService(storage=storage_backend, llm_service=_llm())
        contradictions = await service.check_new_memory(ws, new.id)

        assert len(contradictions) == 1
        record = contradictions[0]
        assert record.memory_a_id == new.id
        assert record.memory_b_id == old.id
        assert record.detection_method == DETECTION_METHOD
        assert record.newer_memory_id is not None

    async def test_uses_the_contradiction_profile(self, storage_backend, embedding_service, ws):
        """Must route through its own activity so it can be pointed at a cheap model."""
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I moved to Seattle.", shared)

        llm = _llm()
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)
        await service.check_new_memory(ws, new.id)

        assert llm.complete.call_args.kwargs["profile"] == LLM_PROFILE_CONTRADICTION

    async def test_truncated_reply_is_not_a_clean_negative(self, storage_backend, embedding_service, ws):
        """finish_reason='length' must be treated as no-result, and must be logged.

        Reasoning models emit a long chain before the JSON. During evaluation an
        undersized token budget truncated every call and scored a confident 0%.
        """
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I moved to Seattle.", shared)

        llm = _llm(content="The user wants me to compare", finish_reason="length")
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)
        service.logger = MagicMock()

        assert await service.check_new_memory(ws, new.id) == []
        assert service.logger.warning.called, "truncation must be surfaced, not swallowed"

    async def test_exact_restatement_short_circuits_without_calling_llm(self, storage_backend, embedding_service, ws):
        """A verbatim repeat is a duplicate, never a contradiction — and needs no call."""
        shared = "same thing"
        await _mem(storage_backend, embedding_service, ws, "I have two cats.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I  HAVE   two cats.", shared)

        llm = _llm()
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)

        assert await service.check_new_memory(ws, new.id) == []
        llm.complete.assert_not_called()

    async def test_llm_not_configured_degrades_quietly(self, storage_backend, embedding_service, ws):
        """Detection runs inside the post-store pipeline; it must never block ingest."""
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I moved to Seattle.", shared)

        llm = AsyncMock()
        llm.complete.side_effect = LLMNotConfiguredError("no llm")
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)

        assert await service.check_new_memory(ws, new.id) == []

    async def test_llm_exception_degrades_quietly(self, storage_backend, embedding_service, ws):
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I moved to Seattle.", shared)

        llm = AsyncMock()
        llm.complete.side_effect = RuntimeError("upstream 500")
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)

        assert await service.check_new_memory(ws, new.id) == []

    async def test_out_of_range_index_is_ignored(self, storage_backend, embedding_service, ws):
        """A model naming a candidate that does not exist must not create a bogus record."""
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "I moved to Seattle.", shared)

        service = LLMContradictionService(
            storage=storage_backend,
            llm_service=_llm(content='{"contradicted_facts": [99, -1]}'),
        )

        assert await service.check_new_memory(ws, new.id) == []

    async def test_no_contradiction_reported_when_model_says_none(self, storage_backend, embedding_service, ws):
        shared = "shared topic"
        await _mem(storage_backend, embedding_service, ws, "I live in Boston.", shared)
        new = await _mem(storage_backend, embedding_service, ws, "Boston winters are rough.", shared)

        service = LLMContradictionService(
            storage=storage_backend,
            llm_service=_llm(content='{"duplicate_facts": [], "contradicted_facts": []}'),
        )

        assert await service.check_new_memory(ws, new.id) == []

    async def test_memory_without_embedding_makes_no_call(self, storage_backend, ws):
        mem = await storage_backend.create_memory(ws, RememberInput(content="no embedding", importance=0.5))
        llm = _llm()
        service = LLMContradictionService(storage=storage_backend, llm_service=llm)

        assert await service.check_new_memory(ws, mem.id) == []
        llm.complete.assert_not_called()
