"""Tests for alias-hop retrieval and the recall evidence contract.

Each test runs in its own freshly-created workspace so results are deterministic
and isolated from memories/associations created by other tests.
"""

import uuid
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import RecallInput, RecallMode, RememberInput
from memorylayer_server.models.workspace import Workspace


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"alias_evidence_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="alias_tenant", name="Alias/Evidence Test", created_at=now, updated_at=now)
    )
    return ws


@pytest.mark.asyncio
async def test_alias_query_retrieves_canonical_memory(memory_service, iso_ws):
    """A query using only an alias retrieves the memory whose content never mentions it."""
    await memory_service.remember(
        iso_ws,
        RememberInput(
            content="The imperial ceremonial structure of the ancient capital", metadata={"aliases": ["Mingtang", "Hall of Light"]}
        ),
    )

    result = await memory_service.recall(
        iso_ws,
        RecallInput(query="Hall of Light", mode=RecallMode.RAG, min_relevance=0.0, include_associations=False),
    )

    assert len(result.memories) == 1
    top = result.memories[0]
    # Retrieved purely via the alias (content has no "Hall of Light"), and the
    # evidence contract records both the keyword arm and the alias match.
    assert top.match_signals is not None
    assert "alias" in top.match_signals
    assert "keyword" in top.match_signals


@pytest.mark.asyncio
async def test_alias_signal_absent_when_query_matches_content_only(memory_service, iso_ws):
    """Alias boost/signal must not fire when the query matched content, not an alias."""
    await memory_service.remember(
        iso_ws,
        RememberInput(content="python backend web framework with type hints", metadata={"aliases": ["Zelda"]}),
    )

    result = await memory_service.recall(
        iso_ws,
        RecallInput(query="python web framework", mode=RecallMode.RAG, min_relevance=0.0, include_associations=False),
    )

    assert len(result.memories) == 1
    assert "alias" not in (result.memories[0].match_signals or [])


@pytest.mark.asyncio
async def test_evidence_contract_records_match_signals(memory_service, iso_ws):
    """Every recalled memory carries the signals that surfaced it."""
    await memory_service.remember(iso_ws, RememberInput(content="distributed systems consensus and raft algorithm"))

    result = await memory_service.recall(
        iso_ws,
        RecallInput(query="raft consensus algorithm", mode=RecallMode.RAG, min_relevance=0.0, include_associations=False),
    )

    assert len(result.memories) == 1
    signals = result.memories[0].match_signals
    assert signals is not None and signals  # non-empty
    assert "keyword" in signals
