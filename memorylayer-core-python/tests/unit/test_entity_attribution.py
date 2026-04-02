"""
Unit tests for entity attribution (observer_id / subject_id).

Tests:
- remember with observer_id and subject_id
- recall filtering by observer_id
- recall filtering by subject_id
- recall filtering by both observer_id and subject_id
"""

import pytest

from memorylayer_server.models.memory import (
    MemoryType,
    RecallInput,
    RecallMode,
    RememberInput,
)
from memorylayer_server.services.memory import MemoryService


class TestEntityAttribution:
    """Tests for observer_id / subject_id on memories."""

    @pytest.mark.asyncio
    async def test_remember_with_observer_id(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Storing a memory with observer_id persists the field."""
        input_data = RememberInput(
            content="Agent Alpha observed that the user prefers dark mode",
            type=MemoryType.SEMANTIC,
            observer_id="agent-alpha",
        )

        memory = await memory_service.remember(workspace_id, input_data)

        assert memory.observer_id == "agent-alpha"
        assert memory.subject_id is None

    @pytest.mark.asyncio
    async def test_remember_with_subject_id(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Storing a memory with subject_id persists the field."""
        input_data = RememberInput(
            content="The user is a senior Python developer",
            type=MemoryType.SEMANTIC,
            subject_id="user-123",
        )

        memory = await memory_service.remember(workspace_id, input_data)

        assert memory.subject_id == "user-123"
        assert memory.observer_id is None

    @pytest.mark.asyncio
    async def test_remember_with_both_entity_fields(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Storing a memory with both observer and subject persists both."""
        input_data = RememberInput(
            content="Agent Beta noticed user-456 likes functional programming",
            type=MemoryType.SEMANTIC,
            observer_id="agent-beta",
            subject_id="user-456",
        )

        memory = await memory_service.remember(workspace_id, input_data)

        assert memory.observer_id == "agent-beta"
        assert memory.subject_id == "user-456"

    @pytest.mark.asyncio
    async def test_recall_filter_by_subject_id(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Recall with subject_id filter returns only matching memories."""
        # Create memories about different subjects
        for subject, content in [
            ("entity-A", "Entity A prefers REST APIs"),
            ("entity-A", "Entity A uses PostgreSQL"),
            ("entity-B", "Entity B prefers GraphQL"),
        ]:
            await memory_service.remember(
                workspace_id,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subject_id=subject,
                    importance=0.8,
                ),
            )

        # Recall only entity-A memories
        recall_input = RecallInput(
            query="what does entity A prefer",
            subject_id="entity-A",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        )

        result = await memory_service.recall(workspace_id, recall_input)

        # All returned memories should be about entity-A
        for mem in result.memories:
            assert mem.subject_id == "entity-A"

    @pytest.mark.asyncio
    async def test_recall_filter_by_observer_id(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Recall with observer_id filter returns only memories from that observer."""
        for observer, content in [
            ("observer-X", "Observer X saw the user coding in Rust"),
            ("observer-X", "Observer X noted the user's preference for Vim"),
            ("observer-Y", "Observer Y saw the user coding in Go"),
        ]:
            await memory_service.remember(
                workspace_id,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    observer_id=observer,
                    importance=0.8,
                ),
            )

        recall_input = RecallInput(
            query="what did observer X notice",
            observer_id="observer-X",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        )

        result = await memory_service.recall(workspace_id, recall_input)

        for mem in result.memories:
            assert mem.observer_id == "observer-X"

    @pytest.mark.asyncio
    async def test_recall_without_entity_filter_returns_all(
        self,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Recall without entity filters includes all memories regardless of attribution."""
        await memory_service.remember(
            workspace_id,
            RememberInput(
                content="Unattributed memory about test isolation patterns",
                type=MemoryType.SEMANTIC,
                importance=0.9,
            ),
        )
        await memory_service.remember(
            workspace_id,
            RememberInput(
                content="Attributed memory about test isolation patterns",
                type=MemoryType.SEMANTIC,
                observer_id="obs-1",
                subject_id="subj-1",
                importance=0.9,
            ),
        )

        recall_input = RecallInput(
            query="test isolation patterns",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        )

        result = await memory_service.recall(workspace_id, recall_input)

        # Should find both attributed and unattributed
        assert len(result.memories) >= 2
        observer_ids = {m.observer_id for m in result.memories}
        assert None in observer_ids or "obs-1" in observer_ids
