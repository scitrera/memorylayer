"""Tests for _global_user workspace functionality.

Mirrors test_global_workspace.py but covers the user-scoped union-across branch
that lets a user's profile/preferences follow them across workspaces without
leaking across users.
"""

import pytest
import pytest_asyncio

from memorylayer_server.config import (
    DEFAULT_TENANT_ID,
    GLOBAL_USER_WORKSPACE_ID,
)
from memorylayer_server.models import (
    MemoryType,
    RecallInput,
    RecallMode,
    RememberInput,
    Workspace,
)


@pytest_asyncio.fixture
async def test_workspace(storage_backend, unique_workspace_id):
    """Create a test workspace."""
    workspace = Workspace(
        id=unique_workspace_id,
        tenant_id=DEFAULT_TENANT_ID,
        name="Test Workspace",
    )
    return await storage_backend.create_workspace(workspace)


@pytest_asyncio.fixture
async def global_user_workspace(storage_backend):
    """Create the _global_user workspace."""
    existing = await storage_backend.get_workspace(GLOBAL_USER_WORKSPACE_ID)
    if existing:
        return existing

    workspace = Workspace(
        id=GLOBAL_USER_WORKSPACE_ID,
        tenant_id=DEFAULT_TENANT_ID,
        name="Global User Workspace",
        description="User-scoped global workspace for profile/preference memories",
    )
    return await storage_backend.create_workspace(workspace)


@pytest.mark.asyncio
async def test_global_user_workspace_id_constant():
    """The constant is the agreed sentinel id."""
    assert GLOBAL_USER_WORKSPACE_ID == "_global_user"


@pytest.mark.asyncio
async def test_recall_input_include_global_user_default_true():
    """Default opt-in: preferences flow to recall unless caller opts out."""
    recall_input = RecallInput(query="x")
    assert recall_input.include_global_user is True


@pytest.mark.asyncio
async def test_recall_input_include_global_user_can_be_false():
    recall_input = RecallInput(query="x", include_global_user=False)
    assert recall_input.include_global_user is False


@pytest.mark.asyncio
async def test_recall_includes_global_user_workspace(memory_service, test_workspace, global_user_workspace):
    """A user's profile memory in _global_user surfaces when recalling in
    another workspace as the same user."""
    user_id = "alice@example.com"

    user_global_memory = await memory_service.remember(
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        input=RememberInput(
            content="Alice prefers dark mode",
            type=MemoryType.SEMANTIC,
            user_id=user_id,
        ),
    )

    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Alice prefers dark mode",
            mode=RecallMode.RAG,
            limit=10,
            user_id=user_id,
            min_relevance=0.0,
        ),
    )

    memory_ids = {m.id for m in result.memories}
    assert user_global_memory.id in memory_ids


@pytest.mark.asyncio
async def test_recall_can_exclude_global_user_workspace(memory_service, test_workspace, global_user_workspace):
    """include_global_user=False prevents cross-workspace leakage."""
    user_id = "alice@example.com"

    user_global_memory = await memory_service.remember(
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        input=RememberInput(
            content="Alice prefers dark mode",
            type=MemoryType.SEMANTIC,
            user_id=user_id,
        ),
    )

    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Alice prefers dark mode",
            mode=RecallMode.RAG,
            limit=10,
            user_id=user_id,
            include_global_user=False,
            include_global=False,
            min_relevance=0.0,
        ),
    )

    memory_ids = {m.id for m in result.memories}
    assert user_global_memory.id not in memory_ids


@pytest.mark.asyncio
async def test_recall_global_user_is_user_isolated(memory_service, test_workspace, global_user_workspace):
    """Alice's user-global memory must NOT be recalled as Bob."""
    alice_memory = await memory_service.remember(
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        input=RememberInput(
            content="Alice prefers dark mode",
            type=MemoryType.SEMANTIC,
            user_id="alice@example.com",
        ),
    )

    # Bob recalls with same query + same cross-workspace flag
    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Alice prefers dark mode",
            mode=RecallMode.RAG,
            limit=10,
            user_id="bob@example.com",
            min_relevance=0.0,
        ),
    )

    memory_ids = {m.id for m in result.memories}
    assert alice_memory.id not in memory_ids


@pytest.mark.asyncio
async def test_recall_global_user_requires_user_id(memory_service, test_workspace, global_user_workspace):
    """Without a user_id, _global_user union is skipped (no leak-everything)."""
    alice_memory = await memory_service.remember(
        workspace_id=GLOBAL_USER_WORKSPACE_ID,
        input=RememberInput(
            content="Alice prefers dark mode",
            type=MemoryType.SEMANTIC,
            user_id="alice@example.com",
        ),
    )

    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Alice prefers dark mode",
            mode=RecallMode.RAG,
            limit=10,
            # user_id intentionally omitted
            min_relevance=0.0,
        ),
    )

    memory_ids = {m.id for m in result.memories}
    assert alice_memory.id not in memory_ids
