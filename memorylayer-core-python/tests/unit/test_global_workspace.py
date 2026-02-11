"""Tests for _global workspace functionality."""
import pytest
import pytest_asyncio
from memorylayer_server.models import (
    RememberInput,
    RecallInput,
    RecallMode,
    MemoryType,
    Workspace,
    WorkspaceSettings,
)
from memorylayer_server.config import GLOBAL_WORKSPACE_ID, DEFAULT_TENANT_ID


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
async def global_workspace(storage_backend):
    """Create the _global workspace."""
    # Check if it already exists
    existing = await storage_backend.get_workspace(GLOBAL_WORKSPACE_ID)
    if existing:
        return existing

    workspace = Workspace(
        id=GLOBAL_WORKSPACE_ID,
        tenant_id=DEFAULT_TENANT_ID,
        name="Global Workspace",
        description="Shared workspace accessible by all workspaces",
    )
    return await storage_backend.create_workspace(workspace)


@pytest.mark.asyncio
async def test_global_workspace_id_constant():
    """Test that GLOBAL_WORKSPACE_ID is correctly defined."""
    assert GLOBAL_WORKSPACE_ID == "_global"


@pytest.mark.asyncio
async def test_recall_input_include_global_default():
    """Test that RecallInput.include_global defaults to True."""
    recall_input = RecallInput(query="test query")
    assert recall_input.include_global is True


@pytest.mark.asyncio
async def test_recall_input_include_global_can_be_disabled():
    """Test that RecallInput.include_global can be set to False."""
    recall_input = RecallInput(query="test query", include_global=False)
    assert recall_input.include_global is False


@pytest.mark.asyncio
async def test_recall_searches_global_workspace_by_default(
        memory_service, test_workspace, global_workspace
):
    """Test that recall searches both workspace and _global by default."""
    # Store memory in test workspace
    workspace_memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Memory in test workspace",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Store memory in _global workspace
    global_memory = await memory_service.remember(
        workspace_id=GLOBAL_WORKSPACE_ID,
        input=RememberInput(
            content="Memory in global workspace",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Recall from test workspace (should include both)
    # Use exact content as query to ensure mock embeddings match
    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Memory in test workspace",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        ),
    )

    # Should find both memories (workspace match should rank higher due to scope boost)
    memory_ids = {m.id for m in result.memories}
    assert workspace_memory.id in memory_ids
    # Note: global workspace memory has different content, so it won't match with mock embeddings
    # Instead, search with global content
    result_global = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Memory in global workspace",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        ),
    )

    # Should find global memory when searching with its content
    global_memory_ids = {m.id for m in result_global.memories}
    assert global_memory.id in global_memory_ids


@pytest.mark.asyncio
async def test_recall_can_exclude_global_workspace(
        memory_service, test_workspace, global_workspace
):
    """Test that recall can exclude _global workspace when include_global=False."""
    # Store memory in test workspace
    workspace_memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Memory in test workspace",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Store memory in _global workspace
    global_memory = await memory_service.remember(
        workspace_id=GLOBAL_WORKSPACE_ID,
        input=RememberInput(
            content="Memory in global workspace",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Recall from test workspace with include_global=False
    # Use exact content as query to ensure mock embeddings match
    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Memory in test workspace",
            mode=RecallMode.RAG,
            limit=10,
            include_global=False,
            min_relevance=0.0,
        ),
    )

    # Should only find workspace memory
    memory_ids = {m.id for m in result.memories}
    assert workspace_memory.id in memory_ids
    assert global_memory.id not in memory_ids
    assert len(result.memories) == 1


@pytest.mark.asyncio
async def test_recall_from_global_workspace_does_not_duplicate(
        memory_service, global_workspace
):
    """Test that searching _global workspace directly doesn't duplicate results."""
    # Store memory in _global workspace
    global_memory = await memory_service.remember(
        workspace_id=GLOBAL_WORKSPACE_ID,
        input=RememberInput(
            content="Memory in global workspace",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Recall from _global workspace directly
    # Use exact content as query to ensure mock embeddings match
    result = await memory_service.recall(
        workspace_id=GLOBAL_WORKSPACE_ID,
        input=RecallInput(
            query="Memory in global workspace",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        ),
    )

    # Should only find one copy
    assert len(result.memories) == 1
    assert result.memories[0].id == global_memory.id


@pytest.mark.asyncio
async def test_scope_boosts_prioritize_workspace_over_global(
        memory_service, test_workspace, global_workspace
):
    """Test that scope boosts prioritize workspace memories over global."""
    # Store identical content in both workspaces
    workspace_memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Important information about the project",
            type=MemoryType.SEMANTIC,
        ),
    )

    global_memory = await memory_service.remember(
        workspace_id=GLOBAL_WORKSPACE_ID,
        input=RememberInput(
            content="Important information about the project",
            type=MemoryType.SEMANTIC,
        ),
    )

    # Recall from test workspace
    # Use exact content as query to ensure mock embeddings match
    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Important information about the project",
            mode=RecallMode.RAG,
            limit=10,
            min_relevance=0.0,
        ),
    )

    # Workspace memory should be ranked higher due to scope boost
    assert len(result.memories) >= 1
    # First result should be from the workspace (higher boost)
    assert result.memories[0].id == workspace_memory.id


@pytest.mark.asyncio
async def test_global_workspace_persists_across_sessions(
        storage_backend, global_workspace
):
    """Test that _global workspace can be retrieved."""
    retrieved = await storage_backend.get_workspace(GLOBAL_WORKSPACE_ID)
    assert retrieved is not None
    assert retrieved.id == GLOBAL_WORKSPACE_ID
    assert retrieved.name == "Global Workspace"
