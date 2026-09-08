"""Tests for the USER-scope WRITE path (Slice 1 of cross-workspace user memory).

The read side (include_global_user fan-out + forced user_id filter) is covered
by test_global_user_workspace.py. These tests cover the missing WRITE seam:
RememberInput.scope=USER routing into _global_user, the configurable
PREFERENCE/DIRECTIVE subtype mapping, the no-user-id guard, cross-user
isolation end-to-end, and byte-identical behavior when scope is unset.
"""

import pytest
import pytest_asyncio

from memorylayer_server.config import (
    DEFAULT_TENANT_ID,
    GLOBAL_USER_WORKSPACE_ID,
)
from memorylayer_server.models import (
    MemoryScope,
    MemoryType,
    RecallInput,
    RecallMode,
    RememberInput,
    Workspace,
)


@pytest_asyncio.fixture
async def test_workspace(storage_backend, unique_workspace_id):
    """Create a test workspace (origin workspace W1)."""
    workspace = Workspace(
        id=unique_workspace_id,
        tenant_id=DEFAULT_TENANT_ID,
        name="Test Workspace W1",
    )
    return await storage_backend.create_workspace(workspace)


@pytest_asyncio.fixture
async def second_workspace(storage_backend):
    """Create a second workspace (W2) to prove follows-the-user."""
    ws_id = "ws_user_scope_w2"
    existing = await storage_backend.get_workspace(ws_id)
    if existing:
        return existing
    workspace = Workspace(id=ws_id, tenant_id=DEFAULT_TENANT_ID, name="Test Workspace W2")
    return await storage_backend.create_workspace(workspace)


@pytest_asyncio.fixture
async def global_user_workspace(storage_backend):
    """Ensure the _global_user workspace exists."""
    existing = await storage_backend.get_workspace(GLOBAL_USER_WORKSPACE_ID)
    if existing:
        return existing
    workspace = Workspace(
        id=GLOBAL_USER_WORKSPACE_ID,
        tenant_id=DEFAULT_TENANT_ID,
        name="Global User Workspace",
    )
    return await storage_backend.create_workspace(workspace)


# --------------------------------------------------------------------------
# Explicit scope=USER routing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_user_routes_to_global_user_workspace(
    memory_service, test_workspace, global_user_workspace
):
    """scope=USER (user_id=A from W1) lands in _global_user with user_id=A and
    origin_workspace_id=W1 in metadata, and rode the enrichment hook."""
    # Content is unique to this test so it does not collide (via the shared
    # storage backend) with sibling tests that also write to _global_user,
    # which would trigger a dedup SKIP and return their pre-existing memory.
    memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Quentin prefers the teal accent palette in his editor",
            type=MemoryType.SEMANTIC,
            user_id="quentin@example.com",
            scope=MemoryScope.USER,
        ),
        inline=True,  # run the post-store/enrich pipeline synchronously
    )

    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    assert memory.user_id == "quentin@example.com"
    assert memory.metadata.get("origin_workspace_id") == test_workspace.id

    # The memory is actually retrievable from the user bucket (it persisted there).
    fetched = await memory_service.storage.get_memory(GLOBAL_USER_WORKSPACE_ID, memory.id)
    assert fetched is not None
    assert fetched.workspace_id == GLOBAL_USER_WORKSPACE_ID


@pytest.mark.asyncio
async def test_scope_user_follows_user_across_workspaces(
    memory_service, test_workspace, second_workspace, global_user_workspace
):
    """A scope=USER pref written from W1 surfaces when user A recalls from
    BOTH W1 and a different workspace W2 (include_global_user fan-out)."""
    pref = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Frank only uses the vim keybindings layout",
            type=MemoryType.SEMANTIC,
            user_id="frank@example.com",
            scope=MemoryScope.USER,
        ),
    )

    for ws in (test_workspace.id, second_workspace.id):
        result = await memory_service.recall(
            workspace_id=ws,
            input=RecallInput(
                query="Frank only uses the vim keybindings layout",
                mode=RecallMode.RAG,
                limit=10,
                user_id="frank@example.com",
                min_relevance=0.0,
            ),
        )
        assert pref.id in {m.id for m in result.memories}, f"pref not found from workspace {ws}"


@pytest.mark.asyncio
async def test_scope_user_cross_user_isolation(
    memory_service, test_workspace, global_user_workspace
):
    """User B must NOT see user A's scope=USER memory (forced user_id filter)."""
    alice_pref = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Grace stores her API keys in a hardware token",
            type=MemoryType.SEMANTIC,
            user_id="grace@example.com",
            scope=MemoryScope.USER,
        ),
    )

    result = await memory_service.recall(
        workspace_id=test_workspace.id,
        input=RecallInput(
            query="Grace stores her API keys in a hardware token",
            mode=RecallMode.RAG,
            limit=10,
            user_id="heidi@example.com",
            min_relevance=0.0,
        ),
    )
    assert alice_pref.id not in {m.id for m in result.memories}


@pytest.mark.asyncio
async def test_scope_user_without_user_id_is_rejected(memory_service, test_workspace):
    """Explicit scope=USER with no user_id is rejected (no unfilterable row)."""
    with pytest.raises(ValueError, match="user_id"):
        await memory_service.remember(
            workspace_id=test_workspace.id,
            input=RememberInput(
                content="Someone prefers dark mode",
                type=MemoryType.SEMANTIC,
                scope=MemoryScope.USER,
            ),
        )


# --------------------------------------------------------------------------
# Configurable subtype mapping (PREFERENCE/DIRECTIVE)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preference_subtype_routes_to_user_scope_when_knob_on(
    memory_service, test_workspace, global_user_workspace
):
    """With the knob ON (default), a PREFERENCE memory routes to user-scope
    without an explicit scope."""
    assert memory_service.user_scope_subtypes_enabled is True

    memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Bob likes concise answers",
            type=MemoryType.SEMANTIC,
            subtype="preference",
            user_id="bob@example.com",
        ),
    )

    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    assert memory.user_id == "bob@example.com"
    assert memory.metadata.get("origin_workspace_id") == test_workspace.id


@pytest.mark.asyncio
async def test_preference_subtype_stays_workspace_when_knob_off(
    memory_service, test_workspace, global_user_workspace
):
    """With the knob OFF, a PREFERENCE memory stays workspace-scoped; an
    explicit scope=USER still routes to user-scope regardless of the knob."""
    # memory_service is a shared extension singleton; restore the knob so this
    # test does not leak state into sibling tests.
    original = memory_service.user_scope_subtypes_enabled
    memory_service.user_scope_subtypes_enabled = False
    try:
        workspace_memory = await memory_service.remember(
            workspace_id=test_workspace.id,
            input=RememberInput(
                content="Carol likes verbose answers",
                type=MemoryType.SEMANTIC,
                subtype="preference",
                user_id="carol@example.com",
            ),
        )
        assert workspace_memory.workspace_id == test_workspace.id
        assert "origin_workspace_id" not in workspace_memory.metadata

        # Explicit scope=USER still wins even with the knob off.
        user_memory = await memory_service.remember(
            workspace_id=test_workspace.id,
            input=RememberInput(
                content="Carol explicitly prefers user-global storage",
                type=MemoryType.SEMANTIC,
                subtype="preference",
                user_id="carol@example.com",
                scope=MemoryScope.USER,
            ),
        )
        assert user_memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    finally:
        memory_service.user_scope_subtypes_enabled = original


@pytest.mark.asyncio
async def test_preference_subtype_without_user_id_scopes_down(
    memory_service, test_workspace
):
    """Implicit subtype mapping with no user_id scopes DOWN to the origin
    workspace (not rejected, not an unfilterable global row)."""
    assert memory_service.user_scope_subtypes_enabled is True

    memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="A preference with no owner",
            type=MemoryType.SEMANTIC,
            subtype="preference",
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


@pytest.mark.asyncio
async def test_directive_subtype_routes_to_user_scope_when_knob_on(
    memory_service, test_workspace, global_user_workspace
):
    """DIRECTIVE is also a user-global subtype and routes to user-scope."""
    memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Always respond in English",
            type=MemoryType.SEMANTIC,
            subtype="directive",
            user_id="dave@example.com",
        ),
    )
    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID


# --------------------------------------------------------------------------
# Default (scope unset) is byte-identical to today's workspace behavior
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_scope_stays_workspace_scoped(
    memory_service, test_workspace, global_user_workspace
):
    """Default (scope unset, non-user-global subtype) stays in the origin
    workspace with no origin_workspace_id stamping — unchanged behavior."""
    memory = await memory_service.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="The build uses cmake",
            type=MemoryType.SEMANTIC,
            subtype="solution",
            user_id="erin@example.com",
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


@pytest.mark.asyncio
async def test_route_user_scope_is_noop_when_unset(memory_service):
    """The routing seam returns the inputs UNCHANGED (same object identity for
    the input) when scope is unset and the subtype is not user-global — proving
    the default path is byte-identical. (_route_user_scope became async in
    Slice 2 to host the optional classifier; the noop path still returns the
    same input object.)"""
    ws = "some_workspace"
    original = RememberInput(content="x", subtype="solution", user_id="z")
    routed_ws, routed_input = await memory_service._route_user_scope(ws, original, None)
    assert routed_ws == ws
    assert routed_input is original  # not even copied


@pytest.mark.asyncio
async def test_route_user_scope_noop_for_user_bucket_passthrough(memory_service):
    """A caller that already targets _global_user is left untouched (the legacy
    manual path remains valid and is not double-routed)."""
    original = RememberInput(content="x", user_id="z", scope=MemoryScope.USER)
    routed_ws, routed_input = await memory_service._route_user_scope(
        GLOBAL_USER_WORKSPACE_ID, original, None
    )
    assert routed_ws == GLOBAL_USER_WORKSPACE_ID
    assert routed_input is original
