"""Unit tests for SkillsResolutionService precedence and shadowing logic."""
from datetime import UTC, datetime, timedelta

import pytest

from memorylayer_server.models.skill import Skill
from memorylayer_server.services.skills.resolution import RequestContext, SkillsResolutionService

_GLOBAL_WS = "_global"


def _make_skill(
    name: str,
    workspace_id: str,
    user_id: str | None = None,
    source_mode: str = "server",
    updated_offset_seconds: int = 0,
) -> Skill:
    now = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=updated_offset_seconds)
    return Skill(
        id=f"skl_{name}_{workspace_id}_{user_id or 'ws'}",
        workspace_id=workspace_id,
        name=name,
        description=f"Skill {name}",
        source_mode=source_mode,
        updated_at=now,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Stub storage for resolution tests
# ---------------------------------------------------------------------------

class _StubStorage:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills

    async def find_skills_by_name(self, name: str, scope_filters: list[dict]) -> list[Skill]:
        results = []
        seen_ids: set[str] = set()
        for sf in scope_filters:
            ws = sf["workspace_id"]
            uid = sf.get("user_id")
            for s in self._skills:
                if s.id in seen_ids:
                    continue
                if s.name != name or s.workspace_id != ws:
                    continue
                if uid is not None:
                    # user-scope filter: only match skills owned by this user
                    if s.user_id == uid:
                        results.append(s)
                        seen_ids.add(s.id)
                else:
                    # workspace/global scope filter: skip user-private skills
                    if s.user_id is None:
                        results.append(s)
                        seen_ids.add(s.id)
        return results


# ---------------------------------------------------------------------------
# Tests: scope precedence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_scope_wins_over_workspace():
    user_skill = _make_skill("pdf-extract", workspace_id="ws1", user_id="alice")
    user_skill = user_skill.model_copy(update={"user_id": "alice"})
    ws_skill = _make_skill("pdf-extract", workspace_id="ws1")

    storage = _StubStorage([user_skill, ws_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")

    result = await svc.resolve("pdf-extract", ctx)
    assert result is not None
    assert result.user_id == "alice"


@pytest.mark.asyncio
async def test_workspace_wins_over_global():
    ws_skill = _make_skill("pdf-extract", workspace_id="ws1")
    global_skill = _make_skill("pdf-extract", workspace_id=_GLOBAL_WS)

    storage = _StubStorage([ws_skill, global_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("pdf-extract", ctx)
    assert result is not None
    assert result.workspace_id == "ws1"


@pytest.mark.asyncio
async def test_global_returned_when_no_workspace_match():
    global_skill = _make_skill("pdf-extract", workspace_id=_GLOBAL_WS)

    storage = _StubStorage([global_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("pdf-extract", ctx)
    assert result is not None
    assert result.workspace_id == _GLOBAL_WS


@pytest.mark.asyncio
async def test_returns_none_when_no_match():
    storage = _StubStorage([])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("nonexistent", ctx)
    assert result is None


# ---------------------------------------------------------------------------
# Tests: source_mode tie-breaker within same scope
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_server_beats_mirrored_same_scope():
    server_skill = _make_skill("tool", workspace_id="ws1", source_mode="server")
    mirrored_skill = _make_skill("tool", workspace_id="ws1", source_mode="mirrored")
    # Give same IDs that differ by mode
    server_skill = server_skill.model_copy(update={"id": "skl_server"})
    mirrored_skill = mirrored_skill.model_copy(update={"id": "skl_mirror"})

    storage = _StubStorage([mirrored_skill, server_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("tool", ctx)
    assert result.id == "skl_server"


@pytest.mark.asyncio
async def test_mirrored_beats_filesystem():
    mirrored = _make_skill("tool", workspace_id="ws1", source_mode="mirrored")
    fs = _make_skill("tool", workspace_id="ws1", source_mode="filesystem")
    mirrored = mirrored.model_copy(update={"id": "skl_mirror"})
    fs = fs.model_copy(update={"id": "skl_fs"})

    storage = _StubStorage([fs, mirrored])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("tool", ctx)
    assert result.id == "skl_mirror"


@pytest.mark.asyncio
async def test_most_recent_wins_same_mode():
    older = _make_skill("tool", workspace_id="ws1", updated_offset_seconds=0)
    newer = _make_skill("tool", workspace_id="ws1", updated_offset_seconds=100)
    older = older.model_copy(update={"id": "skl_old"})
    newer = newer.model_copy(update={"id": "skl_new"})

    storage = _StubStorage([older, newer])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1")

    result = await svc.resolve("tool", ctx)
    assert result.id == "skl_new"


# ---------------------------------------------------------------------------
# Tests: user isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_a_skill_not_visible_to_user_b():
    """User A's user-scoped skill must not be returned for user B."""
    alice_skill = _make_skill("private-skill", workspace_id="ws1")
    alice_skill = alice_skill.model_copy(update={"user_id": "alice"})

    storage = _StubStorage([alice_skill])
    svc = SkillsResolutionService(storage)

    # User B (bob) resolves — should get nothing
    ctx_bob = RequestContext(workspace_id="ws1", user_id="bob")
    result = await svc.resolve("private-skill", ctx_bob)
    assert result is None


@pytest.mark.asyncio
async def test_user_a_skill_visible_to_user_a():
    alice_skill = _make_skill("private-skill", workspace_id="ws1")
    alice_skill = alice_skill.model_copy(update={"user_id": "alice"})

    storage = _StubStorage([alice_skill])
    svc = SkillsResolutionService(storage)

    ctx_alice = RequestContext(workspace_id="ws1", user_id="alice")
    result = await svc.resolve("private-skill", ctx_alice)
    assert result is not None
    assert result.user_id == "alice"


# ---------------------------------------------------------------------------
# Tests: apply_shadowing
# ---------------------------------------------------------------------------

def test_apply_shadowing_keeps_winner_per_name():
    user_skill = _make_skill("tool", workspace_id="ws1")
    user_skill = user_skill.model_copy(update={"id": "skl_user", "user_id": "alice"})
    ws_skill = _make_skill("tool", workspace_id="ws1")
    ws_skill = ws_skill.model_copy(update={"id": "skl_ws"})

    svc = SkillsResolutionService(storage=None)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")

    result = svc.apply_shadowing([user_skill, ws_skill], ctx)
    assert len(result) == 1
    assert result[0].id == "skl_user"


def test_apply_shadowing_multiple_names():
    s1 = _make_skill("tool-a", workspace_id="ws1")
    s2 = _make_skill("tool-b", workspace_id="ws1")

    svc = SkillsResolutionService(storage=None)
    ctx = RequestContext(workspace_id="ws1")

    result = svc.apply_shadowing([s1, s2], ctx)
    names = {s.name for s in result}
    assert names == {"tool-a", "tool-b"}


def test_apply_shadowing_empty():
    svc = SkillsResolutionService(storage=None)
    ctx = RequestContext(workspace_id="ws1")
    assert svc.apply_shadowing([], ctx) == []


# ---------------------------------------------------------------------------
# Tests: scope_hint filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scope_hint_user_returns_user_variant_over_local():
    """scope_hint='user' must return the user-scoped skill even when a local
    (workspace-scoped) skill exists and would normally win without the hint."""
    user_skill = _make_skill("pdf-extract", workspace_id="ws1")
    user_skill = user_skill.model_copy(update={"id": "skl_user", "user_id": "alice"})
    ws_skill = _make_skill("pdf-extract", workspace_id="ws1")
    ws_skill = ws_skill.model_copy(update={"id": "skl_ws"})

    storage = _StubStorage([user_skill, ws_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")

    # Without scope_hint, user scope wins anyway — but let's confirm the hint
    # explicitly restricts to user scope:
    result = await svc.resolve("pdf-extract", ctx, scope_hint="user")
    assert result is not None
    assert result.id == "skl_user"


@pytest.mark.asyncio
async def test_scope_hint_user_returns_none_when_no_user_variant():
    """scope_hint='user' must return None when no user-scoped skill exists,
    even if workspace or global variants are present."""
    ws_skill = _make_skill("pdf-extract", workspace_id="ws1")

    storage = _StubStorage([ws_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")

    result = await svc.resolve("pdf-extract", ctx, scope_hint="user")
    assert result is None


@pytest.mark.asyncio
async def test_scope_hint_workspace_skips_user_scope():
    """scope_hint='workspace' must not return a user-scoped skill."""
    user_skill = _make_skill("pdf-extract", workspace_id="ws1")
    user_skill = user_skill.model_copy(update={"id": "skl_user", "user_id": "alice"})
    ws_skill = _make_skill("pdf-extract", workspace_id="ws1")
    ws_skill = ws_skill.model_copy(update={"id": "skl_ws"})

    storage = _StubStorage([user_skill, ws_skill])
    svc = SkillsResolutionService(storage)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")

    result = await svc.resolve("pdf-extract", ctx, scope_hint="workspace")
    assert result is not None
    assert result.id == "skl_ws"


# ---------------------------------------------------------------------------
# Tests: visible_scopes_for
# ---------------------------------------------------------------------------

def test_visible_scopes_with_user():
    svc = SkillsResolutionService(storage=None)
    ctx = RequestContext(workspace_id="ws1", user_id="alice")
    scopes = svc.visible_scopes_for(ctx)
    assert {"workspace_id": "ws1", "user_id": "alice"} in scopes
    assert {"workspace_id": "ws1"} in scopes
    assert {"workspace_id": "_global"} in scopes


def test_visible_scopes_no_user():
    svc = SkillsResolutionService(storage=None)
    ctx = RequestContext(workspace_id="ws1")
    scopes = svc.visible_scopes_for(ctx)
    # Should NOT include user scope
    assert all("user_id" not in s for s in scopes)
    assert {"workspace_id": "_global"} in scopes
