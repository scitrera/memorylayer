"""Unit tests for McpServerResolutionService — 4-tier precedence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memorylayer_server.models.mcp_server import McpServer
from memorylayer_server.services.mcp_servers.resolution import (
    McpServerResolutionService,
    RequestContext,
    _scope_rank,
)

WS = "ws1"
GLOBAL_WS = "_global"
GLOBAL_USER_WS = "_global_user"


def _server(name: str, workspace_id: str, user_id=None, source_mode="server", offset_secs=0) -> McpServer:
    now = datetime.now(UTC) + timedelta(seconds=offset_secs)
    # Determine transport-required fields
    return McpServer(
        id=f"mcp_{name[:8].ljust(8, '0')}0000",
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        transport="stdio",
        command="npx",
        source_mode=source_mode,
        updated_at=now,
        created_at=now,
    )


class MockScopeStorage:
    def __init__(self, servers: list[McpServer]):
        self._servers = servers

    async def find_mcp_servers_by_name(self, name, scope_filters):
        results = []
        for s in self._servers:
            if s.name != name:
                continue
            for f in scope_filters:
                if s.workspace_id == f["workspace_id"]:
                    uid = f.get("user_id")
                    # A scope without user_id only matches records with user_id=None
                    if uid is None and s.user_id is None:
                        results.append(s)
                        break
                    elif uid is not None and s.user_id == uid:
                        results.append(s)
                        break
        return results


class TestScopeRank:
    def test_local_rank_is_0(self):
        s = _server("pg", WS, user_id="u1")
        assert _scope_rank(s, WS, "u1") == 0

    def test_project_rank_is_1(self):
        s = _server("pg", WS)
        assert _scope_rank(s, WS, "u1") == 1

    def test_user_rank_is_2(self):
        s = _server("pg", GLOBAL_USER_WS, user_id="u1")
        assert _scope_rank(s, WS, "u1") == 2

    def test_global_rank_is_3(self):
        s = _server("pg", GLOBAL_WS)
        assert _scope_rank(s, WS, "u1") == 3

    def test_other_users_local_is_not_0(self):
        s = _server("pg", WS, user_id="u2")
        # user_id doesn't match ctx_user_id "u1" → falls to project or higher
        rank = _scope_rank(s, WS, "u1")
        assert rank != 0


class TestResolutionService:
    @pytest.mark.asyncio
    async def test_local_beats_project(self):
        local = _server("pg", WS, user_id="u1")
        project = _server("pg", WS)
        storage = MockScopeStorage([local, project])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("pg", ctx)
        assert result.user_id == "u1"

    @pytest.mark.asyncio
    async def test_project_beats_user_scope(self):
        project = _server("pg", WS)
        user_scope = _server("pg", GLOBAL_USER_WS, user_id="u1")
        storage = MockScopeStorage([project, user_scope])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("pg", ctx)
        assert result.workspace_id == WS
        assert result.user_id is None

    @pytest.mark.asyncio
    async def test_user_scope_beats_global(self):
        user_scope = _server("pg", GLOBAL_USER_WS, user_id="u1")
        global_s = _server("pg", GLOBAL_WS)
        storage = MockScopeStorage([user_scope, global_s])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("pg", ctx)
        assert result.workspace_id == GLOBAL_USER_WS

    @pytest.mark.asyncio
    async def test_global_returned_when_no_others(self):
        global_s = _server("pg", GLOBAL_WS)
        storage = MockScopeStorage([global_s])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("pg", ctx)
        assert result.workspace_id == GLOBAL_WS

    @pytest.mark.asyncio
    async def test_all_four_tiers_local_wins(self):
        local = _server("pg", WS, user_id="u1")
        project = _server("pg", WS)
        user_scope = _server("pg", GLOBAL_USER_WS, user_id="u1")
        global_s = _server("pg", GLOBAL_WS)
        storage = MockScopeStorage([global_s, user_scope, project, local])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("pg", ctx)
        assert result.user_id == "u1"
        assert result.workspace_id == WS

    @pytest.mark.asyncio
    async def test_mode_tiebreaker_server_beats_mirrored(self):
        mirrored = _server("pg", WS, source_mode="mirrored")
        server_mode = _server("pg", WS, source_mode="server")
        storage = MockScopeStorage([mirrored, server_mode])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS)
        result = await svc.resolve("pg", ctx)
        assert result.source_mode == "server"

    @pytest.mark.asyncio
    async def test_mode_tiebreaker_mirrored_beats_filesystem(self):
        fs = _server("pg", WS, source_mode="filesystem")
        mirrored = _server("pg", WS, source_mode="mirrored")
        storage = MockScopeStorage([fs, mirrored])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS)
        result = await svc.resolve("pg", ctx)
        assert result.source_mode == "mirrored"

    @pytest.mark.asyncio
    async def test_recency_tiebreaker(self):
        older = _server("pg", WS, offset_secs=-100)
        newer = _server("pg", WS, offset_secs=0)
        storage = MockScopeStorage([older, newer])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS)
        result = await svc.resolve("pg", ctx)
        assert result.updated_at == newer.updated_at

    @pytest.mark.asyncio
    async def test_user_a_local_not_visible_to_user_b(self):
        user_a_local = _server("pg", WS, user_id="user-a")
        storage = MockScopeStorage([user_a_local])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="user-b")
        result = await svc.resolve("pg", ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_returns_none_when_no_match(self):
        storage = MockScopeStorage([])
        svc = McpServerResolutionService(storage)
        ctx = RequestContext(WS, user_id="u1")
        result = await svc.resolve("nonexistent", ctx)
        assert result is None


class TestApplyShadowing:
    def _svc(self, servers):
        return McpServerResolutionService(MockScopeStorage(servers))

    def test_shadowing_returns_winner_per_name(self):
        local = _server("pg", WS, user_id="u1")
        project = _server("pg", WS)
        svc = self._svc([local, project])
        ctx = RequestContext(WS, user_id="u1")
        result = svc.apply_shadowing([local, project], ctx)
        assert len(result) == 1
        assert result[0].user_id == "u1"

    def test_shadowing_keeps_multiple_different_names(self):
        pg = _server("pg", WS)
        redis = _server("redis", WS)
        svc = self._svc([pg, redis])
        ctx = RequestContext(WS)
        result = svc.apply_shadowing([pg, redis], ctx)
        assert len(result) == 2
        names = {s.name for s in result}
        assert names == {"pg", "redis"}

    def test_shadowing_empty_list(self):
        svc = self._svc([])
        ctx = RequestContext(WS)
        assert svc.apply_shadowing([], ctx) == []


class TestVisibleScopes:
    def test_with_user_id_includes_local_user_and_global(self):
        svc = McpServerResolutionService(None)
        ctx = RequestContext(WS, user_id="u1")
        scopes = svc.visible_scopes_for(ctx)
        ws_ids = [s["workspace_id"] for s in scopes]
        assert WS in ws_ids
        assert GLOBAL_USER_WS in ws_ids
        assert GLOBAL_WS in ws_ids

    def test_without_user_id_no_user_scopes(self):
        svc = McpServerResolutionService(None)
        ctx = RequestContext(WS)
        scopes = svc.visible_scopes_for(ctx)
        assert all("user_id" not in s for s in scopes)
        ws_ids = [s["workspace_id"] for s in scopes]
        assert GLOBAL_USER_WS not in ws_ids
