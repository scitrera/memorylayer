"""
Unit tests for Phase 2: OBO authz enforcement — grant ceiling + workspace scope.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from memorylayer_server.models.auth import AuthorityContext, PrincipalRef, RequestContext
from memorylayer_server.models.authz import AuthorizationContext, AuthorizationDecision
from memorylayer_server.services.authentication.aether import (
    AetherAuthenticationService,
    HEADER_AUTH_AUTHORITY_MODE,
    HEADER_AUTH_SUBJECT_ID,
    HEADER_AUTH_SUBJECT_TYPE,
    HEADER_AUTH_TENANT_ID,
    HEADER_AUTH_USER_ID,
    HEADER_AUTH_WORKSPACE_ACCESS,
    HEADER_AUTH_WORKSPACE_SCOPE,
    META_ACCESS_LEVEL,
    META_GRANT_MAX_ACCESS_LEVEL,
)
from memorylayer_server.services.authorization.aether import (
    ACCESS_ADMIN,
    ACCESS_READ,
    ACCESS_READWRITE,
    AetherAuthorizationService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(headers: dict) -> MagicMock:
    req = MagicMock()
    req.headers = headers
    return req


def _make_auth_service(workspace_id_for_ensure="ws_default") -> AetherAuthenticationService:
    session_service = MagicMock()
    session_service.get = AsyncMock(side_effect=Exception("not found"))
    workspace_service = MagicMock()
    workspace_service.ensure_workspace = AsyncMock(return_value=None)
    return AetherAuthenticationService(
        session_service=session_service,
        workspace_service=workspace_service,
        implicit_session_create=False,
    )


def _make_authz_service() -> AetherAuthorizationService:
    v = MagicMock()
    v.environ = MagicMock(return_value=None)
    return AetherAuthorizationService(v=None)


def _authz_ctx(access_level: int, grant_ceiling: int | None = None, resource="memories", action="read") -> AuthorizationContext:
    metadata = {META_ACCESS_LEVEL: access_level}
    if grant_ceiling is not None:
        metadata[META_GRANT_MAX_ACCESS_LEVEL] = grant_ceiling
    return AuthorizationContext(
        tenant_id="t1",
        workspace_id="ws1",
        user_id="alice",
        resource=resource,
        action=action,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Grant ceiling tests (authorization/aether.py)
# ---------------------------------------------------------------------------

class TestGrantCeilingIntersect:
    def setup_method(self):
        self.svc = _make_authz_service()

    @pytest.mark.asyncio
    async def test_no_ceiling_uses_granted_level_allow(self):
        ctx = _authz_ctx(access_level=ACCESS_READ, grant_ceiling=None, action="read")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_no_ceiling_uses_granted_level_deny(self):
        ctx = _authz_ctx(access_level=0, grant_ceiling=None, action="read")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_ceiling_caps_high_level_to_deny(self):
        # Subject has READWRITE (20) but grant ceiling is READ (10)
        # recall requires READ (10) → still allowed
        ctx = _authz_ctx(access_level=ACCESS_READWRITE, grant_ceiling=ACCESS_READ, action="read")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_ceiling_caps_write_to_deny(self):
        # Subject has READWRITE (20) but grant ceiling is READ (10)
        # remember requires READWRITE (20) → denied by ceiling
        ctx = _authz_ctx(access_level=ACCESS_READWRITE, grant_ceiling=ACCESS_READ, action="write")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_ceiling_equal_to_required_allows(self):
        # Subject has ADMIN, ceiling is exactly READWRITE — write should allow
        ctx = _authz_ctx(access_level=ACCESS_ADMIN, grant_ceiling=ACCESS_READWRITE, action="write")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.ALLOW

    @pytest.mark.asyncio
    async def test_ceiling_zero_denies_everything(self):
        ctx = _authz_ctx(access_level=ACCESS_ADMIN, grant_ceiling=0, action="read")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.DENY

    @pytest.mark.asyncio
    async def test_admin_route_denied_when_ceiling_below_admin(self):
        ctx = _authz_ctx(access_level=ACCESS_ADMIN, grant_ceiling=ACCESS_READWRITE, resource="admin", action="read")
        decision = await self.svc.authorize(ctx)
        assert decision == AuthorizationDecision.DENY


# ---------------------------------------------------------------------------
# Workspace scope enforcement (authentication/aether.py::resolve_workspace)
# ---------------------------------------------------------------------------

class TestWorkspaceScopeEnforcement:
    def setup_method(self):
        self.svc = _make_auth_service()

    @pytest.mark.asyncio
    async def test_no_authority_no_scope_check(self):
        ws = await self.svc.resolve_workspace("ws_1", None, "t1", authority=None)
        assert ws == "ws_1"

    @pytest.mark.asyncio
    async def test_direct_mode_no_scope_check(self):
        authority = AuthorityContext(mode="direct", workspace_scope=["ws_allowed"])
        ws = await self.svc.resolve_workspace("ws_other", None, "t1", authority=authority)
        assert ws == "ws_other"

    @pytest.mark.asyncio
    async def test_obo_workspace_in_scope_allowed(self):
        authority = AuthorityContext(
            mode="on_behalf_of",
            workspace_scope=["ws_1", "ws_2"],
        )
        ws = await self.svc.resolve_workspace("ws_1", None, "t1", authority=authority)
        assert ws == "ws_1"

    @pytest.mark.asyncio
    async def test_obo_workspace_not_in_scope_raises_403(self):
        authority = AuthorityContext(
            mode="on_behalf_of",
            workspace_scope=["ws_allowed"],
        )
        with pytest.raises(HTTPException) as exc_info:
            await self.svc.resolve_workspace("ws_other", None, "t1", authority=authority)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_obo_wildcard_scope_allows_any_workspace(self):
        authority = AuthorityContext(
            mode="on_behalf_of",
            workspace_scope=["*"],
        )
        ws = await self.svc.resolve_workspace("ws_anything", None, "t1", authority=authority)
        assert ws == "ws_anything"

    @pytest.mark.asyncio
    async def test_obo_empty_scope_list_no_restriction(self):
        # None/empty workspace_scope means any workspace is allowed
        authority = AuthorityContext(
            mode="on_behalf_of",
            workspace_scope=None,
        )
        ws = await self.svc.resolve_workspace("ws_anything", None, "t1", authority=authority)
        assert ws == "ws_anything"


# ---------------------------------------------------------------------------
# build_context end-to-end with workspace scope
# ---------------------------------------------------------------------------

class TestBuildContextWorkspaceScope:
    def setup_method(self):
        self.svc = _make_auth_service()

    @pytest.mark.asyncio
    async def test_obo_request_with_allowed_workspace_succeeds(self):
        req = _make_request({
            HEADER_AUTH_TENANT_ID: "t1",
            HEADER_AUTH_USER_ID: "alice",
            HEADER_AUTH_WORKSPACE_ACCESS: "20",
            HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
            HEADER_AUTH_SUBJECT_TYPE: "user",
            HEADER_AUTH_SUBJECT_ID: "alice",
            HEADER_AUTH_WORKSPACE_SCOPE: "ws_allowed,ws_other",
            "X-Workspace-ID": "ws_allowed",
        })
        ctx = await self.svc.build_context(req)
        assert ctx.workspace_id == "ws_allowed"

    @pytest.mark.asyncio
    async def test_obo_request_with_disallowed_workspace_raises_403(self):
        req = _make_request({
            HEADER_AUTH_TENANT_ID: "t1",
            HEADER_AUTH_USER_ID: "alice",
            HEADER_AUTH_WORKSPACE_ACCESS: "20",
            HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
            HEADER_AUTH_SUBJECT_TYPE: "user",
            HEADER_AUTH_SUBJECT_ID: "alice",
            HEADER_AUTH_WORKSPACE_SCOPE: "ws_allowed",
            "X-Workspace-ID": "ws_forbidden",
        })
        with pytest.raises(HTTPException) as exc_info:
            await self.svc.build_context(req)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# require_authorization passes actor/authority fields through
# ---------------------------------------------------------------------------

class TestRequireAuthorizationOBOFields:
    @pytest.mark.asyncio
    async def test_actor_authority_fields_populated_in_authz_ctx(self):
        svc = _make_authz_service()

        # Patch authorize to capture the context passed to it
        captured = {}

        async def capture_authorize(ctx):
            captured["ctx"] = ctx
            return AuthorizationDecision.ALLOW

        svc.authorize = capture_authorize

        ctx = RequestContext(
            tenant_id="t1",
            workspace_id="ws1",
            user_id="alice",
            actor=PrincipalRef(type="service", id="sv.platform-api"),
            authority=AuthorityContext(
                mode="on_behalf_of",
                grant_id="g_abc",
                subject=PrincipalRef(type="user", id="alice"),
                max_access_level=ACCESS_READ,
            ),
            metadata={META_ACCESS_LEVEL: ACCESS_READWRITE},
        )

        await svc.require_authorization(ctx, "memories", "read")

        authz_ctx = captured["ctx"]
        assert authz_ctx.actor_type == "service"
        assert authz_ctx.actor_id == "sv.platform-api"
        assert authz_ctx.subject_type == "user"
        assert authz_ctx.subject_id == "alice"
        assert authz_ctx.authority_mode == "on_behalf_of"
        assert authz_ctx.grant_id == "g_abc"
        assert authz_ctx.max_access_level == ACCESS_READ
