"""
Unit tests for Phase 1: OBO identity model and Aether header parsing.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.models.auth import (
    AuthorityContext,
    PrincipalRef,
    RequestContext,
)
from memorylayer_server.services.authentication.aether import (
    HEADER_AUTH_ACTOR_ID,
    HEADER_AUTH_ACTOR_TYPE,
    HEADER_AUTH_AUTHORITY_MODE,
    HEADER_AUTH_GRANT_ID,
    HEADER_AUTH_MAX_ACCESS_LEVEL,
    HEADER_AUTH_ROOT_SUBJECT_ID,
    HEADER_AUTH_ROOT_SUBJECT_TYPE,
    HEADER_AUTH_SUBJECT_ID,
    HEADER_AUTH_SUBJECT_TYPE,
    HEADER_AUTH_TENANT_ID,
    HEADER_AUTH_USER_ID,
    HEADER_AUTH_WORKSPACE_ACCESS,
    HEADER_AUTH_WORKSPACE_SCOPE,
    META_AUTHORITY_MODE,
    META_GRANT_ID,
    META_GRANT_MAX_ACCESS_LEVEL,
    AetherAuthenticationService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(headers: dict) -> MagicMock:
    """Build a mock FastAPI Request with the given headers."""
    req = MagicMock()
    req.headers = headers
    return req


def _make_service() -> AetherAuthenticationService:
    session_service = MagicMock()
    session_service.get = AsyncMock(side_effect=Exception("not found"))
    workspace_service = MagicMock()
    workspace_service.ensure_workspace = AsyncMock(return_value=None)
    return AetherAuthenticationService(
        session_service=session_service,
        workspace_service=workspace_service,
        implicit_session_create=False,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestPrincipalRef:
    def test_fields(self):
        p = PrincipalRef(type="user", id="alice")
        assert p.type == "user"
        assert p.id == "alice"


class TestAuthorityContext:
    def test_defaults_to_direct(self):
        a = AuthorityContext()
        assert a.mode == "direct"
        assert a.grant_id is None
        assert a.subject is None

    def test_obo_fields(self):
        subject = PrincipalRef(type="user", id="alice")
        a = AuthorityContext(
            mode="on_behalf_of",
            grant_id="g_abc",
            subject=subject,
            max_access_level=10,
            workspace_scope=["ws_1", "ws_2"],
        )
        assert a.mode == "on_behalf_of"
        assert a.grant_id == "g_abc"
        assert a.subject.id == "alice"
        assert a.max_access_level == 10
        assert a.workspace_scope == ["ws_1", "ws_2"]


class TestRequestContextEffectiveSubject:
    def test_direct_mode_returns_user_id(self):
        ctx = RequestContext(
            tenant_id="t1",
            workspace_id="ws1",
            user_id="alice",
            authority=AuthorityContext(mode="direct"),
        )
        assert ctx.effective_subject_id() == "alice"

    def test_obo_mode_returns_subject_id(self):
        ctx = RequestContext(
            tenant_id="t1",
            workspace_id="ws1",
            user_id="alice",  # shim — should be overridden
            authority=AuthorityContext(
                mode="on_behalf_of",
                subject=PrincipalRef(type="user", id="alice"),
            ),
        )
        assert ctx.effective_subject_id() == "alice"

    def test_no_authority_returns_user_id(self):
        ctx = RequestContext(tenant_id="t1", workspace_id="ws1", user_id="bob")
        assert ctx.effective_subject_id() == "bob"

    def test_obo_authority_no_subject_falls_back_to_user_id(self):
        ctx = RequestContext(
            tenant_id="t1",
            workspace_id="ws1",
            user_id="bob",
            authority=AuthorityContext(mode="on_behalf_of", subject=None),
        )
        assert ctx.effective_subject_id() == "bob"


# ---------------------------------------------------------------------------
# AetherAuthenticationService._extract_authority_context tests
# ---------------------------------------------------------------------------


class TestExtractAuthorityContext:
    def setup_method(self):
        self.svc = _make_service()

    def _identity(self, user_id=None):
        from memorylayer_server.models.auth import AuthIdentity

        return AuthIdentity(tenant_id="t1", user_id=user_id)

    def test_no_obo_headers_returns_direct(self):
        req = _make_request({})
        identity = self._identity("alice")
        actor, authority = self.svc._extract_authority_context(req, identity)
        assert authority.mode == "direct"
        # actor synthesized from identity
        assert actor is not None
        assert actor.id == "alice"

    def test_explicit_direct_mode(self):
        req = _make_request(
            {
                HEADER_AUTH_AUTHORITY_MODE: "direct",
                HEADER_AUTH_ACTOR_TYPE: "user",
                HEADER_AUTH_ACTOR_ID: "alice",
            }
        )
        identity = self._identity("alice")
        actor, authority = self.svc._extract_authority_context(req, identity)
        assert authority.mode == "direct"
        assert actor.type == "user"
        assert actor.id == "alice"

    def test_obo_mode_full_headers(self):
        req = _make_request(
            {
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_ACTOR_TYPE: "service",
                HEADER_AUTH_ACTOR_ID: "sv.platform-api",
                HEADER_AUTH_GRANT_ID: "g_abc123",
                HEADER_AUTH_SUBJECT_TYPE: "user",
                HEADER_AUTH_SUBJECT_ID: "alice",
                HEADER_AUTH_ROOT_SUBJECT_TYPE: "user",
                HEADER_AUTH_ROOT_SUBJECT_ID: "alice",
                HEADER_AUTH_MAX_ACCESS_LEVEL: "10",
                HEADER_AUTH_WORKSPACE_SCOPE: "ws_1,ws_2",
            }
        )
        identity = self._identity("alice")
        actor, authority = self.svc._extract_authority_context(req, identity)

        assert actor.type == "service"
        assert actor.id == "sv.platform-api"

        assert authority.mode == "on_behalf_of"
        assert authority.grant_id == "g_abc123"
        assert authority.subject.type == "user"
        assert authority.subject.id == "alice"
        assert authority.root_subject.id == "alice"
        assert authority.max_access_level == 10
        assert authority.workspace_scope == ["ws_1", "ws_2"]

    def test_obo_mode_missing_subject_falls_back_to_direct(self):
        req = _make_request(
            {
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_ACTOR_TYPE: "service",
                HEADER_AUTH_ACTOR_ID: "sv.platform-api",
                HEADER_AUTH_GRANT_ID: "g_abc123",
                # No subject headers
            }
        )
        identity = self._identity()
        actor, authority = self.svc._extract_authority_context(req, identity)
        assert authority.mode == "direct"

    def test_obo_invalid_max_access_level_ignored(self):
        req = _make_request(
            {
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_SUBJECT_TYPE: "user",
                HEADER_AUTH_SUBJECT_ID: "alice",
                HEADER_AUTH_MAX_ACCESS_LEVEL: "notanumber",
            }
        )
        identity = self._identity("alice")
        _, authority = self.svc._extract_authority_context(req, identity)
        assert authority.mode == "on_behalf_of"
        assert authority.max_access_level is None

    def test_workspace_scope_absent_means_any(self):
        req = _make_request(
            {
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_SUBJECT_TYPE: "user",
                HEADER_AUTH_SUBJECT_ID: "alice",
            }
        )
        identity = self._identity("alice")
        _, authority = self.svc._extract_authority_context(req, identity)
        assert authority.workspace_scope is None


# ---------------------------------------------------------------------------
# build_context integration (mocked workspace/session services)
# ---------------------------------------------------------------------------


class TestBuildContext:
    def setup_method(self):
        self.svc = _make_service()

    @pytest.mark.asyncio
    async def test_direct_mode_user_id_preserved(self):
        req = _make_request(
            {
                HEADER_AUTH_TENANT_ID: "tenant1",
                HEADER_AUTH_USER_ID: "alice",
                HEADER_AUTH_WORKSPACE_ACCESS: "20",
            }
        )
        ctx = await self.svc.build_context(req)
        assert ctx.tenant_id == "tenant1"
        assert ctx.user_id == "alice"
        assert ctx.authority.mode == "direct"
        assert ctx.metadata[META_AUTHORITY_MODE] == "direct"
        assert META_GRANT_ID not in ctx.metadata

    @pytest.mark.asyncio
    async def test_obo_mode_user_id_set_to_subject(self):
        req = _make_request(
            {
                HEADER_AUTH_TENANT_ID: "tenant1",
                HEADER_AUTH_USER_ID: "alice",  # proxy echoes subject here in OBO
                HEADER_AUTH_WORKSPACE_ACCESS: "20",
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_ACTOR_TYPE: "service",
                HEADER_AUTH_ACTOR_ID: "sv.platform-api",
                HEADER_AUTH_GRANT_ID: "g_abc123",
                HEADER_AUTH_SUBJECT_TYPE: "user",
                HEADER_AUTH_SUBJECT_ID: "alice",
                HEADER_AUTH_MAX_ACCESS_LEVEL: "10",
            }
        )
        ctx = await self.svc.build_context(req)
        assert ctx.user_id == "alice"
        assert ctx.actor.id == "sv.platform-api"
        assert ctx.authority.mode == "on_behalf_of"
        assert ctx.authority.grant_id == "g_abc123"
        assert ctx.authority.max_access_level == 10
        assert ctx.metadata[META_GRANT_ID] == "g_abc123"
        assert ctx.metadata[META_GRANT_MAX_ACCESS_LEVEL] == 10
        assert ctx.metadata[META_AUTHORITY_MODE] == "on_behalf_of"

    @pytest.mark.asyncio
    async def test_effective_subject_id_matches_user_id_in_obo(self):
        req = _make_request(
            {
                HEADER_AUTH_TENANT_ID: "tenant1",
                HEADER_AUTH_USER_ID: "alice",
                HEADER_AUTH_AUTHORITY_MODE: "on_behalf_of",
                HEADER_AUTH_ACTOR_TYPE: "service",
                HEADER_AUTH_ACTOR_ID: "sv.platform-api",
                HEADER_AUTH_SUBJECT_TYPE: "user",
                HEADER_AUTH_SUBJECT_ID: "alice",
            }
        )
        ctx = await self.svc.build_context(req)
        assert ctx.effective_subject_id() == ctx.user_id == "alice"

    @pytest.mark.asyncio
    async def test_no_headers_falls_back_to_default_tenant(self):
        req = _make_request({})
        ctx = await self.svc.build_context(req)
        assert ctx.tenant_id is not None  # DEFAULT_TENANT_ID
        assert ctx.authority.mode == "direct"
