"""
Aether Authentication Service for MemoryLayer.

Reads trusted X-Auth-* headers injected by the upstream Aether auth-proxy
(Go gateway). The gateway has already validated credentials; this service
simply extracts the verified identity from the forwarded headers.

When running without the gateway (e.g. local development), the service
degrades gracefully to default-tenant mode.

Headers consumed (existing):
    X-Auth-Tenant-ID        Tenant identifier (required in production)
    X-Auth-User-ID          Authenticated user identifier
    X-Auth-API-Key-ID       API key identifier for audit trails
    X-Auth-Workspace-Access Numeric access level (10/20/30/40/50)
    X-Auth-Principal-Type   Principal type (user, agent, task)
    X-Auth-Scopes           Comma-separated scope strings

OBO headers (new, injected by auth-proxy after grant validation):
    X-Auth-Actor-Type           Authenticated connection identity type
    X-Auth-Actor-ID             Authenticated connection identity id
    X-Auth-Authority-Mode       "direct" or "on_behalf_of"
    X-Auth-Grant-ID             Aether grant_id (after validation)
    X-Auth-Subject-Type         Whose authority is being exercised (type)
    X-Auth-Subject-ID           Whose authority is being exercised (id)
    X-Auth-Root-Subject-Type    Top of delegation chain (type)
    X-Auth-Root-Subject-ID      Top of delegation chain (id)
    X-Auth-Audience-Type        Grant audience binding (type)
    X-Auth-Audience-ID          Grant audience binding (id)
    X-Auth-Max-Access-Level     Grant ceiling (numeric)
    X-Auth-Workspace-Scope      Comma-separated allowed workspaces; absent = any
"""

import logging
from collections.abc import Iterable

from fastapi import Request
from pydantic import BaseModel
from scitrera_app_framework import Variables, ext_parse_bool

from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE,
    DEFAULT_TENANT_ID,
    DEFAULT_WORKSPACE_ID,
    MEMORYLAYER_SESSION_IMPLICIT_CREATE,
)
from memorylayer_server.models.auth import (
    AuthIdentity,
    AuthorityContext,
    PrincipalRef,
    RequestContext,
)
from memorylayer_server.models.session import Session
from memorylayer_server.services.authentication.base import (
    HEADER_SESSION_ID,
    AuthenticationService,
    AuthenticationServicePluginBase,
)
from memorylayer_server.services.session import EXT_SESSION_SERVICE, SessionService
from memorylayer_server.services.workspace import EXT_WORKSPACE_SERVICE, WorkspaceService

# ---------------------------------------------------------------------------
# Header constants (must match aether3-go internal/authproxy/middleware.go)
# ---------------------------------------------------------------------------
HEADER_AUTH_TENANT_ID = "X-Auth-Tenant-ID"
HEADER_AUTH_USER_ID = "X-Auth-User-ID"
HEADER_AUTH_API_KEY_ID = "X-Auth-API-Key-ID"
HEADER_AUTH_WORKSPACE_ACCESS = "X-Auth-Workspace-Access"
HEADER_AUTH_PRINCIPAL_TYPE = "X-Auth-Principal-Type"
HEADER_AUTH_SCOPES = "X-Auth-Scopes"

# OBO header constants
HEADER_AUTH_ACTOR_TYPE = "X-Auth-Actor-Type"
HEADER_AUTH_ACTOR_ID = "X-Auth-Actor-ID"
HEADER_AUTH_AUTHORITY_MODE = "X-Auth-Authority-Mode"
HEADER_AUTH_GRANT_ID = "X-Auth-Grant-ID"
HEADER_AUTH_SUBJECT_TYPE = "X-Auth-Subject-Type"
HEADER_AUTH_SUBJECT_ID = "X-Auth-Subject-ID"
HEADER_AUTH_ROOT_SUBJECT_TYPE = "X-Auth-Root-Subject-Type"
HEADER_AUTH_ROOT_SUBJECT_ID = "X-Auth-Root-Subject-ID"
HEADER_AUTH_AUDIENCE_TYPE = "X-Auth-Audience-Type"
HEADER_AUTH_AUDIENCE_ID = "X-Auth-Audience-ID"
HEADER_AUTH_MAX_ACCESS_LEVEL = "X-Auth-Max-Access-Level"
HEADER_AUTH_WORKSPACE_SCOPE = "X-Auth-Workspace-Scope"

# Metadata keys stored on RequestContext.metadata
META_ACCESS_LEVEL = "aether_access_level"
META_PRINCIPAL_TYPE = "aether_principal_type"
META_SCOPES = "aether_scopes"
META_GRANT_ID = "aether_grant_id"
META_GRANT_MAX_ACCESS_LEVEL = "aether_grant_max_access_level"
META_AUTHORITY_MODE = "aether_authority_mode"

# Default access level when the header is absent (dev/fallback)
DEFAULT_ACCESS_LEVEL = 0


class AetherAuthenticationService(AuthenticationService):
    """Authentication service that trusts Aether gateway-injected headers.

    The upstream Go auth-proxy validates credentials (API key, JWT, etc.)
    and injects ``X-Auth-*`` headers before forwarding the request.  This
    service reads those headers to build the :class:`RequestContext`.

    When the gateway headers are absent the service falls back to
    :data:`DEFAULT_TENANT_ID`, making it safe for local development
    without the gateway running.
    """

    def __init__(
        self,
        session_service: SessionService,
        workspace_service: WorkspaceService,
        implicit_session_create: bool = True,
        logger: logging.Logger | None = None,
    ):
        super().__init__(logger)
        self.session_service = session_service
        self.workspace_service = workspace_service
        self._implicit_session_create = implicit_session_create

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    async def verify_api_key(self, api_key: str | None) -> AuthIdentity:
        """Not used in Aether mode -- identity comes from gateway headers.

        Returns a default identity; the real extraction happens in
        :meth:`build_context`.
        """
        return AuthIdentity(
            tenant_id=DEFAULT_TENANT_ID,
            user_id=None,
            api_key_id=None,
        )

    async def resolve_session(self, session_id: str | None) -> Session | None:
        """Resolve session from session service.

        Returns ``None`` if session not found or expired.
        """
        if not session_id:
            return None

        try:
            return await self.session_service.get(session_id)
        except Exception as e:
            self.logger.debug("Session %s not found: %s", session_id, e)
            return None

    async def resolve_workspace(
        self,
        request_workspace_id: str | None,
        session: Session | None,
        tenant_id: str,
        authority: AuthorityContext | None = None,
    ) -> str:
        """Resolve workspace with priority order and auto-creation.

        Priority:
        1. ``request_workspace_id`` (explicit override)
        2. ``session.workspace_id`` (from session)
        3. ``DEFAULT_WORKSPACE_ID`` ("_default")

        When ``authority`` is OBO mode with a workspace_scope list, the resolved
        workspace must be in that list (or the list must contain "*") — otherwise
        raises HTTP 403.  OSS open-auth passes ``authority=None`` so this is a no-op.
        """
        workspace_id = request_workspace_id or (session.workspace_id if session else None) or DEFAULT_WORKSPACE_ID

        # Enforce workspace scope from OBO grant
        if authority is not None and authority.mode == "on_behalf_of" and authority.workspace_scope:
            if "*" not in authority.workspace_scope and workspace_id not in authority.workspace_scope:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=403,
                    detail=f"workspace '{workspace_id}' is not in grant scope",
                )

        await self.workspace_service.ensure_workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            auto_create=True,
        )

        return workspace_id

    async def ensure_session(
        self,
        session_id: str,
        workspace_id: str,
        tenant_id: str,
    ) -> Session | None:
        """Auto-create session for unknown session_id when workspace is explicit.

        Gated on ``MEMORYLAYER_SESSION_IMPLICIT_CREATE`` config flag.
        """
        if not self._implicit_session_create:
            self.logger.debug(
                "Implicit session creation disabled, skipping for session %s",
                session_id,
            )
            return None

        try:
            session = Session.create_with_ttl(
                session_id=session_id,
                workspace_id=workspace_id,
                ttl_seconds=3600,
                tenant_id=tenant_id,
                metadata={"recreated": True},
            )
            created = await self.session_service.create_session(workspace_id, session)
            self.logger.info(
                "Implicitly created session %s in workspace %s",
                session_id,
                workspace_id,
            )
            return created
        except Exception as e:
            self.logger.warning(
                "Failed to implicitly create session %s: %s",
                session_id,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # build_context override -- the core of Aether integration
    # ------------------------------------------------------------------

    async def build_context(
        self,
        request: Request,
        body: BaseModel | None = None,
    ) -> RequestContext:
        """Build :class:`RequestContext` from Aether gateway headers.

        Extracts identity from ``X-Auth-*`` headers injected by the
        upstream auth-proxy.  Falls back to default tenant when headers
        are absent (local development without the gateway).
        """
        # 1. Extract identity from gateway headers
        identity = self._extract_identity_from_headers(request)

        # 2. Extract access level and additional gateway metadata
        access_level = self._extract_access_level(request)
        principal_type = request.headers.get(HEADER_AUTH_PRINCIPAL_TYPE)
        scopes_raw = request.headers.get(HEADER_AUTH_SCOPES, "")
        scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()] if scopes_raw else []

        # 3. Parse OBO authority context
        actor, authority = self._extract_authority_context(request, identity)

        # 4. When OBO is active, user_id == subject.id (backward-compat shim)
        user_id = identity.user_id
        if authority and authority.mode == "on_behalf_of" and authority.subject:
            user_id = authority.subject.id

        # 5. Extract and resolve session
        session_id = request.headers.get(HEADER_SESSION_ID)
        session = await self.resolve_session(session_id) if session_id else None

        # 6. Extract workspace_id from (in order): request body, query string,
        # ``X-Workspace-ID`` header.  Many endpoints accept ``workspace_id``
        # as a FastAPI ``Query`` parameter (e.g. /v1/threads/{id}/messages,
        # /v1/memories, etc.), and callers like the cowork agent's TI2
        # bridge pass it that way; without this fallback resolve_workspace
        # would fall through to ``DEFAULT_WORKSPACE_ID="_default"`` and
        # then raise 403 when the OBO grant scope doesn't include
        # ``_default``.  Body and header remain supported for endpoints
        # that don't expose workspace_id as a query.
        request_workspace_id = getattr(body, "workspace_id", None) if body else None
        if not request_workspace_id:
            request_workspace_id = request.query_params.get("workspace_id")
        if not request_workspace_id:
            request_workspace_id = request.headers.get("X-Workspace-ID")

        # 7. Resolve effective workspace (with OBO scope enforcement)
        workspace_id = await self.resolve_workspace(
            request_workspace_id=request_workspace_id,
            session=session,
            tenant_id=identity.tenant_id,
            authority=authority,
        )

        # 8. Implicit session creation
        if session_id and session is None and request_workspace_id:
            session = await self.ensure_session(
                session_id,
                workspace_id,
                identity.tenant_id,
            )

        self.logger.debug(
            "Built aether context: tenant=%s, user=%s, workspace=%s, session=%s, access_level=%d, authority_mode=%s",
            identity.tenant_id,
            user_id,
            workspace_id,
            session.id if session else None,
            access_level,
            authority.mode if authority else "direct",
        )

        # Build metadata — include grant keys so Phase 2 authz can read them
        metadata: dict = {
            META_ACCESS_LEVEL: access_level,
            META_PRINCIPAL_TYPE: principal_type,
            META_SCOPES: scopes,
            META_AUTHORITY_MODE: authority.mode if authority else "direct",
        }
        if authority and authority.grant_id:
            metadata[META_GRANT_ID] = authority.grant_id
        if authority and authority.max_access_level is not None:
            metadata[META_GRANT_MAX_ACCESS_LEVEL] = authority.max_access_level

        return RequestContext(
            tenant_id=identity.tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
            actor=actor,
            authority=authority,
            session=session,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_identity_from_headers(self, request: Request) -> AuthIdentity:
        """Extract :class:`AuthIdentity` from gateway-injected headers.

        Falls back to default tenant when headers are absent.
        """
        tenant_id = request.headers.get(HEADER_AUTH_TENANT_ID)
        user_id = request.headers.get(HEADER_AUTH_USER_ID)
        api_key_id = request.headers.get(HEADER_AUTH_API_KEY_ID)

        if not tenant_id:
            self.logger.debug(
                "No %s header found; falling back to default tenant",
                HEADER_AUTH_TENANT_ID,
            )
            tenant_id = DEFAULT_TENANT_ID

        return AuthIdentity(
            tenant_id=tenant_id,
            user_id=user_id or None,
            api_key_id=api_key_id or None,
        )

    @staticmethod
    def _extract_access_level(request: Request) -> int:
        """Parse the numeric access level from the gateway header.

        Returns :data:`DEFAULT_ACCESS_LEVEL` (0) when the header is
        absent or malformed.
        """
        raw = request.headers.get(HEADER_AUTH_WORKSPACE_ACCESS)
        if not raw:
            return DEFAULT_ACCESS_LEVEL
        try:
            return int(raw)
        except (ValueError, TypeError):
            return DEFAULT_ACCESS_LEVEL

    def _extract_authority_context(
        self,
        request: Request,
        identity: AuthIdentity,
    ) -> tuple[PrincipalRef | None, AuthorityContext | None]:
        """Parse OBO authority headers into actor + AuthorityContext.

        Returns (actor, authority). Both may be None when OBO headers
        are absent (direct mode without explicit actor headers).

        When authority mode is "on_behalf_of" but required subject headers
        are missing, falls back to direct mode and logs a warning — the
        auth-proxy is the source of truth, so a malformed OBO packet is
        treated as non-OBO rather than a hard error.
        """
        actor_type = request.headers.get(HEADER_AUTH_ACTOR_TYPE)
        actor_id = request.headers.get(HEADER_AUTH_ACTOR_ID)
        authority_mode = request.headers.get(HEADER_AUTH_AUTHORITY_MODE, "direct")

        # Build actor ref if headers present
        actor: PrincipalRef | None = None
        if actor_type and actor_id:
            actor = PrincipalRef(type=actor_type, id=actor_id)
        elif identity.user_id:
            # Synthesize actor from existing identity when no explicit actor headers
            principal_type = request.headers.get(HEADER_AUTH_PRINCIPAL_TYPE, "user")
            actor = PrincipalRef(type=principal_type, id=identity.user_id)

        if authority_mode != "on_behalf_of":
            # Direct mode — no OBO context needed
            return actor, AuthorityContext(mode="direct")

        # OBO mode — extract subject and grant details
        subject_type = request.headers.get(HEADER_AUTH_SUBJECT_TYPE)
        subject_id = request.headers.get(HEADER_AUTH_SUBJECT_ID)

        if not subject_type or not subject_id:
            self.logger.warning("OBO mode requested but subject headers missing; falling back to direct")
            return actor, AuthorityContext(mode="direct")

        subject = PrincipalRef(type=subject_type, id=subject_id)

        # Optional root subject
        root_subject: PrincipalRef | None = None
        rs_type = request.headers.get(HEADER_AUTH_ROOT_SUBJECT_TYPE)
        rs_id = request.headers.get(HEADER_AUTH_ROOT_SUBJECT_ID)
        if rs_type and rs_id:
            root_subject = PrincipalRef(type=rs_type, id=rs_id)

        # Grant ceiling
        max_access_level: int | None = None
        raw_max = request.headers.get(HEADER_AUTH_MAX_ACCESS_LEVEL)
        if raw_max:
            try:
                max_access_level = int(raw_max)
            except (ValueError, TypeError):
                self.logger.warning("Invalid %s header value: %s", HEADER_AUTH_MAX_ACCESS_LEVEL, raw_max)

        # Workspace scope
        workspace_scope: list[str] | None = None
        raw_scope = request.headers.get(HEADER_AUTH_WORKSPACE_SCOPE)
        if raw_scope:
            workspace_scope = [s.strip() for s in raw_scope.split(",") if s.strip()]

        authority = AuthorityContext(
            mode="on_behalf_of",
            grant_id=request.headers.get(HEADER_AUTH_GRANT_ID) or None,
            subject=subject,
            root_subject=root_subject,
            audience_type=request.headers.get(HEADER_AUTH_AUDIENCE_TYPE) or None,
            audience_id=request.headers.get(HEADER_AUTH_AUDIENCE_ID) or None,
            max_access_level=max_access_level,
            workspace_scope=workspace_scope,
        )

        return actor, authority


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class AetherAuthenticationServicePlugin(AuthenticationServicePluginBase):
    """Enterprise plugin that enables Aether gateway authentication.

    Activated when ``MEMORYLAYER_AUTHENTICATION_SERVICE=aether``.
    """

    PROVIDER_NAME = "aether"

    def initialize(self, v: Variables, logger: logging.Logger) -> AetherAuthenticationService:
        session_service: SessionService = self.get_extension(EXT_SESSION_SERVICE, v=v)
        workspace_service: WorkspaceService = self.get_extension(EXT_WORKSPACE_SERVICE, v=v)

        implicit_create = v.environ(
            MEMORYLAYER_SESSION_IMPLICIT_CREATE,
            default=DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE,
            type_fn=ext_parse_bool,
        )

        return AetherAuthenticationService(
            session_service=session_service,
            workspace_service=workspace_service,
            implicit_session_create=implicit_create,
            logger=logger,
        )

    def get_dependencies(self, v: Variables) -> Iterable[str]:
        return (EXT_SESSION_SERVICE, EXT_WORKSPACE_SERVICE)


__all__ = [
    "AetherAuthenticationService",
    "AetherAuthenticationServicePlugin",
    "HEADER_AUTH_TENANT_ID",
    "HEADER_AUTH_USER_ID",
    "HEADER_AUTH_API_KEY_ID",
    "HEADER_AUTH_WORKSPACE_ACCESS",
    "HEADER_AUTH_PRINCIPAL_TYPE",
    "HEADER_AUTH_SCOPES",
    "HEADER_AUTH_ACTOR_TYPE",
    "HEADER_AUTH_ACTOR_ID",
    "HEADER_AUTH_AUTHORITY_MODE",
    "HEADER_AUTH_GRANT_ID",
    "HEADER_AUTH_SUBJECT_TYPE",
    "HEADER_AUTH_SUBJECT_ID",
    "HEADER_AUTH_ROOT_SUBJECT_TYPE",
    "HEADER_AUTH_ROOT_SUBJECT_ID",
    "HEADER_AUTH_AUDIENCE_TYPE",
    "HEADER_AUTH_AUDIENCE_ID",
    "HEADER_AUTH_MAX_ACCESS_LEVEL",
    "HEADER_AUTH_WORKSPACE_SCOPE",
    "META_ACCESS_LEVEL",
    "META_PRINCIPAL_TYPE",
    "META_SCOPES",
    "META_GRANT_ID",
    "META_GRANT_MAX_ACCESS_LEVEL",
    "META_AUTHORITY_MODE",
    "DEFAULT_ACCESS_LEVEL",
]
