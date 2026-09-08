"""
Authentication service interface.

The AuthenticationService handles:
1. API key verification (noop in OSS)
2. Session resolution from X-Session-ID header
3. Workspace resolution with priority order
4. Building the RequestContext for API endpoints

Enterprise implementations can extend this for:
- API key generation and verification
- JWT token validation
- RBAC integration
- Gateway-injected identity headers (e.g. Aether auth-proxy)
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel

from ...config import (
    DEFAULT_MEMORYLAYER_AUTHENTICATION_SERVICE,
    MEMORYLAYER_AUTHENTICATION_SERVICE,
)
from ...models.auth import AuthIdentity, RequestContext

if TYPE_CHECKING:
    from ...models.session import Session

from .._constants import EXT_AUTHENTICATION_SERVICE
from .._plugin_factory import make_service_plugin_base

# Header names
HEADER_AUTHORIZATION = "Authorization"
HEADER_SESSION_ID = "X-Session-ID"

# HTTP methods that must not have side effects. Auth resolves a workspace on
# EVERY request, so auto-creating during resolution made a plain read create
# rows: the admin console's cross-workspace list endpoints (GET
# /v1/admin/{skills,memories,jobs,documents,...}) take ``workspace_id`` as a
# *filter*, and typing "jgl-field" into that filter box left eight workspaces
# behind -- one per debounced keystroke (j, jg, jgl, jgl-, ...). The same
# request shape reaches every one of those endpoints, so the fix belongs here
# rather than in any single caller.
SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def request_may_create(request: Request) -> bool:
    """Whether this request is allowed to materialise resources as a side effect."""
    return request.method.upper() not in SAFE_HTTP_METHODS


async def ensure_resolved_workspace(
    workspace_service,
    workspace_id: str,
    tenant_id: str,
    *,
    may_create: bool,
    implicit_create: bool,
    explicitly_named: bool,
) -> None:
    """Apply the create-during-auth rules. Shared by both authenticators.

    Three cases, in order:

    * **Safe method** (``may_create`` False) — do nothing at all, not even a
      lookup. A read against a workspace that does not exist finds nothing,
      which is the correct answer; the admin console's "Filter workspace" box
      depends on this staying quiet rather than erroring.
    * **Implicit create enabled** (OSS) — materialise it. A malformed id
      surfaces as HTTP 400 rather than the service's ``ValueError`` so the
      broken caller sees which value was rejected.
    * **Implicit create disabled** (enterprise) — a write that NAMES a
      workspace must name one that exists, so a missing one is 404 here rather
      than an opaque foreign-key 500 from whatever the handler writes first.
      Only checked when the caller named it: falling back to ``_default`` or to
      the session's workspace is not a request to use a specific workspace, and
      failing those would deadlock bootstrap (``POST /v1/workspaces`` itself
      resolves ``_default`` on its way to creating the first workspace).
    """
    if not may_create:
        return

    if implicit_create:
        try:
            await workspace_service.ensure_workspace(
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                auto_create=True,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return

    if not explicitly_named:
        return

    if await workspace_service.ensure_workspace(
        workspace_id=workspace_id,
        tenant_id=tenant_id,
        auto_create=False,
    ):
        return

    raise HTTPException(
        status_code=404,
        detail=f"workspace not found: {workspace_id}",
    )


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationService(ABC):
    """
    Abstract base class for authentication services.

    Responsible for:
    - Verifying API credentials
    - Resolving session from request headers
    - Building RequestContext with resolved workspace
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def verify_api_key(self, api_key: str | None) -> AuthIdentity:
        """
        Verify API key and return identity.

        Args:
            api_key: The API key from Authorization header (may be None)

        Returns:
            AuthIdentity with tenant_id and optional user_id

        Raises:
            AuthenticationError: If API key is invalid (not in OSS)
        """
        pass

    @abstractmethod
    async def resolve_session(self, session_id: str | None) -> Optional["Session"]:
        """
        Resolve session from session ID.

        Args:
            session_id: Session ID from X-Session-ID header

        Returns:
            Session if found and valid, None otherwise
        """
        pass

    @abstractmethod
    async def resolve_workspace(
        self,
        request_workspace_id: str | None,
        session: Optional["Session"],
        tenant_id: str,
        allow_create: bool = True,
    ) -> str:
        """
        Resolve effective workspace_id using priority order.

        Priority:
        1. request_workspace_id (explicit override in body)
        2. session.workspace_id (from session)
        3. DEFAULT_WORKSPACE_ID ("_default")

        Ensures the workspace exists (auto-creating in OSS) only when
        ``allow_create`` is set. Resolution always yields an id either way —
        a read of a workspace that does not exist simply finds nothing.

        Args:
            request_workspace_id: Explicit workspace from request body
            session: Resolved session (may be None)
            tenant_id: Tenant ID for auto-creation
            allow_create: Whether this request may create the workspace as a
                side effect. Callers pass False for safe HTTP methods.

        Returns:
            Resolved workspace_id
        """
        pass

    async def ensure_session(
        self,
        session_id: str,
        workspace_id: str,
        tenant_id: str,
    ) -> Optional["Session"]:
        """
        Ensure a session exists, creating it if necessary.

        Called when a request provides X-Session-ID but the session
        doesn't exist (expired or never created). Only called when
        the client explicitly provides a workspace context.

        Default implementation: no implicit creation (returns None).
        Override in subclass to enable.

        Args:
            session_id: Session ID from request
            workspace_id: Resolved workspace ID
            tenant_id: Authenticated tenant ID

        Returns:
            Newly created session, or None if implicit creation disabled
        """
        return None

    async def build_context(
        self,
        request: Request,
        body: BaseModel | None = None,
    ) -> RequestContext:
        """
        Build full RequestContext from request headers and body.

        This is the main entry point for endpoints.

        Args:
            request: FastAPI Request object (for headers)
            body: Parsed request body (for workspace_id override)

        Returns:
            RequestContext with resolved tenant, workspace, session

        Raises:
            AuthenticationError: If authentication fails
        """
        # 1. Extract API key from Authorization header
        api_key = self._extract_api_key(request)

        # 2. Verify API key and get identity
        identity = await self.verify_api_key(api_key)

        # 3. Extract and resolve session
        session_id = request.headers.get(HEADER_SESSION_ID)
        session = await self.resolve_session(session_id) if session_id else None

        # 4. Extract workspace_id from (in order): request body, query string,
        # ``X-Workspace-ID`` header. Many endpoints accept ``workspace_id`` as a
        # FastAPI ``Query`` parameter (e.g. /v1/threads/{id}/messages,
        # /v1/memories) and then use it in preference to the context's
        # workspace. Without the query fallback, resolve_workspace's auto-create
        # ensures a DIFFERENT workspace than the handler goes on to write to.
        #
        # That is not cosmetic: ``chat_threads.workspace_id`` is a foreign key
        # onto ``workspaces(id)``, so appending to a thread in a workspace that
        # was never created fails with a bare sqlite IntegrityError, surfaced as
        # an opaque 500 "Failed to append messages" naming nothing about the
        # workspace. Routes whose body model has no ``workspace_id`` field at
        # all — MessagesAppendRequest is one — hit this on EVERY call, so a
        # client using any workspace other than "_default" cannot append.
        #
        # The Aether authenticator has carried this fallback for a while; this
        # is the same fix on the default (OSS) path, in the same order.
        request_workspace_id = getattr(body, "workspace_id", None) if body else None
        if not request_workspace_id:
            request_workspace_id = request.query_params.get("workspace_id")
        if not request_workspace_id:
            request_workspace_id = request.headers.get("X-Workspace-ID")

        # 5. Resolve effective workspace
        may_create = request_may_create(request)
        workspace_id = await self.resolve_workspace(
            request_workspace_id=request_workspace_id,
            session=session,
            tenant_id=identity.tenant_id,
            allow_create=may_create,
        )

        # Implicit session creation: if session_id was provided but session
        # not found, and client explicitly provided a workspace, auto-create.
        # Held to the same rule as the workspace above -- a session row carries
        # a workspace foreign key, so creating one on a GET both violates the
        # method's contract and can fail against a workspace that (correctly)
        # was not created either.
        if may_create and session_id and session is None and request_workspace_id:
            session = await self.ensure_session(session_id, workspace_id, identity.tenant_id)

        self.logger.debug(
            "Built context: tenant=%s, workspace=%s, session=%s",
            identity.tenant_id,
            workspace_id,
            session.id if session else None,
        )

        return RequestContext(
            tenant_id=identity.tenant_id,
            workspace_id=workspace_id,
            user_id=identity.user_id,
            session=session,
        )

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from Authorization header."""
        auth_header = request.headers.get(HEADER_AUTHORIZATION)
        if not auth_header:
            return None

        # Support "Bearer <token>" format
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Also support raw token
        return auth_header


# noinspection PyAbstractClass
AuthenticationServicePluginBase = make_service_plugin_base(
    ext_name=EXT_AUTHENTICATION_SERVICE,
    config_key=MEMORYLAYER_AUTHENTICATION_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_AUTHENTICATION_SERVICE,
)
