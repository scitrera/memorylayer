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
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from fastapi import Request
from pydantic import BaseModel

from ...models.auth import AuthIdentity, RequestContext

if TYPE_CHECKING:
    from ...models.session import Session

from .._constants import EXT_AUTHENTICATION_SERVICE

# Header names
HEADER_AUTHORIZATION = "Authorization"
HEADER_SESSION_ID = "X-Session-ID"


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

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def verify_api_key(self, api_key: Optional[str]) -> AuthIdentity:
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
    async def resolve_session(self, session_id: Optional[str]) -> Optional["Session"]:
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
        request_workspace_id: Optional[str],
        session: Optional["Session"],
        tenant_id: str,
    ) -> str:
        """
        Resolve effective workspace_id using priority order.

        Priority:
        1. request_workspace_id (explicit override in body)
        2. session.workspace_id (from session)
        3. DEFAULT_WORKSPACE_ID ("_default")

        Also ensures workspace exists (auto-creates in OSS).

        Args:
            request_workspace_id: Explicit workspace from request body
            session: Resolved session (may be None)
            tenant_id: Tenant ID for auto-creation

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
        body: Optional[BaseModel] = None,
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

        # 4. Extract workspace_id from body or X-Workspace-ID header
        request_workspace_id = getattr(body, "workspace_id", None) if body else None
        if not request_workspace_id:
            request_workspace_id = request.headers.get("X-Workspace-ID")

        # 5. Resolve effective workspace
        workspace_id = await self.resolve_workspace(
            request_workspace_id=request_workspace_id,
            session=session,
            tenant_id=identity.tenant_id,
        )

        # Implicit session creation: if session_id was provided but session
        # not found, and client explicitly provided a workspace, auto-create
        if session_id and session is None and request_workspace_id:
            session = await self.ensure_session(
                session_id, workspace_id, identity.tenant_id
            )

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

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from Authorization header."""
        auth_header = request.headers.get(HEADER_AUTHORIZATION)
        if not auth_header:
            return None

        # Support "Bearer <token>" format
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Also support raw token
        return auth_header
