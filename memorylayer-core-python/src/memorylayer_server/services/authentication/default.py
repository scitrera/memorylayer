"""
Default (OSS) authentication service implementation.

This implementation provides:
- Open permissions (no API key verification)
- Session resolution via session service
- Workspace auto-creation on first access
"""
import logging
from typing import Optional

from scitrera_app_framework import Plugin, Variables, get_extension

from .base import (
    AuthenticationService,
    EXT_AUTHENTICATION_SERVICE,
)
from ...models.auth import AuthIdentity
from ...models.session import Session
from ...config import DEFAULT_TENANT_ID, DEFAULT_WORKSPACE_ID
from ...services.session import SessionService, EXT_SESSION_SERVICE
from ...services.workspace import WorkspaceService, EXT_WORKSPACE_SERVICE


class OpenAuthenticationService(AuthenticationService):
    """
    OSS authentication service with open permissions.

    - API key verification: Always succeeds, returns default tenant
    - Session resolution: Looks up session via session service
    - Workspace resolution: Auto-creates workspaces as needed
    """

    def __init__(
        self,
        session_service: SessionService,
        workspace_service: WorkspaceService,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.session_service = session_service
        self.workspace_service = workspace_service

    async def verify_api_key(self, api_key: Optional[str]) -> AuthIdentity:
        """
        Verify API key - always succeeds in OSS.

        Returns default tenant identity regardless of API key.
        """
        # OSS: No verification, always return default tenant
        # Enterprise would validate the key and extract tenant/user
        return AuthIdentity(
            tenant_id=DEFAULT_TENANT_ID,
            user_id=None,
            api_key_id=None,
        )

    async def resolve_session(self, session_id: Optional[str]) -> Optional[Session]:
        """
        Resolve session from session service.

        Returns None if session not found or expired.
        """
        if not session_id:
            return None

        try:
            session = await self.session_service.get(session_id)
            return session
        except Exception as e:
            self.logger.debug("Session %s not found: %s", session_id, e)
            return None

    async def resolve_workspace(
        self,
        request_workspace_id: Optional[str],
        session: Optional[Session],
        tenant_id: str,
    ) -> str:
        """
        Resolve workspace with priority order and auto-creation.

        Priority:
        1. request_workspace_id (explicit override)
        2. session.workspace_id (from session)
        3. DEFAULT_WORKSPACE_ID ("_default")
        """
        # Priority resolution
        workspace_id = (
            request_workspace_id
            or (session.workspace_id if session else None)
            or DEFAULT_WORKSPACE_ID
        )

        # Auto-create workspace if needed (OSS "just works" pattern)
        await self.workspace_service.ensure_workspace(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            auto_create=True,
        )

        return workspace_id


class OpenAuthenticationServicePlugin(Plugin):
    """Plugin to register the OSS authentication service."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_AUTHENTICATION_SERVICE

    def initialize(self, v: Variables, logger: logging.Logger) -> OpenAuthenticationService:
        session_service = get_extension(EXT_SESSION_SERVICE, v)
        workspace_service = get_extension(EXT_WORKSPACE_SERVICE, v)

        return OpenAuthenticationService(
            session_service=session_service,
            workspace_service=workspace_service,
            logger=logger,
        )


def get_authentication_service(v: Variables) -> AuthenticationService:
    """Get the authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)
