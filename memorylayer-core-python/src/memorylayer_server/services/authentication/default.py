"""
Default (OSS) authentication service implementation.

This implementation provides:
- Open permissions (no API key verification)
- Session resolution via session service
- Workspace auto-creation on first access
"""

import logging
from collections.abc import Iterable

from scitrera_app_framework import Variables, ext_parse_bool, get_extension

from ...config import (
    DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE,
    DEFAULT_TENANT_ID,
    DEFAULT_WORKSPACE_ID,
    MEMORYLAYER_SESSION_IMPLICIT_CREATE,
)
from ...models.auth import AuthIdentity
from ...models.session import Session
from ...services.session import EXT_SESSION_SERVICE, SessionService
from ...services.workspace import EXT_WORKSPACE_SERVICE, WorkspaceService
from .base import (
    EXT_AUTHENTICATION_SERVICE,
    AuthenticationService,
    AuthenticationServicePluginBase,
)


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
        implicit_session_create: bool = True,
        logger: logging.Logger | None = None,
    ):
        super().__init__(logger)
        self.session_service = session_service
        self.workspace_service = workspace_service
        self._implicit_session_create = implicit_session_create

    async def verify_api_key(self, api_key: str | None) -> AuthIdentity:
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

    async def resolve_session(self, session_id: str | None) -> Session | None:
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
        request_workspace_id: str | None,
        session: Session | None,
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
        workspace_id = request_workspace_id or (session.workspace_id if session else None) or DEFAULT_WORKSPACE_ID

        # Auto-create workspace if needed (OSS "just works" pattern)
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
        """
        Auto-create session for unknown session_id when workspace is explicit.

        Gated on MEMORYLAYER_SESSION_IMPLICIT_CREATE config flag.
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


class OpenAuthenticationServicePlugin(AuthenticationServicePluginBase):
    """Plugin to register the OSS authentication service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> OpenAuthenticationService:
        session_service = self.get_extension(EXT_SESSION_SERVICE, v=v)
        workspace_service = self.get_extension(EXT_WORKSPACE_SERVICE, v=v)

        implicit_create = v.environ(
            MEMORYLAYER_SESSION_IMPLICIT_CREATE,
            default=DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE,
            type_fn=ext_parse_bool,
        )

        return OpenAuthenticationService(
            session_service=session_service,
            workspace_service=workspace_service,
            implicit_session_create=implicit_create,
            logger=logger,
        )

    # noinspection PyMethodMayBeStatic
    def get_dependencies(self, v: Variables) -> Iterable[str]:
        return (
            EXT_SESSION_SERVICE,
            EXT_WORKSPACE_SERVICE,
        )


def get_authentication_service(v: Variables) -> AuthenticationService:
    """Get the authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)
