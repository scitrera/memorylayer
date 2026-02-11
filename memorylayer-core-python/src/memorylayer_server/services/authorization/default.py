"""Open permissions authorization - allows everything (OSS default)."""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import (
    AuthorizationService,
    AuthorizationServicePluginBase
)
from ...models.authz import AuthorizationDecision, AuthorizationContext


class OpenPermissionsAuthorizationService(AuthorizationService):
    """Default authorization that allows all operations.

    This is the default - no restrictions.
    Custom implementations can provide RBAC, tenant isolation, etc.
    """

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized OpenPermissionsAuthorizationService (allow-all mode)")

    async def authorize(self, context: AuthorizationContext) -> AuthorizationDecision:
        """Always allow - OSS default."""
        self.logger.debug(
            "Authorization check (allow-all): resource=%s action=%s workspace=%s",
            context.resource, context.action, context.workspace_id
        )
        return AuthorizationDecision.ALLOW

    async def get_allowed_workspaces(
        self,
        tenant_id: str,
        user_id: str
    ) -> list[str]:
        """Return wildcard - all workspaces allowed."""
        return ["*"]

    async def get_user_role(
        self,
        tenant_id: str,
        workspace_id: str,
        user_id: str
    ) -> Optional[str]:
        """Return admin role - full access in OSS mode."""
        return "admin"


class OpenPermissionsAuthorizationPlugin(AuthorizationServicePluginBase):
    """Plugin for open permissions authorization."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> AuthorizationService:
        return OpenPermissionsAuthorizationService(v=v)
