"""Authorization Service - Pluggable permission checking interface."""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from fastapi import HTTPException, status
from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_AUTHORIZATION_SERVICE, DEFAULT_MEMORYLAYER_AUTHORIZATION_SERVICE
from ...models.authz import AuthorizationDecision, AuthorizationContext

if TYPE_CHECKING:
    from ...models.auth import RequestContext

from .._constants import EXT_AUTHORIZATION_SERVICE


class AuthorizationService(ABC):
    """Abstract authorization service interface.

    This is the pluggable authorization layer for MemoryLayer.

    The default (OpenPermissionsAuthorizationService) allows all operations.
    Custom implementations can provide RBAC, tenant isolation, etc.
    """

    async def require_authorization(
        self,
        ctx: 'RequestContext',
        resource: str,
        action: str,
        resource_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Check authorization and raise HTTPException(403) if denied.

        This is a convenience method that combines common authorization patterns:
        - Builds AuthorizationContext from provided params
        - Calls authorize() to check permission
        - Raises HTTPException with 403 if denied

        Tenant isolation is enforced at the storage backend level, so explicit
        tenant_id verification is not needed here.

        Args:
            ctx: Request context from AuthenticationService.build_context()
            resource: Resource type (e.g., 'memories', 'workspaces', 'sessions')
            action: Action type (e.g., 'read', 'write', 'create', 'delete')
            resource_id: Optional specific resource ID
            workspace_id: Optional workspace ID (uses ctx.workspace_id if not provided)

        Raises:
            HTTPException: 403 Forbidden if authorization denied
        """
        authz_ctx = AuthorizationContext(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id or ctx.workspace_id,
            user_id=ctx.user_id,
            resource=resource,
            action=action,
            resource_id=resource_id,
        )

        decision = await self.authorize(authz_ctx)
        if decision == AuthorizationDecision.DENY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {resource}"
            )

    @abstractmethod
    async def authorize(self, context: AuthorizationContext) -> AuthorizationDecision:
        """Check if the operation is authorized.

        Args:
            context: Authorization context with tenant/workspace/user/resource/action

        Returns:
            AuthorizationDecision.ALLOW, DENY, or ABSTAIN
        """
        pass

    @abstractmethod
    async def get_allowed_workspaces(
            self,
            tenant_id: str,
            user_id: str
    ) -> list[str]:
        """Get list of workspace IDs user can access.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            List of workspace IDs (empty = no access, ['*'] = all workspaces)
        """
        pass

    @abstractmethod
    async def get_user_role(
            self,
            tenant_id: str,
            workspace_id: str,
            user_id: str
    ) -> Optional[str]:
        """Get user's role in a workspace.

        Args:
            tenant_id: Tenant identifier
            workspace_id: Workspace identifier
            user_id: User identifier

        Returns:
            Role string (admin, developer, reader) or None if no access
        """
        pass


# noinspection PyAbstractClass
class AuthorizationServicePluginBase(Plugin):
    """Base plugin for authorization service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_AUTHORIZATION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_AUTHORIZATION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_AUTHORIZATION_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_AUTHORIZATION_SERVICE, DEFAULT_MEMORYLAYER_AUTHORIZATION_SERVICE)
