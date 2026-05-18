"""
Aether Authorization Service for MemoryLayer.

Maps numeric access levels from the Aether gateway (injected via
``X-Auth-Workspace-Access`` header) to MemoryLayer resource/action
permissions.

Access level hierarchy (matches aether3-go ``internal/acl/types.go``):

    ======  =====  =================================================
    Level   Name   Capabilities
    ======  =====  =================================================
     0      NONE   No access
    10      READ   recall, search, get/list memories/sessions/workspaces
    20      READWRITE  + remember, create_session, update_memory, create_workspace
    30      MANAGE + delete_memory, delete_workspace, manage_sessions, document_upload
    40      ADMIN  + tenant admin, bulk operations, data export
    50      SUPERADMIN  Reserved for system-level operations
    ======  =====  =================================================
"""

import logging

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from memorylayer_server.models.authz import AuthorizationContext, AuthorizationDecision
from memorylayer_server.services.authentication.aether import (
    META_ACCESS_LEVEL,
    META_GRANT_MAX_ACCESS_LEVEL,
)
from memorylayer_server.services.authorization.base import (
    AuthorizationService,
    AuthorizationServicePluginBase,
)

# ---------------------------------------------------------------------------
# Access level constants (mirror aether3-go internal/acl/types.go)
# ---------------------------------------------------------------------------
ACCESS_NONE = 0
ACCESS_READ = 10
ACCESS_READWRITE = 20
ACCESS_MANAGE = 30
ACCESS_ADMIN = 40
ACCESS_SUPERADMIN = 50

# ---------------------------------------------------------------------------
# Permission map: (resource, action) -> minimum access level required
# ---------------------------------------------------------------------------
_PERMISSION_MAP: dict[tuple[str, str], int] = {
    # Memories
    ("memories", "read"): ACCESS_READ,
    ("memories", "write"): ACCESS_READWRITE,
    ("memories", "delete"): ACCESS_MANAGE,
    # Sessions
    ("sessions", "read"): ACCESS_READ,
    ("sessions", "write"): ACCESS_READWRITE,
    ("sessions", "delete"): ACCESS_MANAGE,
    # Workspaces
    ("workspaces", "read"): ACCESS_READ,
    ("workspaces", "write"): ACCESS_READWRITE,
    ("workspaces", "delete"): ACCESS_MANAGE,
    # Documents
    ("documents", "read"): ACCESS_READ,
    ("documents", "write"): ACCESS_MANAGE,
    ("documents", "delete"): ACCESS_MANAGE,
    # Chat / threads
    ("threads", "read"): ACCESS_READ,
    ("threads", "write"): ACCESS_READWRITE,
    ("threads", "delete"): ACCESS_MANAGE,
    # Entities
    ("entities", "read"): ACCESS_READ,
    ("entities", "write"): ACCESS_READWRITE,
    # Datasets
    ("datasets", "read"): ACCESS_READ,
    ("datasets", "write"): ACCESS_READWRITE,
    ("datasets", "delete"): ACCESS_MANAGE,
    # Context environment
    ("context", "read"): ACCESS_READ,
    ("context", "write"): ACCESS_READWRITE,
    # Admin catch-all
    ("admin", "read"): ACCESS_ADMIN,
    ("admin", "write"): ACCESS_ADMIN,
    ("admin", "delete"): ACCESS_ADMIN,
}

# Default level required when a (resource, action) pair is not explicitly
# mapped.  MANAGE is a safe conservative default.
_DEFAULT_REQUIRED_LEVEL = ACCESS_MANAGE


def _access_level_name(level: int) -> str:
    """Return a human-readable name for an access level."""
    names = {
        ACCESS_NONE: "NONE",
        ACCESS_READ: "READ",
        ACCESS_READWRITE: "READWRITE",
        ACCESS_MANAGE: "MANAGE",
        ACCESS_ADMIN: "ADMIN",
        ACCESS_SUPERADMIN: "SUPERADMIN",
    }
    return names.get(level, "UNKNOWN")


def _role_from_access_level(level: int) -> str | None:
    """Map a numeric access level to a human-readable role name."""
    if level >= ACCESS_ADMIN:
        return "admin"
    if level >= ACCESS_MANAGE:
        return "manager"
    if level >= ACCESS_READWRITE:
        return "developer"
    if level >= ACCESS_READ:
        return "reader"
    return None


class AetherAuthorizationService(AuthorizationService):
    """Authorization service backed by Aether gateway access levels.

    The numeric access level stored in ``RequestContext.metadata``
    (key :data:`META_ACCESS_LEVEL`) is compared against the minimum
    level required for the requested resource/action pair.

    When the access level metadata is absent (e.g. running without the
    gateway), the service falls back to **deny** for safety.  Use the
    open-permissions plugin for local development without the gateway.
    """

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized AetherAuthorizationService")

    async def authorize(self, context: AuthorizationContext) -> AuthorizationDecision:
        """Evaluate authorization using the Aether access level.

        Reads the access level from ``context.metadata[META_ACCESS_LEVEL]``
        and compares it to the minimum level required for
        ``(context.resource, context.action)``.
        """
        granted_level = context.metadata.get(META_ACCESS_LEVEL, ACCESS_NONE)

        # Apply grant ceiling: effective_access = min(subject_level, grant.max_access_level)
        # Only applied when OBO is active and a ceiling is present in metadata.
        grant_ceiling = context.metadata.get(META_GRANT_MAX_ACCESS_LEVEL)
        if grant_ceiling is not None:
            granted_level = min(granted_level, grant_ceiling)

        # Handle wildcard admin actions
        if context.action == "*":
            required = ACCESS_ADMIN
        else:
            required = _PERMISSION_MAP.get(
                (context.resource, context.action),
                _DEFAULT_REQUIRED_LEVEL,
            )

        if granted_level >= required:
            self.logger.debug(
                "ALLOW: resource=%s action=%s required=%s(%d) granted=%s(%d) tenant=%s workspace=%s user=%s",
                context.resource,
                context.action,
                _access_level_name(required),
                required,
                _access_level_name(granted_level),
                granted_level,
                context.tenant_id,
                context.workspace_id,
                context.user_id,
            )
            return AuthorizationDecision.ALLOW

        self.logger.warning(
            "DENY: resource=%s action=%s required=%s(%d) granted=%s(%d) tenant=%s workspace=%s user=%s",
            context.resource,
            context.action,
            _access_level_name(required),
            required,
            _access_level_name(granted_level),
            granted_level,
            context.tenant_id,
            context.workspace_id,
            context.user_id,
        )
        return AuthorizationDecision.DENY

    async def get_allowed_workspaces(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[str]:
        """Return wildcard -- workspace scoping is enforced at the gateway.

        The Aether auth-proxy already scopes the request to the correct
        workspace via ``X-Workspace-ID``.  All workspaces the user can
        reach have already been validated.
        """
        return ["*"]

    async def get_user_role(
        self,
        tenant_id: str,
        workspace_id: str,
        user_id: str,
    ) -> str | None:
        """Derive a role name from the access level.

        Since the actual access level is per-request (on the context
        metadata), this method cannot give a definitive answer without
        the request context.  It returns ``None`` to signal that role
        information is request-scoped and should be derived from the
        authorization context metadata.
        """
        return None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class AetherAuthorizationServicePlugin(AuthorizationServicePluginBase):
    """Enterprise plugin that enables Aether gateway authorization.

    Activated when ``MEMORYLAYER_AUTHORIZATION_SERVICE=aether``.
    """

    PROVIDER_NAME = "aether"

    def initialize(self, v: Variables, logger: logging.Logger) -> AetherAuthorizationService:
        return AetherAuthorizationService(v=v)


# ---------------------------------------------------------------------------
# Public helpers for use by other enterprise services
# ---------------------------------------------------------------------------


def get_required_access_level(resource: str, action: str) -> int:
    """Look up the minimum access level for a resource/action pair.

    Returns the mapped level or :data:`_DEFAULT_REQUIRED_LEVEL` if the
    pair is not explicitly configured.
    """
    if action == "*":
        return ACCESS_ADMIN
    return _PERMISSION_MAP.get((resource, action), _DEFAULT_REQUIRED_LEVEL)


__all__ = [
    "AetherAuthorizationService",
    "AetherAuthorizationServicePlugin",
    "ACCESS_NONE",
    "ACCESS_READ",
    "ACCESS_READWRITE",
    "ACCESS_MANAGE",
    "ACCESS_ADMIN",
    "ACCESS_SUPERADMIN",
    "get_required_access_level",
]
