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
    #
    # write/delete are READWRITE (not MANAGE): a principal with write access to
    # a workspace can manage that workspace's documents — create, update, and
    # remove them. This matches the convention used by every other content
    # resource here (memories/sessions/threads write=READWRITE) and, crucially,
    # the platform files-library flow: the platform-bridge front-door already
    # verifies the acting user's can_write_to_workspace before issuing the
    # delete, so requiring MANAGE at this layer only rejected legitimate owners
    # (a workspace admin deleting their own private-workspace files). MANAGE was
    # also an outlier for documents:write specifically (all other write=RW).
    ("documents", "read"): ACCESS_READ,
    ("documents", "write"): ACCESS_READWRITE,
    ("documents", "delete"): ACCESS_READWRITE,
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
    # Skills — read/write/delete resolve via the action defaults below; "mcp"
    # is a non-standard action (read/use MCP-backed skills), so map it explicitly.
    ("skills", "mcp"): ACCESS_READ,
    # Create actions for user-content resources are write-level (READWRITE),
    # matching each resource's "write" entry. The API enforces action="create"
    # (NOT "write") for these, and without an explicit entry "create" falls
    # through to the MANAGE default (_DEFAULT_REQUIRED_LEVEL) — wrongly
    # rejecting an ordinary RW writer. Observed: the platform-bridge committing
    # a chat memory into the user's own private workspace →
    # "memories:create DENY required=MANAGE(30) granted=READWRITE(20)".
    ("memories", "create"): ACCESS_READWRITE,
    ("sessions", "create"): ACCESS_READWRITE,
    ("documents", "create"): ACCESS_READWRITE,
    ("workspaces", "create"): ACCESS_READWRITE,
    # Admin catch-all
    ("admin", "read"): ACCESS_ADMIN,
    ("admin", "write"): ACCESS_ADMIN,
    ("admin", "delete"): ACCESS_ADMIN,
}

# Default level required for an unknown action (one not covered by the
# action-aware defaults below). MANAGE is a safe conservative fallback.
_DEFAULT_REQUIRED_LEVEL = ACCESS_MANAGE

# Action-aware fallback for (resource, action) pairs not enumerated in
# _PERMISSION_MAP. Without this, every un-enumerated resource (e.g. skills,
# applications, collections, knowledgebase, ...) fell through to MANAGE — so an
# ordinary READ wrongly required MANAGE(30).
#
# Deliberately only READ-class actions are relaxed here: a read should require
# READ. Writes/creates/deletes of un-enumerated resources stay at the
# conservative MANAGE default — many are enterprise *management* resources whose
# write level is a policy decision; add an explicit _PERMISSION_MAP entry to
# grant write at READWRITE where intended. Explicit entries always win.
_ACTION_DEFAULT_LEVEL: dict[str, int] = {
    "read": ACCESS_READ,
    "list": ACCESS_READ,
}


def _required_level(resource: str, action: str) -> int:
    """Minimum access level for a (resource, action).

    Precedence: wildcard ``*`` => ADMIN; explicit ``_PERMISSION_MAP`` entry;
    else the action-aware default; else (unknown action) MANAGE.
    """
    if action == "*":
        return ACCESS_ADMIN
    explicit = _PERMISSION_MAP.get((resource, action))
    if explicit is not None:
        return explicit
    return _ACTION_DEFAULT_LEVEL.get(action, _DEFAULT_REQUIRED_LEVEL)


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

        required = _required_level(context.resource, context.action)

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

    Returns the explicit mapped level, else the action-aware default (e.g.
    ``read`` => READ, ``write`` => READWRITE, ``delete`` => MANAGE), else
    MANAGE for an unknown action.
    """
    return _required_level(resource, action)


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
