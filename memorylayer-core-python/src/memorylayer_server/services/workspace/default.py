"""Default workspace service implementation."""

import re
from datetime import UTC, datetime
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...config import DEFAULT_CONTEXT_ID, DEFAULT_TENANT_ID
from ...models import Workspace
from ...models.workspace import Context
from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from .base import WorkspaceServicePluginBase

# Ids that auto-creation is willing to materialise. Leading underscore is
# allowed for the reserved workspaces (_default, _global, _global_user).
#
# This gates CREATION ONLY -- an existing workspace resolves no matter how it
# is spelled, so tightening the rule never strands data already in the table.
# The rule exists because auto-creation turns a caller's typo into a permanent
# row: production picked up `/workspace` and `/sahara` (a path passed where an
# id was expected) and `workspace:_global` (a caller that prefixed the id, and
# whose real target `_global` was created 22ms later). None of those are
# identifiers anyone chose; all three would have been rejected here.
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")


def is_creatable_workspace_id(workspace_id: str) -> bool:
    """Whether ``workspace_id`` is well-formed enough to be auto-created."""
    return bool(workspace_id) and _WORKSPACE_ID_RE.match(workspace_id) is not None


class WorkspaceService:
    """
    Core workspace service implementing workspace operations.

    This service coordinates workspace management with storage backend integration.
    """

    def __init__(self, storage: StorageBackend, v: Variables = None):
        """
        Initialize workspace service.

        Args:
            storage: Storage backend for workspace persistence
            v: Variables for logging context
        """
        self._storage = storage
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized WorkspaceService")

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """
        Create a new workspace.

        Args:
            workspace: Workspace object to create

        Returns:
            Created workspace with generated fields

        Raises:
            ValueError: If workspace validation fails
        """
        self.logger.info("Creating workspace: %s for tenant: %s", workspace.name, workspace.tenant_id)

        # Create workspace via storage backend
        created = await self._storage.create_workspace(workspace)

        self.logger.info("Created workspace: %s", created.id)
        return created

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """
        Get workspace by ID.

        Args:
            workspace_id: Unique workspace identifier

        Returns:
            Workspace if found, None otherwise
        """
        self.logger.debug("Getting workspace: %s", workspace_id)
        return await self._storage.get_workspace(workspace_id)

    async def list_workspaces(
        self,
        *,
        tags: list[str] | None = None,
        match: str = "all",
    ) -> list["Workspace"]:
        """
        List workspaces, optionally filtered by tag.

        Args:
            tags: Optional tag filter. When provided, only workspaces carrying the tag(s)
                are returned.
            match: 'all' requires every tag; 'any' requires at least one.

        Returns:
            List of matching workspaces
        """
        self.logger.debug("Listing workspaces (tags=%s, match=%s)", tags, match)
        return await self._storage.list_workspaces(tags=tags, match=match)

    async def ensure_workspace(
        self,
        workspace_id: str,
        tenant_id: str = None,
        auto_create: bool = True,
    ) -> Workspace | None:
        """
        Ensure a workspace exists, optionally creating it if missing.

        This supports the "just works" API pattern where workspaces are
        auto-created on first access (e.g., MCP auto-discovery from git repo name).

        Args:
            workspace_id: Workspace ID to ensure exists
            tenant_id: Tenant ID for new workspace (defaults to _default)
            auto_create: If True, create workspace if it doesn't exist

        Returns:
            Workspace if found or created, None if not found and auto_create=False

        Raises:
            ValueError: If the workspace is missing and its id is too malformed
                to create (see :func:`is_creatable_workspace_id`).
        """
        self.logger.debug("Ensuring workspace exists: %s", workspace_id)

        # Check if workspace exists
        workspace = await self._storage.get_workspace(workspace_id)
        if workspace:
            return workspace

        if not auto_create:
            self.logger.debug("Workspace not found and auto_create=False: %s", workspace_id)
            return None

        # Refuse to mint a row for something that isn't an identifier. Raising
        # (rather than quietly returning None) is deliberate: the caller is
        # broken, and a silent no-op here would surface downstream as an opaque
        # foreign-key 500 naming nothing about the workspace.
        if not is_creatable_workspace_id(workspace_id):
            raise ValueError(f"cannot auto-create workspace with malformed id: {workspace_id!r}")

        # Auto-create workspace
        self.logger.info("Auto-creating workspace: %s", workspace_id)
        tenant_id = tenant_id or DEFAULT_TENANT_ID
        now = datetime.now(UTC)

        workspace = Workspace(
            id=workspace_id,
            tenant_id=tenant_id,
            name=workspace_id,  # Use ID as name for auto-created workspaces
            settings={},
            created_at=now,
            updated_at=now,
        )

        created = await self._storage.create_workspace(workspace)
        self.logger.info("Auto-created workspace: %s for tenant: %s", created.id, tenant_id)
        return created

    async def ensure_default_context(self, workspace_id: str) -> None:
        """Ensure the _default context exists for a workspace.

        Creates it via storage if missing.  This is a lightweight
        bootstrapping step — no separate ContextService required.
        """
        if hasattr(self._storage, "get_context"):
            existing = await self._storage.get_context(workspace_id, f"{workspace_id}:{DEFAULT_CONTEXT_ID}")
            if existing:
                return
        if hasattr(self._storage, "create_context"):
            default_context = Context(
                id=f"{workspace_id}:{DEFAULT_CONTEXT_ID}",
                workspace_id=workspace_id,
                name=DEFAULT_CONTEXT_ID,
                description="Default context for the workspace",
                settings={},
            )
            try:
                await self._storage.create_context(workspace_id, default_context)
                self.logger.debug("Created _default context for workspace: %s", workspace_id)
            except Exception:
                # Already exists (race condition) or storage doesn't support contexts
                pass

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace.

        Args:
            workspace_id: Workspace ID to delete

        Returns:
            True if deleted, False if not found
        """
        existing = await self._storage.get_workspace(workspace_id)
        if not existing:
            return False

        deleted = await self._storage.delete_workspace(workspace_id)
        if not deleted:
            self.logger.warning(
                "Storage backend did not delete existing workspace: %s",
                workspace_id,
            )
            return False

        self.logger.info("Deleted workspace: %s", workspace_id)
        return True

    async def update_workspace(self, workspace: Workspace) -> Workspace:
        """
        Update workspace settings.

        Args:
            workspace: Workspace with updated fields

        Returns:
            Updated workspace

        Raises:
            ValueError: If workspace doesn't exist
        """
        self.logger.info("Updating workspace: %s", workspace.id)

        updated = await self._storage.update_workspace(
            workspace.id,
            name=workspace.name,
            settings=workspace.settings,
            tags=workspace.tags,
        )

        if not updated:
            raise ValueError(f"Workspace not found: {workspace.id}")

        self.logger.info("Updated workspace: %s", workspace.id)
        return updated


class DefaultWorkspaceServicePlugin(WorkspaceServicePluginBase):
    """Default workspace service plugin."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> WorkspaceService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return WorkspaceService(storage=storage, v=v)
