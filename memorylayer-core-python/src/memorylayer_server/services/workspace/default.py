"""Default workspace service implementation."""
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models import Workspace
from ...models.workspace import Context
from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from ...config import DEFAULT_TENANT_ID, DEFAULT_CONTEXT_ID
from .base import WorkspaceServicePluginBase


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
        self.logger.info(
            "Creating workspace: %s for tenant: %s",
            workspace.name,
            workspace.tenant_id
        )

        # Create workspace via storage backend
        created = await self._storage.create_workspace(workspace)

        self.logger.info("Created workspace: %s", created.id)
        return created

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get workspace by ID.

        Args:
            workspace_id: Unique workspace identifier

        Returns:
            Workspace if found, None otherwise
        """
        self.logger.debug("Getting workspace: %s", workspace_id)
        return await self._storage.get_workspace(workspace_id)

    async def ensure_workspace(
            self,
            workspace_id: str,
            tenant_id: str = None,
            auto_create: bool = True,
    ) -> Optional[Workspace]:
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
        """
        self.logger.debug("Ensuring workspace exists: %s", workspace_id)

        # Check if workspace exists
        workspace = await self._storage.get_workspace(workspace_id)
        if workspace:
            return workspace

        if not auto_create:
            self.logger.debug("Workspace not found and auto_create=False: %s", workspace_id)
            return None

        # Auto-create workspace
        self.logger.info("Auto-creating workspace: %s", workspace_id)
        tenant_id = tenant_id or DEFAULT_TENANT_ID
        now = datetime.now(timezone.utc)

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
        if hasattr(self._storage, 'get_context'):
            existing = await self._storage.get_context(workspace_id, f"{workspace_id}:{DEFAULT_CONTEXT_ID}")
            if existing:
                return
        if hasattr(self._storage, 'create_context'):
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

    async def update_workspace(self, workspace: Workspace) -> Workspace:
        """
        Update workspace settings.

        Note: Storage backend may not have update_workspace yet, so we store
        the updated workspace directly via create_workspace (upsert behavior).

        Args:
            workspace: Workspace with updated fields

        Returns:
            Updated workspace

        Raises:
            ValueError: If workspace doesn't exist
        """
        self.logger.info("Updating workspace: %s", workspace.id)

        # Check if workspace exists
        existing = await self._storage.get_workspace(workspace.id)
        if not existing:
            raise ValueError(f"Workspace not found: {workspace.id}")

        # Storage backend doesn't have update_workspace yet, so we use create
        # (assuming upsert behavior - in production, this would be update_workspace)
        updated = await self._storage.create_workspace(workspace)

        self.logger.info("Updated workspace: %s", workspace.id)
        return updated


class DefaultWorkspaceServicePlugin(WorkspaceServicePluginBase):
    """Default workspace service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> WorkspaceService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return WorkspaceService(
            storage=storage,
            v=v
        )
