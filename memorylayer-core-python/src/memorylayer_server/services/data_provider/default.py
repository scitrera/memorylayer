"""Default (local) Data Provider Service implementation."""

import logging
from datetime import UTC, datetime

from scitrera_app_framework import Variables, get_extension, get_logger

from ...models.data_provider import DataProvider, DataProviderType
from ...utils import generate_id
from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import DataProviderServicePluginBase
from .base import DataProviderService

_DEFAULT_PROVIDER_ID = "_default"
_DEFAULT_PROVIDER_NAME = "Default Local Provider"


class LocalDataProviderService(DataProviderService):
    """Data provider service backed by StorageBackend.

    Supports only LOCAL provider type in OSS. Automatically creates a
    ``_default`` local provider per workspace on first access if none exists.
    """

    def __init__(self, storage: StorageBackend, v: Variables):
        self._storage = storage
        self.logger = get_logger(v, name="DataProviderService")

    async def _ensure_default_provider(self, workspace_id: str) -> None:
        """Create the _default local provider for the workspace if it does not exist."""
        existing = await self._storage.get_data_provider(workspace_id, _DEFAULT_PROVIDER_ID)
        if existing:
            return

        now = datetime.now(UTC)
        default_provider = DataProvider(
            id=_DEFAULT_PROVIDER_ID,
            workspace_id=workspace_id,
            name=_DEFAULT_PROVIDER_NAME,
            provider_type=DataProviderType.LOCAL,
            description="Auto-created default local provider",
            enabled=True,
            connection_args={},
            metadata={},
            created_at=now,
            updated_at=now,
        )
        try:
            await self._storage.create_data_provider(workspace_id, default_provider)
            self.logger.info("Created default local provider for workspace %s", workspace_id)
        except Exception as e:
            self.logger.debug("Could not create default provider for workspace %s: %s", workspace_id, e)

    async def create_provider(self, workspace_id: str, provider: DataProvider) -> DataProvider:
        if not provider.id:
            provider = provider.model_copy(update={"id": generate_id("dp")})
        result = await self._storage.create_data_provider(workspace_id, provider)
        self.logger.info("Created data provider %s in workspace %s", result.id, workspace_id)
        return result

    async def get_provider(self, workspace_id: str, provider_id: str) -> DataProvider | None:
        await self._ensure_default_provider(workspace_id)
        return await self._storage.get_data_provider(workspace_id, provider_id)

    async def list_providers(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DataProvider], int]:
        await self._ensure_default_provider(workspace_id)
        return await self._storage.list_data_providers(workspace_id, limit=limit, offset=offset)

    async def update_provider(
        self,
        workspace_id: str,
        provider_id: str,
        **updates,
    ) -> DataProvider | None:
        result = await self._storage.update_data_provider(workspace_id, provider_id, **updates)
        if result:
            self.logger.info("Updated data provider %s in workspace %s", provider_id, workspace_id)
        return result

    async def delete_provider(self, workspace_id: str, provider_id: str) -> bool:
        result = await self._storage.delete_data_provider(workspace_id, provider_id)
        if result:
            self.logger.info("Deleted data provider %s from workspace %s", provider_id, workspace_id)
        return result

    async def sync(self, workspace_id: str, provider_id: str) -> list:
        """No-op sync for local providers. Returns empty list."""
        self.logger.debug("Sync requested for local provider %s in workspace %s (no-op)", provider_id, workspace_id)
        return []


class LocalDataProviderServicePlugin(DataProviderServicePluginBase):
    """Plugin for the local data provider service."""

    PROVIDER_NAME = "local"

    def initialize(self, v: Variables, logger: logging.Logger) -> LocalDataProviderService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return LocalDataProviderService(storage=storage, v=v)
