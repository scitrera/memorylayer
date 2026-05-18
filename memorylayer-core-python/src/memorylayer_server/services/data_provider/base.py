"""Data Provider Service — Base interface."""

from abc import ABC, abstractmethod

from ...models.data_provider import DataProvider


class DataProviderService(ABC):
    """Interface for data provider registry management."""

    @abstractmethod
    async def create_provider(self, workspace_id: str, provider: DataProvider) -> DataProvider:
        """Create a new data provider entry."""
        pass

    @abstractmethod
    async def get_provider(self, workspace_id: str, provider_id: str) -> DataProvider | None:
        """Get a data provider by ID."""
        pass

    @abstractmethod
    async def list_providers(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DataProvider], int]:
        """List data providers for a workspace. Returns (providers, total_count)."""
        pass

    @abstractmethod
    async def update_provider(
        self,
        workspace_id: str,
        provider_id: str,
        **updates,
    ) -> DataProvider | None:
        """Update data provider fields. Returns updated provider or None if not found."""
        pass

    @abstractmethod
    async def delete_provider(self, workspace_id: str, provider_id: str) -> bool:
        """Delete a data provider. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    async def sync(self, workspace_id: str, provider_id: str) -> list:
        """Trigger a sync for the provider. Returns list of documents synced.

        For LOCAL providers this is a no-op and returns an empty list.
        Enterprise providers may return ingestion job IDs or document counts.
        """
        pass
