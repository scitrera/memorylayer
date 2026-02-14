"""Default decay service implementation."""
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models import Memory
from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from .base import DecayService, DecayServicePluginBase, DecaySettings, DecayResult


class DefaultDecayService(DecayService):
    """Default decay implementation using storage backend directly."""

    def __init__(self, storage: StorageBackend, v: Variables = None):
        self._storage = storage
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def decay_workspace(self, workspace_id: str, settings: Optional[DecaySettings] = None) -> DecayResult:
        settings = settings or DecaySettings()
        result = DecayResult()

        memories = await self._storage.get_memories_for_decay(
            workspace_id,
            min_age_days=settings.min_age_days,
            exclude_pinned=True,
        )
        result.processed = len(memories)

        now = datetime.now(timezone.utc)
        for memory in memories:
            last_access = memory.last_accessed_at or memory.created_at
            # Ensure timezone-aware comparison
            if last_access.tzinfo is None:
                last_access = last_access.replace(tzinfo=timezone.utc)
            days_since_access = max(0, (now - last_access).days)

            new_importance = max(
                settings.min_importance,
                memory.importance * (settings.decay_rate ** days_since_access)
            )

            if abs(new_importance - memory.importance) > 0.001:
                await self._storage.update_memory(
                    workspace_id, memory.id,
                    importance=new_importance,
                    decay_factor=new_importance / max(memory.importance, 0.001),
                )
                result.decayed += 1

        self.logger.debug(
            "Decay pass for workspace %s: %d processed, %d decayed",
            workspace_id, result.processed, result.decayed
        )
        return result

    async def archive_stale_memories(self, workspace_id: str, settings: Optional[DecaySettings] = None) -> int:
        settings = settings or DecaySettings()

        candidates = await self._storage.get_archival_candidates(
            workspace_id,
            max_importance=settings.archive_threshold,
            max_access_count=settings.archive_max_access_count,
            min_age_days=settings.archive_min_age_days,
        )

        archived = 0
        for memory in candidates:
            await self._storage.update_memory(
                workspace_id, memory.id,
                status='archived',
            )
            archived += 1

        if archived:
            self.logger.info(
                "Archived %d stale memories in workspace %s",
                archived, workspace_id
            )
        return archived

    async def calculate_access_boost(self, memory: Memory, boost_factor: Optional[float] = None) -> Optional[float]:
        boost = boost_factor or DecaySettings().access_boost
        if not memory or memory.pinned:
            return memory.importance if memory else None

        new_importance = min(1.0, memory.importance * boost)
        return new_importance

    async def boost_on_access(self, workspace_id: str, memory_id: str, boost_factor: Optional[float] = None) -> Optional[float]:
        memory = await self._storage.get_memory(workspace_id, memory_id, track_access=False)
        if not memory or memory.pinned:
            return memory.importance if memory else None

        new_importance = await self.calculate_access_boost(memory, boost_factor=boost_factor)
        if abs(new_importance - memory.importance) > 0.001:
            await self._storage.update_memory(
                workspace_id, memory_id,
                importance=new_importance,
            )
        return new_importance

    async def decay_all_workspaces(self, settings: Optional[DecaySettings] = None) -> DecayResult:
        settings = settings or DecaySettings()
        total = DecayResult()

        workspaces = await self._storage.list_all_workspace_ids()

        for ws_id in workspaces:
            ws_result = await self.decay_workspace(ws_id, settings)
            total.processed += ws_result.processed
            total.decayed += ws_result.decayed

            archived = await self.archive_stale_memories(ws_id, settings)
            total.archived += archived

        self.logger.info(
            "Decay all workspaces: %d processed, %d decayed, %d archived",
            total.processed, total.decayed, total.archived
        )
        return total


class DefaultDecayServicePlugin(DecayServicePluginBase):
    """Plugin that creates the default decay service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> DecayService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultDecayService(storage=storage, v=v)
