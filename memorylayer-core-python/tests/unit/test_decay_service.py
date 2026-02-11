"""
Unit tests for DecayService.

Tests decay formula, archival criteria, boost logic, and pinned exclusion.
"""
import pytest
from datetime import datetime, timezone, timedelta

from memorylayer_server.models.memory import RememberInput, MemoryType, MemoryStatus
from memorylayer_server.services.memory import MemoryService
from memorylayer_server.services.storage.base import StorageBackend
from memorylayer_server.services.decay.base import DecaySettings, DecayResult
from memorylayer_server.services.decay import EXT_DECAY_SERVICE

from scitrera_app_framework import get_extension


@pytest.fixture
def decay_service(v):
    """Get the decay service."""
    return get_extension(EXT_DECAY_SERVICE, v)


class TestDecaySettings:
    """Tests for DecaySettings defaults."""

    def test_default_values(self):
        s = DecaySettings()
        assert s.decay_rate == 0.95
        assert s.min_importance == 0.1
        assert s.min_age_days == 7
        assert s.access_boost == 1.1
        assert s.archive_threshold == 0.2
        assert s.archive_min_age_days == 90
        assert s.archive_max_access_count == 3

    def test_custom_values(self):
        s = DecaySettings(decay_rate=0.8, min_importance=0.05)
        assert s.decay_rate == 0.8
        assert s.min_importance == 0.05


class TestDecayResult:
    """Tests for DecayResult."""

    def test_default_values(self):
        r = DecayResult()
        assert r.processed == 0
        assert r.decayed == 0
        assert r.archived == 0


class TestBoostOnAccess:
    """Tests for boost_on_access."""

    @pytest.mark.asyncio
    async def test_boost_increases_importance(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Boost test memory", type=MemoryType.SEMANTIC, importance=0.5),
        )
        new_importance = await decay_service.boost_on_access(workspace_id, memory.id)
        assert new_importance is not None
        assert new_importance > 0.5

    @pytest.mark.asyncio
    async def test_boost_caps_at_one(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="High importance boost test", type=MemoryType.SEMANTIC, importance=0.99),
        )
        new_importance = await decay_service.boost_on_access(workspace_id, memory.id)
        assert new_importance is not None
        assert new_importance <= 1.0

    @pytest.mark.asyncio
    async def test_boost_custom_factor(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str,
    ):
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Custom boost factor test", type=MemoryType.SEMANTIC, importance=0.5),
        )
        new_importance = await decay_service.boost_on_access(workspace_id, memory.id, boost_factor=1.5)
        assert new_importance is not None
        assert new_importance == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_boost_nonexistent_memory(self, decay_service, workspace_id: str):
        result = await decay_service.boost_on_access(workspace_id, "mem_nonexistent")
        assert result is None


class TestDecayWorkspace:
    """Tests for decay_workspace."""

    @pytest.mark.asyncio
    async def test_decay_reduces_importance(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Memories with old last_accessed_at should have reduced importance."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Decay reduction test", type=MemoryType.SEMANTIC, importance=0.8),
        )
        # Make the memory look old by updating created_at and last_accessed_at
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            created_at=old_time,
            last_accessed_at=old_time,
        )

        settings = DecaySettings(min_age_days=1)
        result = await decay_service.decay_workspace(workspace_id, settings)

        assert result.processed > 0

        # Fetch and check importance decreased
        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.importance < 0.8

    @pytest.mark.asyncio
    async def test_decay_respects_min_age(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Recent memories should NOT be decayed."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Recent no-decay test", type=MemoryType.SEMANTIC, importance=0.8),
        )
        original_importance = memory.importance

        settings = DecaySettings(min_age_days=365)  # Nothing is 365 days old
        result = await decay_service.decay_workspace(workspace_id, settings)

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.importance == original_importance

    @pytest.mark.asyncio
    async def test_decay_skips_pinned(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Pinned memories should NOT be decayed."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Pinned no-decay test", type=MemoryType.SEMANTIC, importance=0.8),
        )
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            pinned=1,
            created_at=old_time,
            last_accessed_at=old_time,
        )

        settings = DecaySettings(min_age_days=1)
        await decay_service.decay_workspace(workspace_id, settings)

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.importance == 0.8

    @pytest.mark.asyncio
    async def test_decay_respects_min_importance_floor(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Importance should never drop below min_importance."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Floor test decay", type=MemoryType.SEMANTIC, importance=0.3),
        )
        very_old = (datetime.now(timezone.utc) - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            created_at=very_old,
            last_accessed_at=very_old,
        )

        settings = DecaySettings(min_age_days=1, min_importance=0.15)
        await decay_service.decay_workspace(workspace_id, settings)

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.importance >= 0.15


class TestArchiveStaleMemories:
    """Tests for archive_stale_memories."""

    @pytest.mark.asyncio
    async def test_archive_low_importance_old_memory(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Low importance, old, rarely accessed memories should be archived."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Stale archive candidate", type=MemoryType.SEMANTIC, importance=0.1),
        )
        very_old = (datetime.now(timezone.utc) - timedelta(days=120)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            created_at=very_old,
            importance=0.1,
        )

        settings = DecaySettings(
            archive_threshold=0.2,
            archive_min_age_days=90,
            archive_max_access_count=3,
        )
        archived_count = await decay_service.archive_stale_memories(workspace_id, settings)
        assert archived_count >= 1

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.status == MemoryStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_archive_skips_high_importance(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """High importance memories should NOT be archived."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="High importance no archive", type=MemoryType.SEMANTIC, importance=0.9),
        )
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            created_at=old_time,
        )

        settings = DecaySettings(archive_threshold=0.2, archive_min_age_days=90)
        await decay_service.archive_stale_memories(workspace_id, settings)

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.status == MemoryStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_archive_skips_pinned(
        self, decay_service, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        """Pinned memories should NOT be archived even if low importance."""
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Pinned no archive test", type=MemoryType.SEMANTIC, importance=0.1),
        )
        old_time = (datetime.now(timezone.utc) - timedelta(days=120)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id,
            created_at=old_time,
            pinned=1,
            importance=0.1,
        )

        settings = DecaySettings(archive_threshold=0.2, archive_min_age_days=90)
        await decay_service.archive_stale_memories(workspace_id, settings)

        updated = await storage_backend.get_memory(workspace_id, memory.id)
        assert updated.status == MemoryStatus.ACTIVE


class TestStorageDecayMethods:
    """Tests for storage backend decay-related methods."""

    @pytest.mark.asyncio
    async def test_get_memories_for_decay(
        self, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Decay eligible storage test", type=MemoryType.SEMANTIC),
        )
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id, created_at=old_time,
        )

        memories = await storage_backend.get_memories_for_decay(
            workspace_id, min_age_days=1, exclude_pinned=True,
        )
        found_ids = [m.id for m in memories]
        assert memory.id in found_ids

    @pytest.mark.asyncio
    async def test_get_memories_for_decay_excludes_pinned(
        self, memory_service: MemoryService,
        workspace_id: str, storage_backend: StorageBackend,
    ):
        memory = await memory_service.remember(
            workspace_id,
            RememberInput(content="Pinned decay exclusion test", type=MemoryType.SEMANTIC),
        )
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        await storage_backend.update_memory(
            workspace_id, memory.id, created_at=old_time, pinned=1,
        )

        memories = await storage_backend.get_memories_for_decay(
            workspace_id, min_age_days=1, exclude_pinned=True,
        )
        found_ids = [m.id for m in memories]
        assert memory.id not in found_ids

    @pytest.mark.asyncio
    async def test_list_all_workspace_ids(self, storage_backend: StorageBackend):
        ids = await storage_backend.list_all_workspace_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0  # At least the default workspace exists
