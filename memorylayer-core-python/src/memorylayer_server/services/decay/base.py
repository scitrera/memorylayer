"""Decay Service - Base interface and plugin."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ...config import MEMORYLAYER_DECAY_PROVIDER, DEFAULT_MEMORYLAYER_DECAY_PROVIDER
from ...models import Memory
from .._constants import EXT_STORAGE_BACKEND, EXT_DECAY_SERVICE
from .._plugin_factory import make_service_plugin_base


@dataclass
class DecaySettings:
    """Configuration for memory decay behavior."""
    decay_rate: float = 0.95  # Per-day decay multiplier
    min_importance: float = 0.1  # Floor - importance never drops below this
    min_age_days: int = 7  # Don't decay memories younger than this
    access_boost: float = 1.1  # Multiplier when memory is accessed
    archive_threshold: float = 0.2  # Archive when importance drops below this
    archive_min_age_days: int = 90  # Only archive memories older than this
    archive_max_access_count: int = 3  # Only archive rarely-accessed memories


@dataclass
class DecayResult:
    """Result of a decay pass."""
    processed: int = 0
    decayed: int = 0
    archived: int = 0


class DecayService(ABC):
    """Interface for memory decay and archival."""

    @abstractmethod
    async def decay_workspace(self, workspace_id: str, settings: Optional[DecaySettings] = None) -> DecayResult:
        """Run decay pass on all eligible memories in a workspace."""
        pass

    @abstractmethod
    async def archive_stale_memories(self, workspace_id: str, settings: Optional[DecaySettings] = None) -> int:
        """Archive stale low-importance memories. Returns count archived."""
        pass

    @abstractmethod
    async def calculate_access_boost(self, memory: Memory, boost_factor: Optional[float] = None) -> Optional[float]:
        pass

    @abstractmethod
    async def boost_on_access(self, workspace_id: str, memory_id: str, boost_factor: Optional[float] = None) -> Optional[float]:
        """Boost importance when memory is accessed. Returns new importance or None if not found."""
        pass

    @abstractmethod
    async def decay_all_workspaces(self, settings: Optional[DecaySettings] = None) -> DecayResult:
        """Run decay and archival across all workspaces."""
        pass


# noinspection PyAbstractClass
DecayServicePluginBase = make_service_plugin_base(
    ext_name=EXT_DECAY_SERVICE,
    config_key=MEMORYLAYER_DECAY_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_DECAY_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)
