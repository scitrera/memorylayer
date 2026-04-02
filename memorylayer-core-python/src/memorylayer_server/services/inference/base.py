"""
Inference Service - Base interface and plugin.

Derives higher-order insights and conclusions from accumulated memories
about an entity (subject). Unlike extraction (which captures what was said),
inference derives what patterns and behaviors *mean* about the entity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ...config import DEFAULT_MEMORYLAYER_INFERENCE_SERVICE, MEMORYLAYER_INFERENCE_SERVICE
from ...models.memory import Memory
from .._constants import (
    EXT_ASSOCIATION_SERVICE,
    EXT_CACHE_SERVICE,
    EXT_INFERENCE_SERVICE,
    EXT_LLM_SERVICE,
    EXT_MEMORY_SERVICE,
    EXT_STORAGE_BACKEND,
)
from .._plugin_factory import make_service_plugin_base


@dataclass
class InferenceResult:
    """Result of an inference derivation operation."""

    subject_id: str
    workspace_id: str
    insights_created: int = 0
    insights_updated: int = 0
    source_memory_count: int = 0
    insights: list[Memory] = field(default_factory=list)


class InferenceService(ABC):
    """Interface for inference/insight derivation service."""

    @abstractmethod
    async def derive_insights(
        self,
        workspace_id: str,
        subject_id: str,
        observer_id: str | None = None,
        force: bool = False,
    ) -> InferenceResult:
        """Derive higher-order insights about a subject from accumulated memories.

        Analyzes all memories where subject_id matches, identifies patterns,
        and creates new INFERENCE-subtype memories capturing derived conclusions.

        Args:
            workspace_id: Workspace boundary
            subject_id: Entity to derive insights about
            observer_id: Optional observer perspective (whose memories to analyze)
            force: Re-derive even if recent insights exist

        Returns:
            InferenceResult with created/updated insight memories
        """
        pass

    @abstractmethod
    async def get_insights(
        self,
        workspace_id: str,
        subject_id: str,
        observer_id: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Retrieve existing derived insights about a subject.

        Args:
            workspace_id: Workspace boundary
            subject_id: Entity whose insights to retrieve
            observer_id: Optional observer perspective filter
            limit: Maximum insights to return

        Returns:
            List of INFERENCE-subtype memories
        """
        pass


# noinspection PyAbstractClass
InferenceServicePluginBase = make_service_plugin_base(
    ext_name=EXT_INFERENCE_SERVICE,
    config_key=MEMORYLAYER_INFERENCE_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_INFERENCE_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_MEMORY_SERVICE, EXT_LLM_SERVICE, EXT_ASSOCIATION_SERVICE, EXT_CACHE_SERVICE),
)
