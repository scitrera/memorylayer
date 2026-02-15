from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_MEMORY_SERVICE, DEFAULT_MEMORYLAYER_MEMORY_SERVICE
from .._constants import (
    EXT_CACHE_SERVICE,
    EXT_CONTRADICTION_SERVICE,
    EXT_DEDUPLICATION_SERVICE,
    EXT_EMBEDDING_PROVIDER,
    EXT_EXTRACTION_SERVICE,
    EXT_MEMORY_SERVICE,
    EXT_RERANKER_SERVICE,
    EXT_SEMANTIC_TIERING_SERVICE,
    EXT_STORAGE_BACKEND,
)

# Recall overfetch multiplier for reranker candidate pool
MEMORYLAYER_MEMORY_RECALL_OVERFETCH = 'MEMORYLAYER_MEMORY_RECALL_OVERFETCH'
DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH = 3

# Maximum memories discovered via association graph expansion
MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION = 'MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION'
DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION = 50

# Default include_associations for recall (graph expansion enabled by default)
MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS = 'MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS'
DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS = True

# Default traverse_depth for recall (multi-hop graph traversal)
MEMORYLAYER_MEMORY_TRAVERSE_DEPTH = 'MEMORYLAYER_MEMORY_TRAVERSE_DEPTH'
DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH = 2


# noinspection PyAbstractClass
class MemoryServicePluginBase(Plugin):
    """Base plugin for memory service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_MEMORY_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MEMORY_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_MEMORY_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_MEMORY_SERVICE, DEFAULT_MEMORYLAYER_MEMORY_SERVICE)
        v.set_default_value(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH)
        v.set_default_value(MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION, DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION)
        v.set_default_value(MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS, DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS)
        v.set_default_value(MEMORYLAYER_MEMORY_TRAVERSE_DEPTH, DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH)

    def get_dependencies(self, v: Variables):
        return (
            EXT_STORAGE_BACKEND,
            EXT_EMBEDDING_PROVIDER,
            EXT_CACHE_SERVICE,
            EXT_SEMANTIC_TIERING_SERVICE,
            EXT_DEDUPLICATION_SERVICE,
            EXT_RERANKER_SERVICE,
            EXT_CONTRADICTION_SERVICE,
            EXT_EXTRACTION_SERVICE,
        )
