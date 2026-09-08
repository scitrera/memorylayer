"""Entity Linker Service — canonical entity -> external knowledge base.

Resolves a canonical entity (surface name + type) to an external KB record (e.g.
a Wikidata QID + label + description). The OSS ``default`` provider is a no-op
(linking disabled); the enterprise ``wikidata`` provider is opt-in per tenant. The
enrichment path (``EntityRegistryService.enrich_entities``) folds a returned link
into the entity ``provenance`` — it never blocks on linking and only writes when a
confident link is returned.
"""

from abc import ABC, abstractmethod

from ...config import (
    DEFAULT_MEMORYLAYER_ENTITY_LINKER_PROVIDER,
    MEMORYLAYER_ENTITY_LINKER_PROVIDER,
)
from ...models.entity_registry import ExternalEntityLink
from .._constants import EXT_ENTITY_LINKER_SERVICE
from .._plugin_factory import make_service_plugin_base


class EntityLinkerService(ABC):
    """Resolve a canonical entity to an external KB record (or None)."""

    @abstractmethod
    async def link(self, name: str, entity_type: str) -> ExternalEntityLink | None:
        """Return the best external link for ``(name, entity_type)``, or ``None``.

        Implementations MUST be HIGH-PRECISION (link only on a confident match, to
        avoid wrong links) and FAIL-SAFE (any error -> ``None``). The enrichment
        path only writes provenance when a link is returned.
        """
        ...


EntityLinkerServicePluginBase = make_service_plugin_base(
    ext_name=EXT_ENTITY_LINKER_SERVICE,
    config_key=MEMORYLAYER_ENTITY_LINKER_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_ENTITY_LINKER_PROVIDER,
)
