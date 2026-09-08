"""Default entity linker — no-op (external linking disabled).

Selected unless ``MEMORYLAYER_ENTITY_LINKER_PROVIDER`` names another provider (the
enterprise ``wikidata`` provider is opt-in per tenant). ``link`` always returns
``None`` so enrichment is a safe no-op out of the box.
"""

import logging

from scitrera_app_framework import Variables

from ...models.entity_registry import ExternalEntityLink
from .base import EntityLinkerService, EntityLinkerServicePluginBase


class DefaultEntityLinkerService(EntityLinkerService):
    """No-op linker: external linking is disabled."""

    async def link(self, name: str, entity_type: str) -> ExternalEntityLink | None:
        return None


class DefaultEntityLinkerServicePlugin(EntityLinkerServicePluginBase):
    """Plugin registration for the no-op entity linker."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> DefaultEntityLinkerService:
        return DefaultEntityLinkerService()
