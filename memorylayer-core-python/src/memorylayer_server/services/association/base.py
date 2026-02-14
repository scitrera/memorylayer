from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_ASSOCIATION_SERVICE, DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE
from .._constants import EXT_ASSOCIATION_SERVICE, EXT_ONTOLOGY_SERVICE, EXT_STORAGE_BACKEND

# ============================================
# Association Configuration
# ============================================
# Threshold for auto-associating similar memories
MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 'MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD'
DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 0.85


# noinspection PyAbstractClass
class AssociationServicePluginBase(Plugin):
    """Base plugin for association service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_ASSOCIATION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_ASSOCIATION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_ASSOCIATION_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_ASSOCIATION_SERVICE, DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE)
        v.set_default_value(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_ONTOLOGY_SERVICE)
