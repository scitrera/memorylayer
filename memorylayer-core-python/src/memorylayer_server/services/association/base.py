from ...config import DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE, MEMORYLAYER_ASSOCIATION_SERVICE
from .._constants import EXT_ASSOCIATION_SERVICE, EXT_ONTOLOGY_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base

# ============================================
# Association Configuration
# ============================================
# Threshold for auto-associating similar memories
MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = "MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD"
DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 0.85


# noinspection PyAbstractClass
AssociationServicePluginBase = make_service_plugin_base(
    ext_name=EXT_ASSOCIATION_SERVICE,
    config_key=MEMORYLAYER_ASSOCIATION_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_ONTOLOGY_SERVICE),
    extra_defaults={
        MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD: DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    },
)
