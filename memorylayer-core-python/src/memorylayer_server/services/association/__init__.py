"""Association service package."""
from .base import (
    AssociationServicePluginBase,
    EXT_ASSOCIATION_SERVICE,
    MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
)
from .default import AssociationService

from scitrera_app_framework import Variables, get_extension


def get_association_service(v: Variables = None) -> AssociationService:
    """Get the association service instance."""
    return get_extension(EXT_ASSOCIATION_SERVICE, v)


__all__ = (
    'AssociationService',
    'AssociationServicePluginBase',
    'get_association_service',
    'EXT_ASSOCIATION_SERVICE',
    'MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD',
    'DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD',
)
