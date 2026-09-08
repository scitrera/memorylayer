"""Internal revision authority used by typed MemoryLayer resource services."""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_STORAGE_BACKEND, EXT_VERSIONED_RESOURCE_SERVICE
from .._plugin_factory import make_service_plugin_base
from .base import VersionedResourceService

MEMORYLAYER_VERSIONED_RESOURCE_PROVIDER = "MEMORYLAYER_VERSIONED_RESOURCE_PROVIDER"
DEFAULT_MEMORYLAYER_VERSIONED_RESOURCE_PROVIDER = "default"

VersionedResourceServicePluginBase = make_service_plugin_base(
    ext_name=EXT_VERSIONED_RESOURCE_SERVICE,
    config_key=MEMORYLAYER_VERSIONED_RESOURCE_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_VERSIONED_RESOURCE_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)


def get_versioned_resource_service(v: Variables = None) -> VersionedResourceService:
    return get_extension(EXT_VERSIONED_RESOURCE_SERVICE, v)


__all__ = (
    "EXT_VERSIONED_RESOURCE_SERVICE",
    "VersionedResourceService",
    "VersionedResourceServicePluginBase",
    "get_versioned_resource_service",
)
