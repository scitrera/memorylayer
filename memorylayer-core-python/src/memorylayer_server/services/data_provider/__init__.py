"""Data provider service package."""

from scitrera_app_framework import Variables, get_extension

from ...config import DEFAULT_MEMORYLAYER_DATA_PROVIDER_PROVIDER, MEMORYLAYER_DATA_PROVIDER_PROVIDER
from .._constants import EXT_DATA_PROVIDER_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base
from .base import DataProviderService

DataProviderServicePluginBase = make_service_plugin_base(
    ext_name=EXT_DATA_PROVIDER_SERVICE,
    config_key=MEMORYLAYER_DATA_PROVIDER_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_DATA_PROVIDER_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)


def get_data_provider_service(v: Variables = None) -> DataProviderService:
    """Get the data provider service instance."""
    return get_extension(EXT_DATA_PROVIDER_SERVICE, v)


__all__ = (
    "DataProviderService",
    "DataProviderServicePluginBase",
    "get_data_provider_service",
    "EXT_DATA_PROVIDER_SERVICE",
)
