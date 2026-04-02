from ...config import MEMORYLAYER_REFLECT_SERVICE, DEFAULT_MEMORYLAYER_REFLECT_SERVICE
from .._constants import EXT_STORAGE_BACKEND, EXT_MEMORY_SERVICE, EXT_REFLECT_SERVICE
from .._plugin_factory import make_service_plugin_base


# noinspection PyAbstractClass
ReflectServicePluginBase = make_service_plugin_base(
    ext_name=EXT_REFLECT_SERVICE,
    config_key=MEMORYLAYER_REFLECT_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_REFLECT_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_MEMORY_SERVICE),
)
