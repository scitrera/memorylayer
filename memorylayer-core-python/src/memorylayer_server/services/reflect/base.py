from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_REFLECT_SERVICE, DEFAULT_MEMORYLAYER_REFLECT_SERVICE
from ..storage import EXT_STORAGE_BACKEND
from ..memory import EXT_MEMORY_SERVICE

# Extension point constant
EXT_REFLECT_SERVICE = 'memorylayer-reflect-service'


# noinspection PyAbstractClass
class ReflectServicePluginBase(Plugin):
    """Base plugin for reflect service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_REFLECT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_REFLECT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_REFLECT_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_REFLECT_SERVICE, DEFAULT_MEMORYLAYER_REFLECT_SERVICE)

    def get_dependencies(self, v: Variables):
        return EXT_STORAGE_BACKEND, EXT_MEMORY_SERVICE
