"""Factory for generating service plugin base classes with common boilerplate."""
from scitrera_app_framework import Plugin, Variables
from scitrera_app_framework.api import enabled_option_pattern


def make_service_plugin_base(
        *,
        ext_name: str,
        config_key: str,
        default_value: str,
        dependencies: tuple[str, ...] = (),
        extra_defaults: dict | None = None,
) -> type[Plugin]:
    """Create a PluginBase class with standard service plugin boilerplate.

    Args:
        ext_name: Extension point name (e.g., EXT_DECAY_SERVICE)
        config_key: Config variable name for provider selection
        default_value: Default provider name
        dependencies: Tuple of extension point names this service depends on
        extra_defaults: Optional dict of additional {config_key: default_value}
            pairs to register in on_registration

    Returns:
        A Plugin subclass with name(), extension_point_name(), is_enabled(),
        on_registration(), and get_dependencies() already implemented.

    Example:
        DecayServicePluginBase = make_service_plugin_base(
            ext_name=EXT_DECAY_SERVICE,
            config_key=MEMORYLAYER_DECAY_PROVIDER,
            default_value=DEFAULT_MEMORYLAYER_DECAY_PROVIDER,
            dependencies=(EXT_STORAGE_BACKEND,),
        )
    """
    _extra = extra_defaults or {}

    # noinspection PyAbstractClass
    class _ServicePluginBase(Plugin):
        PROVIDER_NAME: str = None

        def name(self) -> str:
            return f"{ext_name}|{self.PROVIDER_NAME}"

        def extension_point_name(self, v: Variables) -> str:
            return ext_name

        def is_enabled(self, v: Variables) -> bool:
            return enabled_option_pattern(self, v, config_key, self_attr='PROVIDER_NAME')

        def on_registration(self, v: Variables) -> None:
            v.set_default_value(config_key, default_value)
            for k, dv in _extra.items():
                v.set_default_value(k, dv)

        def get_dependencies(self, v: Variables):
            return dependencies

    _ServicePluginBase.__qualname__ = f"make_service_plugin_base.<{ext_name}>"
    return _ServicePluginBase
