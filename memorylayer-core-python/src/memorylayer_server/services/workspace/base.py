"""
Workspace Service - Business logic for workspace operations.

Operations:
- create_workspace: Create a new workspace
- get_workspace: Retrieve workspace by ID
- update_workspace: Update workspace settings
"""

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_WORKSPACE_SERVICE, DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE
from .._constants import EXT_STORAGE_BACKEND, EXT_WORKSPACE_SERVICE


# noinspection PyAbstractClass
class WorkspaceServicePluginBase(Plugin):
    """Base plugin for workspace service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_WORKSPACE_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_WORKSPACE_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_WORKSPACE_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_WORKSPACE_SERVICE, DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)
