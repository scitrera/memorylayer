"""
Workspace Service - Business logic for workspace operations.

Operations:
- create_workspace: Create a new workspace
- get_workspace: Retrieve workspace by ID
- update_workspace: Update workspace settings
"""

from ...config import MEMORYLAYER_WORKSPACE_SERVICE, DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE
from .._constants import EXT_STORAGE_BACKEND, EXT_WORKSPACE_SERVICE
from .._plugin_factory import make_service_plugin_base


# noinspection PyAbstractClass
WorkspaceServicePluginBase = make_service_plugin_base(
    ext_name=EXT_WORKSPACE_SERVICE,
    config_key=MEMORYLAYER_WORKSPACE_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND,),
)
