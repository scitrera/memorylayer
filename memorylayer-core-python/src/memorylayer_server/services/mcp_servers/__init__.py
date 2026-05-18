"""MCP servers service package."""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base
from .base import McpServerService

EXT_MCP_SERVERS_SERVICE = "memorylayer-mcp-servers-service"
MEMORYLAYER_MCP_SERVERS_PROVIDER = "MEMORYLAYER_MCP_SERVERS_PROVIDER"
DEFAULT_MEMORYLAYER_MCP_SERVERS_PROVIDER = "default"

McpServerServicePluginBase = make_service_plugin_base(
    ext_name=EXT_MCP_SERVERS_SERVICE,
    config_key=MEMORYLAYER_MCP_SERVERS_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_MCP_SERVERS_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)


def get_mcp_servers_service(v: Variables = None) -> McpServerService:
    """Get the MCP servers service instance."""
    return get_extension(EXT_MCP_SERVERS_SERVICE, v)


__all__ = (
    "McpServerService",
    "McpServerServicePluginBase",
    "get_mcp_servers_service",
    "EXT_MCP_SERVERS_SERVICE",
)
