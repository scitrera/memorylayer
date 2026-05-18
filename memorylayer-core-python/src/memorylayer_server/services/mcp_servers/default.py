"""Default McpServerService plugin implementation."""

import logging

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import McpServerServicePluginBase
from .base import McpServerService


class DefaultMcpServerServicePlugin(McpServerServicePluginBase):
    """Plugin for the default McpServerService backed by StorageBackend."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> McpServerService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return McpServerService(storage=storage)
