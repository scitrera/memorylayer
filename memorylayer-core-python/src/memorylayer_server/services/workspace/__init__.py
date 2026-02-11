"""Workspace service package."""
from .base import (
    WorkspaceServicePluginBase,
    EXT_WORKSPACE_SERVICE,
)
from .default import WorkspaceService

from scitrera_app_framework import Variables, get_extension


def get_workspace_service(v: Variables = None) -> WorkspaceService:
    """Get the workspace service instance."""
    return get_extension(EXT_WORKSPACE_SERVICE, v)


__all__ = (
    'WorkspaceService',
    'WorkspaceServicePluginBase',
    'get_workspace_service',
    'EXT_WORKSPACE_SERVICE',
)
