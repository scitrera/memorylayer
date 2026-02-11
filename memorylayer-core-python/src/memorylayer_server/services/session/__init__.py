"""Session service package."""
from .base import (
    SessionService,
    SessionServicePluginBase,
    EXT_SESSION_SERVICE,
    CommitResult,
    CommitOptions,
)

from scitrera_app_framework import Variables, get_extension


def get_session_service(v: Variables = None) -> SessionService:
    """Get the session service instance."""
    return get_extension(EXT_SESSION_SERVICE, v)


__all__ = (
    'SessionService',
    'SessionServicePluginBase',
    'get_session_service',
    'EXT_SESSION_SERVICE',
    'CommitResult',
    'CommitOptions',
)
