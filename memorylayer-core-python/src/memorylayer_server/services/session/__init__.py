"""Session service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_SESSION_SERVICE,
    CommitOptions,
    CommitResult,
    SessionService,
    SessionServicePluginBase,
)


def get_session_service(v: Variables = None) -> SessionService:
    """Get the session service instance."""
    return get_extension(EXT_SESSION_SERVICE, v)


__all__ = (
    "SessionService",
    "SessionServicePluginBase",
    "get_session_service",
    "EXT_SESSION_SERVICE",
    "CommitResult",
    "CommitOptions",
)
