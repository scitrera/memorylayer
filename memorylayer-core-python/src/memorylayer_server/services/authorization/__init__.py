"""Authorization service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_AUTHORIZATION_SERVICE,
    AuthorizationService,
    AuthorizationServicePluginBase,
)


def get_authorization_service(v: Variables = None) -> AuthorizationService:
    """Get the authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


__all__ = (
    "AuthorizationService",
    "AuthorizationServicePluginBase",
    "get_authorization_service",
    "EXT_AUTHORIZATION_SERVICE",
)
