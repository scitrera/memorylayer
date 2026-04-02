"""
Authentication service for MemoryLayer.

Provides identity verification and request context resolution.
"""

from .base import (
    EXT_AUTHENTICATION_SERVICE,
    HEADER_AUTHORIZATION,
    HEADER_SESSION_ID,
    AuthenticationError,
    AuthenticationService,
    AuthenticationServicePluginBase,
)
from .default import (
    OpenAuthenticationService,
    OpenAuthenticationServicePlugin,
    get_authentication_service,
)

__all__ = [
    "AuthenticationService",
    "AuthenticationServicePluginBase",
    "AuthenticationError",
    "EXT_AUTHENTICATION_SERVICE",
    "HEADER_AUTHORIZATION",
    "HEADER_SESSION_ID",
    "OpenAuthenticationService",
    "OpenAuthenticationServicePlugin",
    "get_authentication_service",
]
