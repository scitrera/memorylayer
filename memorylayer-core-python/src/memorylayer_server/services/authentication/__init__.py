"""
Authentication service for MemoryLayer.

Provides identity verification and request context resolution.
"""
from .base import (
    AuthenticationService,
    AuthenticationError,
    EXT_AUTHENTICATION_SERVICE,
    HEADER_AUTHORIZATION,
    HEADER_SESSION_ID,
)
from .default import (
    OpenAuthenticationService,
    OpenAuthenticationServicePlugin,
    get_authentication_service,
)

__all__ = [
    "AuthenticationService",
    "AuthenticationError",
    "EXT_AUTHENTICATION_SERVICE",
    "HEADER_AUTHORIZATION",
    "HEADER_SESSION_ID",
    "OpenAuthenticationService",
    "OpenAuthenticationServicePlugin",
    "get_authentication_service",
]
