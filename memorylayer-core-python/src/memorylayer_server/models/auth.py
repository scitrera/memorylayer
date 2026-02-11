"""
Authentication and authorization context models.

These models represent the resolved identity and context for API requests.
"""
from dataclasses import dataclass
from typing import Optional

from .session import Session


@dataclass
class AuthIdentity:
    """
    Verified identity from authentication.

    In OSS, this always returns default tenant with no user.
    In Enterprise, this is populated from API key or JWT verification.
    """
    tenant_id: str
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None  # For audit/tracking


@dataclass
class RequestContext:
    """
    Fully resolved context for an API request.

    This is the contract between authentication and business logic.
    All service operations should use this context for scoping.

    Resolution priority for workspace_id:
    1. Explicit workspace_id in request body (override)
    2. Session's workspace_id (from X-Session-ID header)
    3. DEFAULT_WORKSPACE_ID ("_default")
    """
    tenant_id: str
    workspace_id: str
    user_id: Optional[str] = None
    session: Optional[Session] = None

    @property
    def session_id(self) -> Optional[str]:
        """Convenience property to get session ID if session exists."""
        return self.session.id if self.session else None

    @property
    def context_id(self) -> Optional[str]:
        """Get context_id from session if available."""
        return self.session.context_id if self.session else None
