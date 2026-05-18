"""
Authentication and authorization context models.

These models represent the resolved identity and context for API requests.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from .session import Session


@dataclass
class PrincipalRef:
    """Reference to an authenticated or delegated principal."""

    type: str  # "user" | "service" | "agent" | "task"
    id: str  # canonical identity string


@dataclass
class AuthorityContext:
    """On-behalf-of delegation context parsed from Aether OBO headers.

    Populated by AetherAuthenticationService when the auth-proxy injects
    X-Auth-Authority-Mode == "on_behalf_of". The OSS open-auth path
    leaves this None (no grant store to enforce against).
    """

    mode: Literal["direct", "on_behalf_of"] = "direct"
    grant_id: str | None = None
    subject: PrincipalRef | None = None  # whose authority is being exercised
    root_subject: PrincipalRef | None = None  # top of delegation chain
    audience_type: str | None = None
    audience_id: str | None = None
    max_access_level: int | None = None  # grant ceiling (None = uncapped)
    workspace_scope: list[str] | None = None  # None or empty = any workspace


@dataclass
class AuthIdentity:
    """
    Verified identity from authentication.

    In OSS, this always returns default tenant with no user.
    In Enterprise, this is populated from API key or JWT verification.
    """

    tenant_id: str
    user_id: str | None = None
    api_key_id: str | None = None  # For audit/tracking


@dataclass
class RequestContext:
    """
    Fully resolved context for an API request.

    This is the contract between authentication and business logic.
    All service operations should use this context for scoping.

    Resolution priority for workspace_id:
    1. Explicit workspace_id in request body (override)
    2. Session's workspace_id (from session)
    3. DEFAULT_WORKSPACE_ID ("_default")

    The metadata dict carries extension-specific data (e.g., gateway-injected
    access levels) without coupling the core model to any particular auth scheme.

    OBO fields:
    - actor: authenticated connection identity (service/agent acting on behalf of subject)
    - authority: delegation context when mode == on_behalf_of
    - user_id: backward-compat shim; == authority.subject.id when OBO is active
    """

    tenant_id: str
    workspace_id: str
    user_id: str | None = None
    actor: PrincipalRef | None = None
    authority: AuthorityContext | None = None
    session: Session | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str | None:
        """Convenience property to get session ID if session exists."""
        return self.session.id if self.session else None

    @property
    def context_id(self) -> str | None:
        """Get context_id from session if available."""
        return self.session.context_id if self.session else None

    def effective_subject_id(self) -> str | None:
        """Return the verified subject identity for data scoping.

        When OBO is active, returns the grant subject (verified by auth-proxy).
        Falls back to user_id for direct auth or legacy callers.
        """
        if self.authority and self.authority.subject:
            return self.authority.subject.id
        return self.user_id
