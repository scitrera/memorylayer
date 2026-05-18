from enum import Enum

from pydantic import BaseModel, Field


class AuthorizationDecision(str, Enum):
    """Authorization decision result."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"  # Let next handler decide (for chain-of-responsibility patterns)


class AuthorizationContext(BaseModel):
    """Context for authorization decisions.

    Contains all information needed to make an authorization decision.
    """

    model_config = {"frozen": True}

    tenant_id: str | None = Field(None, description="Tenant identifier")
    workspace_id: str | None = Field(None, description="Workspace identifier")
    user_id: str | None = Field(None, description="User identifier")
    resource: str = Field("", description="Resource type (e.g., 'memories', 'workspaces')")
    action: str = Field("", description="Action type (e.g., 'read', 'write', 'delete')")
    resource_id: str | None = Field(None, description="Specific resource ID")
    metadata: dict = Field(default_factory=dict, description="Additional context")

    # OBO actor/authority fields — populated from RequestContext when available
    actor_type: str | None = Field(None, description="Authenticated connection principal type")
    actor_id: str | None = Field(None, description="Authenticated connection principal id")
    subject_type: str | None = Field(None, description="Subject (on-behalf-of) principal type")
    subject_id: str | None = Field(None, description="Subject (on-behalf-of) principal id")
    authority_mode: str | None = Field(None, description="'direct' or 'on_behalf_of'")
    grant_id: str | None = Field(None, description="Aether grant id when OBO is active")
    max_access_level: int | None = Field(None, description="Grant ceiling from OBO grant")
