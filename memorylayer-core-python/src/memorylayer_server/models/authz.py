from enum import Enum
from typing import Optional

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

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    resource: str = Field("", description="Resource type (e.g., 'memories', 'workspaces')")
    action: str = Field("", description="Action type (e.g., 'read', 'write', 'delete')")
    resource_id: Optional[str] = Field(None, description="Specific resource ID")
    metadata: dict = Field(default_factory=dict, description="Additional context")
