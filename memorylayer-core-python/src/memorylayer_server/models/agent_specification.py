"""Typed reusable agent-specification API models."""

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_AGENT_SPEC_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
PermissionMode = Literal["inherit", "ask", "allow", "read_only"]


class AgentSpecification(BaseModel):
    id: str
    tenant_id: str
    workspace_id: str
    key: str
    name: str
    description: str
    instructions: str
    invocation_guidance: str = ""
    model: str = ""
    max_turns: int = 0
    allowed_tools: list[str]
    denied_tools: list[str]
    skills: list[str]
    mcp_servers: list[str]
    permission_mode: PermissionMode = "inherit"
    exec_policy_hint: str = ""
    background: bool = False
    enabled: bool = True
    schema_version: int
    metadata: dict[str, Any]
    revision: int
    etag: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class AgentSpecificationContent(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4096)
    instructions: str = Field(min_length=1, max_length=65536)
    invocation_guidance: str = Field(default="", max_length=4096)
    model: str = Field(default="", max_length=200)
    max_turns: int = Field(default=0, ge=0, le=1000)
    allowed_tools: list[str] = Field(default_factory=list, max_length=512)
    denied_tools: list[str] = Field(default_factory=list, max_length=512)
    skills: list[str] = Field(default_factory=list, max_length=512)
    mcp_servers: list[str] = Field(default_factory=list, max_length=512)
    permission_mode: PermissionMode = "inherit"
    exec_policy_hint: str = Field(default="", max_length=4096)
    background: bool = False
    enabled: bool = True
    schema_version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None

    @field_validator("name", "description", "instructions")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("allowed_tools", "denied_tools", "skills", "mcp_servers")
    @classmethod
    def normalize_unique_names(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list values cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("list values must be unique")
        return normalized


class AgentSpecificationCreateInput(AgentSpecificationContent):
    key: str = Field(min_length=1, max_length=200)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not _AGENT_SPEC_KEY.fullmatch(value):
            raise ValueError("key must start with an alphanumeric and contain only alphanumerics, '.', '_', '/', or '-'")
        return value


class AgentSpecificationReplaceInput(AgentSpecificationContent):
    pass
