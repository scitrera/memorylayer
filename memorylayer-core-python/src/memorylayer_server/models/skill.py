"""Skill domain models for MemoryLayer OSS.

Defines the Skill and SkillFile models plus input/update types.
Skills are agent harness knowledge units (SKILL.md + optional files)
stored in MemoryLayer and auto-mirrored as procedural memories.
"""
import re
from datetime import UTC, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Spec-mandated name rules: 1-64 chars, [a-z0-9-], no leading/trailing/consecutive hyphens
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def validate_skill_name(name: str) -> str:
    """Validate skill name against agentskills spec rules.

    Rules: 1-64 chars, lowercase letters/digits/hyphens only,
    no leading, trailing, or consecutive hyphens.
    """
    if not name:
        raise ValueError("Skill name cannot be empty")
    if len(name) > 64:
        raise ValueError(f"Skill name must be 64 chars or fewer, got {len(name)}")
    if "--" in name:
        raise ValueError("Skill name cannot contain consecutive hyphens")
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            "Skill name must contain only lowercase letters, digits, and hyphens, "
            "and must not start or end with a hyphen"
        )
    return name


class Skill(BaseModel):
    """Agent skill stored in MemoryLayer.

    Mirrors the agentskills.io spec: a SKILL.md manifest (frontmatter +
    body) plus an optional bundle of scripts, references, and assets.
    Source mode controls where canonical storage lives.
    """

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Skill ID (skl_<12hex>)")
    tenant_id: str = Field("", description="Tenant scope")
    workspace_id: str = Field(..., description="Workspace scope")
    user_id: Optional[str] = Field(None, description="User scope (set for user-private skills)")

    name: str = Field(..., description="Skill name (agentskills spec: 1-64 chars, [a-z0-9-])")
    description: str = Field(..., description="Skill description (1-1024 chars)")
    version: str = Field("0.1.0", description="Skill version (semver-ish)")
    license: Optional[str] = Field(None, description="License identifier")
    compatibility: Optional[str] = Field(None, description="Compatibility notes (max 500 chars)")
    allowed_tools: Optional[str] = Field(None, description="Space-separated tool allowlist (experimental)")

    body: str = Field("", description="SKILL.md body (post-frontmatter markdown)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary spec-allowed extras")

    source_mode: Literal["server", "filesystem", "mirrored"] = Field(
        "server", description="Canonical storage location"
    )
    manifest_hash: str = Field("", description="SHA-256 of canonical SKILL.md")
    bundle_hash: str = Field("", description="SHA-256 over sorted (path, hash) pairs of all skill files")

    enabled: bool = Field(True, description="Whether the skill is active")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_skill_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Skill description cannot be empty")
        if len(v) > 1024:
            raise ValueError(f"Skill description must be 1024 chars or fewer, got {len(v)}")
        return v

    @field_validator("compatibility")
    @classmethod
    def validate_compatibility(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 500:
            raise ValueError(f"Skill compatibility must be 500 chars or fewer, got {len(v)}")
        return v


class SkillFile(BaseModel):
    """A single file within a skill bundle.

    Stored inline (content as bytes) in OSS/SQLite. Enterprise can
    swap to object storage by adding a blob_uri column.
    """

    model_config = {"from_attributes": True}

    id: str = Field(..., description="SkillFile ID (sklf_<12hex>)")
    skill_id: str = Field(..., description="Parent skill ID")
    path: str = Field(..., description="Relative path within skill root (e.g. scripts/extract.py)")
    kind: Literal["script", "reference", "asset", "other"] = Field(
        ..., description="File kind derived from top-level directory"
    )
    content: bytes = Field(..., description="Raw file content")
    content_hash: str = Field(..., description="SHA-256 of content")
    size_bytes: int = Field(..., description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type (sniffed)")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SkillCreateInput(BaseModel):
    """Request model for creating a new skill."""

    name: str = Field(..., description="Skill name (agentskills spec)")
    description: str = Field(..., description="Skill description")
    version: str = Field("0.1.0", description="Skill version")
    license: Optional[str] = Field(None, description="License identifier")
    compatibility: Optional[str] = Field(None, description="Compatibility notes")
    allowed_tools: Optional[str] = Field(None, description="Space-separated tool allowlist")
    body: str = Field("", description="SKILL.md body markdown")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    source_mode: Literal["server", "filesystem", "mirrored"] = Field("server")
    workspace_id: Optional[str] = Field(None, description="Target workspace (overrides header)")
    user_id: Optional[str] = Field(None, description="User scope override")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_skill_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Skill description cannot be empty")
        if len(v) > 1024:
            raise ValueError(f"Skill description must be 1024 chars or fewer, got {len(v)}")
        return v


class SkillUpdateInput(BaseModel):
    """Request model for updating an existing skill (all fields optional)."""

    description: Optional[str] = Field(None, description="New description")
    version: Optional[str] = Field(None, description="New version")
    license: Optional[str] = Field(None, description="New license")
    compatibility: Optional[str] = Field(None, description="New compatibility notes")
    allowed_tools: Optional[str] = Field(None, description="New tool allowlist")
    body: Optional[str] = Field(None, description="New SKILL.md body")
    metadata: Optional[dict[str, Any]] = Field(None, description="Metadata to merge")
    source_mode: Optional[Literal["server", "filesystem", "mirrored"]] = Field(None)
    manifest_hash: Optional[str] = Field(None, description="Updated manifest hash")
    bundle_hash: Optional[str] = Field(None, description="Updated bundle hash")
    enabled: Optional[bool] = Field(None, description="Enable/disable the skill")

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not v.strip():
                raise ValueError("Skill description cannot be empty")
            if len(v) > 1024:
                raise ValueError(f"Skill description must be 1024 chars or fewer, got {len(v)}")
        return v

    @field_validator("compatibility")
    @classmethod
    def validate_compatibility(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 500:
            raise ValueError(f"Skill compatibility must be 500 chars or fewer, got {len(v)}")
        return v


class SkillFileInput(BaseModel):
    """Request model for adding or updating a file in a skill bundle."""

    path: str = Field(..., description="Relative path within skill root")
    content: bytes = Field(..., description="Raw file content")
    mime_type: Optional[str] = Field(None, description="MIME type (optional, sniffed if omitted)")
