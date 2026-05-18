"""Data provider domain models for MemoryLayer OSS.

Defines the DataProvider model and type enum. In OSS, the only provider
type is LOCAL (direct upload / local filesystem). Enterprise extends
this with external sources (S3, GCS, SharePoint, Confluence, web).
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DataProviderType(str, Enum):
    """Built-in data provider types for OSS.

    Enterprise adds additional types (s3, gcs, azure_blob, sharepoint,
    confluence, web) via its own enum or string values.
    """

    LOCAL = "local"


class DataProvider(BaseModel):
    """Data provider registry entry.

    Represents a configured source for document ingestion. In OSS, only
    the LOCAL type is supported (direct upload). Enterprise extends with
    external sources and scheduled sync.
    """

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Provider ID")
    tenant_id: str = Field("", description="Tenant scope (multi-tenant)")
    workspace_id: str = Field(..., description="Workspace scope")
    name: str = Field(..., description="Provider name")
    # str (not enum) so enterprise can add its own types without modifying OSS
    provider_type: str = Field(..., description="Provider type (local, s3, gcs, ...)")
    description: str | None = Field(None, description="Provider description")
    enabled: bool = Field(True, description="Whether the provider is active")
    connection_args: dict[str, Any] = Field(default_factory=dict, description="Non-sensitive connection arguments")
    schedule: str | None = Field(None, description="Cron schedule for auto-sync (enterprise)")
    last_sync_at: datetime | None = Field(None, description="Last successful sync time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Provider name cannot be empty")
        return v.strip()
