"""Typed prompt-note API models."""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

_PROMPT_NOTE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


class PromptNote(BaseModel):
    id: str
    tenant_id: str
    workspace_id: str
    key: str
    title: str
    content: str
    enabled: bool
    schema_version: int
    metadata: dict[str, Any]
    revision: int
    etag: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class PromptNoteCreateInput(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=65536)
    enabled: bool = True
    schema_version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not _PROMPT_NOTE_KEY.fullmatch(value):
            raise ValueError("key must start with an alphanumeric and contain only alphanumerics, '.', '_', '/', or '-'")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title cannot be blank")
        return value


class PromptNoteReplaceInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=65536)
    enabled: bool = True
    schema_version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title cannot be blank")
        return value
