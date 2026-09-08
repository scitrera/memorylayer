"""Deterministic versioning helpers for native skill manifests.

Skill bundle files are child resources. Their aggregate ``bundle_hash`` is
intentionally absent from the manifest state hash so a file upload does not
invalidate an otherwise independent manifest refinement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ...models.skill import Skill
from ...utils import to_utc_iso


def manifest_state(skill: Skill) -> dict[str, Any]:
    """Return the complete semantic state protected by the skill ETag."""

    return {
        "tenant_id": skill.tenant_id,
        "workspace_id": skill.workspace_id,
        "user_id": skill.user_id,
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
        "license": skill.license,
        "compatibility": skill.compatibility,
        "allowed_tools": skill.allowed_tools,
        "body": skill.body,
        "metadata": skill.metadata,
        "source_mode": skill.source_mode,
        "manifest_hash": skill.manifest_hash,
        "enabled": skill.enabled,
        "deleted_at": skill.deleted_at,
    }


def canonical_hash(value: Any) -> str:
    """Hash JSON with the same stable rules used by versioned resources."""

    def default(item: Any) -> str:
        if isinstance(item, datetime):
            return to_utc_iso(item)
        raise TypeError(f"unsupported canonical value: {type(item).__name__}")

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def manifest_etag(revision: int, skill: Skill) -> str:
    """Build the opaque CAS token for one manifest revision."""

    return f'"skill-{revision}-{canonical_hash(manifest_state(skill))}"'
