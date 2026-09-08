"""Normalize connector metadata into the knowledge-work application profile.

This module is deliberately deterministic and metadata-only.  It recognizes a
small allowlist of fields commonly emitted by document, work-management,
source-control, messaging, and incident connectors.  It never mines free text
and never replaces values in a caller-authored ``metadata["knowledge_work"]``
profile.

The normalizer is a boundary adapter, not another ontology or identity store:
its output is the same profile consumed by ``EntityRelationService`` and entity
references are still resolved by the shared entity registry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...models.memory import MemoryScope, MemoryType, RememberInput
from ..entity_relation.metadata import KNOWLEDGE_WORK_METADATA_KEY

KNOWLEDGE_WORK_NORMALIZATION_KEY = "knowledge_work_normalization"
KNOWLEDGE_WORK_NORMALIZATION_VERSION = "1.0"

_EMPTY_IDENTITY_VALUES = frozenset({"", "none", "null", "n/a", "na", "unknown", "unassigned"})
_PROFILE_RELATION_FIELDS = frozenset(
    {
        "owner",
        "assignee",
        "author",
        "contributor",
        "reviewer",
        "approver",
        "decision_maker",
        "project",
        "organization",
        "part_of",
        "member_of",
        "depends_on",
        "blocks",
        "blocked_by",
        "references",
        "supersedes",
        "based_on",
        "affects",
        "supports",
        "supported_by",
        "contradicts",
        "about",
        "relations",
    }
)


@dataclass(frozen=True)
class KnowledgeWorkNormalization:
    """Result of one connector normalization pass."""

    metadata: dict[str, Any]
    profile: dict[str, Any] | None
    connector_type: str
    mapped_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FieldSpec:
    default_type: str
    paths: tuple[str, ...]


_ACTOR_FIELDS: dict[str, _FieldSpec] = {
    "owner": _FieldSpec("person", ("owner", "owners", "owned_by", "fields.owner", "github_owner")),
    "assignee": _FieldSpec(
        "person",
        ("assignee", "assignees", "assigned_to", "responsible", "fields.assignee", "fields.assignees", "github_assignees"),
    ),
    "author": _FieldSpec(
        "person",
        (
            "author",
            "authors",
            "creator",
            "created_by",
            "createdBy",
            "fields.creator",
            "fields.reporter",
            "reporter",
            "user",
            "sender",
            "from",
            "github_author",
            "slack_sender",
            "teams_sender",
        ),
    ),
    "contributor": _FieldSpec("person", ("contributor", "contributors", "collaborators", "fields.contributors")),
    "reviewer": _FieldSpec(
        "person",
        ("reviewer", "reviewers", "reviewed_by", "requested_reviewers", "fields.reviewers", "github_reviewers"),
    ),
    "approver": _FieldSpec(
        "person",
        ("approver", "approvers", "approved_by", "fields.approvers", "github_approvers"),
    ),
    "decision_maker": _FieldSpec(
        "person",
        ("decision_maker", "decision_makers", "decided_by", "fields.decision_maker"),
    ),
}

_TARGET_FIELDS: dict[str, _FieldSpec] = {
    "project": _FieldSpec(
        "project",
        ("project", "projects", "fields.project", "repository", "repo", "github_repo", "space", "workspace"),
    ),
    "organization": _FieldSpec("org", ("organization", "organizations", "org", "team", "fields.organization")),
    "part_of": _FieldSpec("work_item", ("part_of", "parent", "parent_item", "fields.parent")),
    "member_of": _FieldSpec("org", ("member_of", "membership", "fields.member_of")),
    "depends_on": _FieldSpec(
        "work_item",
        ("depends_on", "dependencies", "requires", "prerequisites", "fields.depends_on", "fields.dependencies"),
    ),
    "blocks": _FieldSpec("work_item", ("blocks", "fields.blocks")),
    "blocked_by": _FieldSpec("work_item", ("blocked_by", "blockers", "fields.blocked_by", "fields.blockers")),
    "references": _FieldSpec("artifact", ("references", "cites", "related_documents", "fields.references")),
    "supersedes": _FieldSpec(
        "artifact",
        ("supersedes", "replaces", "previous_version", "fields.supersedes", "fields.replaces"),
    ),
    "based_on": _FieldSpec("artifact", ("based_on", "basis", "source_documents", "fields.based_on")),
    "affects": _FieldSpec(
        "concept",
        ("affects", "affected_resources", "components", "component", "service", "services", "fields.affects", "fields.components"),
    ),
    "supports": _FieldSpec("concept", ("supports", "evidence_for", "fields.supports")),
    "supported_by": _FieldSpec("artifact", ("supported_by", "evidence", "fields.supported_by")),
    "contradicts": _FieldSpec("concept", ("contradicts", "conflicts_with", "fields.contradicts")),
    "about": _FieldSpec(
        "topic",
        ("about", "topics", "topic", "labels", "tags", "fields.labels", "github_labels"),
    ),
}

_QUALIFIER_PATHS: dict[str, tuple[str, ...]] = {
    "status": ("status", "state", "fields.status", "github_state"),
    "role": ("role", "fields.role"),
    "valid_from": ("valid_from", "start_at", "start_date", "fields.start_date"),
    "valid_to": ("valid_to", "due_at", "due_date", "end_date", "fields.due_date"),
    "source_created_at": ("created_at", "created", "createdDateTime", "fields.created"),
    "source_updated_at": ("updated_at", "updated", "modifiedTime", "fields.updated"),
}

_SUBJECT_NAME_PATHS = (
    "title",
    "summary",
    "name",
    "subject",
    "fields.summary",
    "fields.title",
    "github_title",
    "filename",
    "source_path",
)

_RECORD_ID_PATHS = (
    "record_id",
    "external_id",
    "key",
    "number",
    "id",
    "fields.key",
    "github_number",
    "gdrive_file_id",
    "slack_message_id",
    "teams_message_id",
    "message_id",
)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    normalized = _normalized_key(key)
    for candidate, value in mapping.items():
        if _normalized_key(str(candidate)) == normalized:
            return value
    return None


def _path_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = _mapping_value(value, component)
    return value


def _first_value(mapping: Mapping[str, Any], paths: Sequence[str]) -> tuple[Any, str | None]:
    roots: list[tuple[str, Mapping[str, Any]]] = [("", mapping)]
    connector_record = _mapping_value(mapping, "connector_record")
    if isinstance(connector_record, Mapping):
        roots.insert(0, ("connector_record.", connector_record))
    for prefix, root in roots:
        for path in paths:
            value = _path_value(root, path)
            if value is not None and value != [] and value != {}:
                return value, f"{prefix}{path}"
    return None, None


def _clean_text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text if text.casefold() not in _EMPTY_IDENTITY_VALUES else None


def _source_key(connector_type: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", connector_type.casefold()).strip("_") or "connector"


def _entity_ref(
    value: Any,
    *,
    default_type: str,
    connector_type: str,
    warnings: list[str],
    path: str,
) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        # Common APIs wrap identities as {"user": {...}} or {"account": {...}}.
        for wrapper in ("user", "account", "person", "actor"):
            nested = _mapping_value(value, wrapper)
            if isinstance(nested, Mapping):
                value = nested
                break

        name = None
        for key in (
            "name",
            "display_name",
            "displayName",
            "full_name",
            "title",
            "label",
            "username",
            "login",
            "emailAddress",
            "email",
            "mail",
            "value",
        ):
            name = _clean_text(_mapping_value(value, key))
            if name:
                break
        if not name:
            warnings.append(f"{path}: ignored entity object without a display name")
            return None
        entity_type = _clean_text(_mapping_value(value, "entity_type")) or _clean_text(_mapping_value(value, "type")) or default_type
        aliases: list[str] = []
        raw_aliases = _mapping_value(value, "aliases")
        if not isinstance(raw_aliases, (list, tuple, set)):
            raw_aliases = [raw_aliases] if raw_aliases is not None else []
        for alias in raw_aliases:
            cleaned = _clean_text(alias)
            if cleaned and cleaned != name and cleaned not in aliases:
                aliases.append(cleaned)
        for key in ("username", "login", "emailAddress", "email", "mail"):
            alias = _clean_text(_mapping_value(value, key))
            if alias and alias != name and alias not in aliases:
                aliases.append(alias)

        external_ids: dict[str, str] = {}
        raw_external_ids = _mapping_value(value, "external_ids")
        if isinstance(raw_external_ids, Mapping):
            external_ids.update(
                {
                    str(key): str(item)
                    for key, item in raw_external_ids.items()
                    if _clean_text(key) and _clean_text(item)
                }
            )
        for key in ("id", "accountId", "account_id", "user_id", "key"):
            external_id = _clean_text(_mapping_value(value, key))
            if external_id:
                external_ids.setdefault(_source_key(connector_type), external_id)
                break
        result: dict[str, Any] = {"name": name, "type": entity_type}
        if aliases:
            result["aliases"] = aliases
        if external_ids:
            result["external_ids"] = external_ids
        confidence = _mapping_value(value, "confidence")
        if confidence is not None:
            result["confidence"] = confidence
        return result

    name = _clean_text(value)
    if not name:
        if value is not None:
            warnings.append(f"{path}: ignored empty or sentinel entity value")
        return None
    return {"name": name, "type": default_type}


def _entity_refs(
    value: Any,
    *,
    default_type: str,
    connector_type: str,
    warnings: list[str],
    path: str,
) -> list[dict[str, Any]]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(values):
        ref = _entity_ref(
            item,
            default_type=default_type,
            connector_type=connector_type,
            warnings=warnings,
            path=f"{path}[{index}]",
        )
        if ref is None:
            continue
        key = (str(ref.get("type") or default_type).casefold(), str(ref["name"]).casefold())
        if key not in seen:
            seen.add(key)
            results.append(ref)
    return results


def _connector_type(metadata: Mapping[str, Any], explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().casefold()
    value, _ = _first_value(metadata, ("connector_type", "source_kind", "provider", "source"))
    return (_clean_text(value) or "generic").casefold()


def _record_kind(metadata: Mapping[str, Any], connector_type: str) -> str:
    value, _ = _first_value(metadata, ("record_type", "object_type", "item_type", "kind", "github_type", "type"))
    kind = (_clean_text(value) or connector_type).casefold().replace("-", "_").replace(" ", "_")
    if connector_type == "github":
        return {"pr": "code_review", "pull_request": "code_review", "readme": "document"}.get(kind, kind)
    return kind


def _subject_type(kind: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().casefold()
    if kind in {"issue", "task", "ticket", "incident", "change_request", "code_review", "pull_request", "pr"}:
        return "work_item"
    if kind in {"project", "program", "repository", "repo"}:
        return "project"
    if kind in {"decision", "architecture_decision", "adr"}:
        return "decision"
    if kind in {"message", "email", "thread", "document", "file", "page", "policy", "readme"}:
        return "artifact"
    return "artifact"


def _subject_ref(
    metadata: Mapping[str, Any],
    *,
    connector_type: str,
    kind: str,
    subject_name: str | None,
    subject_type: str | None,
    record_id: str | int | None,
) -> dict[str, Any] | None:
    raw_name = subject_name
    if not raw_name:
        value, _ = _first_value(metadata, _SUBJECT_NAME_PATHS)
        raw_name = _clean_text(value)
    name = _clean_text(raw_name)
    if not name:
        return None

    raw_id: Any = record_id
    if raw_id is None:
        raw_id, _ = _first_value(metadata, _RECORD_ID_PATHS)
    external_id = _clean_text(raw_id)
    aliases: list[str] = []
    if external_id and external_id != name:
        aliases.append(external_id)
    if connector_type == "github":
        repository, _ = _first_value(metadata, ("github_repo", "repository", "repo"))
        number, _ = _first_value(metadata, ("github_number", "number"))
        repo_text, number_text = _clean_text(repository), _clean_text(number)
        qualified = f"{repo_text}#{number_text}" if repo_text and number_text else None
        if qualified and qualified != name and qualified not in aliases:
            aliases.append(qualified)

    result: dict[str, Any] = {"name": name, "type": _subject_type(kind, subject_type)}
    if aliases:
        result["aliases"] = aliases
    if external_id:
        result["external_ids"] = {_source_key(connector_type): external_id}
    return result


def normalize_connector_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    connector_type: str | None = None,
    subject_name: str | None = None,
    subject_type: str | None = None,
    record_id: str | int | None = None,
) -> KnowledgeWorkNormalization:
    """Return metadata augmented with a deterministic knowledge-work profile.

    Normalization activates only when ``connector_type`` is supplied or the
    metadata declares ``connector_type``/``source_kind``.  Existing profile
    fields always win; mapped connector values only fill missing fields.
    """

    result_metadata = deepcopy(dict(metadata or {}))
    source = _connector_type(result_metadata, connector_type)
    if connector_type and "connector_type" not in result_metadata:
        result_metadata["connector_type"] = source

    explicit_present = KNOWLEDGE_WORK_METADATA_KEY in result_metadata
    explicit_profile = result_metadata.get(KNOWLEDGE_WORK_METADATA_KEY)
    if explicit_present and not isinstance(explicit_profile, dict):
        return KnowledgeWorkNormalization(
            metadata=result_metadata,
            profile=None,
            connector_type=source,
            warnings=("knowledge_work: explicit profile is not an object; normalization skipped",),
        )

    # Do not reinterpret arbitrary memory metadata.  Callers opt in through a
    # connector declaration or an explicit profile.
    connector_declared = bool(connector_type) or any(
        key in result_metadata for key in ("connector_type", "source_kind", "connector_record")
    )
    if not connector_declared and not explicit_present:
        return KnowledgeWorkNormalization(result_metadata, None, source)

    profile: dict[str, Any] = deepcopy(explicit_profile) if isinstance(explicit_profile, dict) else {}
    mapped: list[str] = []
    warnings: list[str] = []
    kind = _record_kind(result_metadata, source)
    subject = _subject_ref(
        result_metadata,
        connector_type=source,
        kind=kind,
        subject_name=subject_name,
        subject_type=subject_type,
        record_id=record_id,
    )
    if "subject" not in profile and subject is not None:
        profile["subject"] = subject
        mapped.append("subject")

    for field_name, spec in (*_ACTOR_FIELDS.items(), *_TARGET_FIELDS.items()):
        if field_name in profile:
            continue
        raw, path = _first_value(result_metadata, spec.paths)
        if path is None:
            continue
        refs = _entity_refs(
            raw,
            default_type=spec.default_type,
            connector_type=source,
            warnings=warnings,
            path=path,
        )
        if refs:
            profile[field_name] = refs[0] if len(refs) == 1 else refs
            mapped.append(field_name)

    for qualifier, paths in _QUALIFIER_PATHS.items():
        if qualifier in profile:
            continue
        value, path = _first_value(result_metadata, paths)
        if path is not None and value is not None:
            profile[qualifier] = value.isoformat() if isinstance(value, datetime) else value
            mapped.append(qualifier)

    # Messaging sources have a useful relation not represented by a shorthand.
    # Keep it as ordinary application-profile data so recipients are resolved
    # through the same registry rather than a source-specific identity table.
    if source in {"email", "mail"} and "relations" not in profile:
        sender, sender_path = _first_value(result_metadata, ("sender", "from"))
        recipients, recipients_path = _first_value(result_metadata, ("to", "recipients"))
        sender_refs = _entity_refs(
            sender,
            default_type="person",
            connector_type=source,
            warnings=warnings,
            path=sender_path or "sender",
        )
        recipient_refs = _entity_refs(
            recipients,
            default_type="person",
            connector_type=source,
            warnings=warnings,
            path=recipients_path or "recipients",
        )
        if sender_refs and recipient_refs:
            profile["relations"] = [
                {"source": sender_refs[0], "relationship": "sent_to", "target": recipient}
                for recipient in recipient_refs
                if recipient["name"].casefold() != sender_refs[0]["name"].casefold()
            ]
            if profile["relations"]:
                mapped.append("relations")

    has_relation = any(field in profile for field in _PROFILE_RELATION_FIELDS)
    if "subject" not in profile and has_relation:
        warnings.append("knowledge_work: relation fields were ignored because no subject could be identified")
    should_emit = explicit_present or ("subject" in profile and has_relation)
    if not should_emit:
        return KnowledgeWorkNormalization(
            metadata=result_metadata,
            profile=None,
            connector_type=source,
            mapped_fields=tuple(mapped),
            warnings=tuple(warnings),
        )

    result_metadata[KNOWLEDGE_WORK_METADATA_KEY] = profile
    if mapped:
        result_metadata.setdefault(
            KNOWLEDGE_WORK_NORMALIZATION_KEY,
            {
                "version": KNOWLEDGE_WORK_NORMALIZATION_VERSION,
                "connector_type": source,
                "record_kind": kind,
                "mapped_fields": mapped,
                **({"warnings": warnings} if warnings else {}),
            },
        )
    return KnowledgeWorkNormalization(
        metadata=result_metadata,
        profile=profile,
        connector_type=source,
        mapped_fields=tuple(mapped),
        warnings=tuple(warnings),
    )


def connector_to_remember_input(
    *,
    connector_type: str,
    content: str,
    source_metadata: Mapping[str, Any] | None = None,
    subject_name: str | None = None,
    subject_type: str | None = None,
    record_id: str | int | None = None,
    timestamp: datetime | None = None,
    importance: float = 0.5,
    context_id: str | None = None,
    observer_id: str | None = None,
    logical_key: str | None = None,
    scope: MemoryScope | None = None,
) -> RememberInput:
    """Build a ``RememberInput`` for a connector-shaped professional record."""

    if not connector_type or not connector_type.strip():
        raise ValueError("connector_to_remember_input requires a connector_type")
    if not content or not content.strip():
        raise ValueError("connector_to_remember_input requires non-empty content")
    normalized = normalize_connector_metadata(
        source_metadata,
        connector_type=connector_type,
        subject_name=subject_name,
        subject_type=subject_type,
        record_id=record_id,
    )
    metadata = normalized.metadata
    metadata.setdefault("source", connector_type.strip().casefold())
    if record_id is not None:
        metadata.setdefault("source_record_id", str(record_id))
    return RememberInput(
        content=content.strip(),
        type=MemoryType.SEMANTIC,
        importance=importance,
        metadata=metadata,
        context_id=context_id,
        observer_id=observer_id,
        logical_key=logical_key,
        event_time=timestamp,
        scope=scope,
    )


__all__ = (
    "KNOWLEDGE_WORK_NORMALIZATION_KEY",
    "KNOWLEDGE_WORK_NORMALIZATION_VERSION",
    "KnowledgeWorkNormalization",
    "connector_to_remember_input",
    "normalize_connector_metadata",
)
