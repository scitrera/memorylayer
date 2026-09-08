"""Deterministic acquisition of knowledge-work relations from source metadata.

Connectors usually know more than the text they deliver: a task payload has an
assignee and project, a document has authors and a parent space, and an approval
record has a reviewer.  ``metadata["knowledge_work"]`` is the small application
profile that preserves those authoritative facts and maps them onto the shared
entity-relation store without an LLM.

The profile deliberately describes *assertions*, not a second graph.  The full
profile remains on the source memory (where role, dates, external ids, and other
qualifiers are inspectable); each emitted edge is evidence-linked back to that
memory and records the exact metadata path used to acquire it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

KNOWLEDGE_WORK_METADATA_KEY = "knowledge_work"

_ENTITY_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class MetadataEntityRef:
    """One typed entity reference in the knowledge-work metadata profile."""

    name: str | None = None
    entity_id: str | None = None
    entity_type: str = "concept"
    aliases: tuple[str, ...] = ()
    confidence: float = 1.0
    external_ids: dict[str, str] = field(default_factory=dict, compare=False, hash=False)
    metadata_path: str = field(default="", compare=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity_id or "", f"{self.entity_type}:{(self.name or '').casefold()}")


@dataclass(frozen=True)
class MetadataRelationCandidate:
    """A normalized relation assertion awaiting entity resolution."""

    source: MetadataEntityRef
    target: MetadataEntityRef
    relationship: str
    metadata_path: str


@dataclass
class MetadataRelationExtraction:
    candidates: list[MetadataRelationCandidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Actor fields use active-voice canonical edges.  This keeps the graph compact
# while the relation service accepts the passive/inverse spellings at its API.
_ACTOR_FIELDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("owners", "owner", "owned_by"), "owns", "person"),
    (("assignees", "assignee", "assigned_to", "responsible"), "responsible_for", "person"),
    (("authors", "author", "creators", "creator", "created_by"), "authored", "person"),
    (("contributors", "contributor"), "contributed_to", "person"),
    (("reviewers", "reviewer", "reviewed_by"), "reviewed", "person"),
    (("approvers", "approver", "approved_by"), "approved", "person"),
    (("decision_makers", "decision_maker", "decided_by"), "decided", "person"),
)

# These fields point outward from the profile subject.
_SUBJECT_FIELDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("projects", "project"), "part_of", "project"),
    (("organizations", "organization"), "part_of", "org"),
    (("part_of",), "part_of", "concept"),
    (("member_of",), "member_of", "org"),
    (("depends_on", "dependencies"), "depends_on", "work_item"),
    (("blocks",), "blocks", "work_item"),
    (("references",), "references", "artifact"),
    (("supersedes", "replaces"), "supersedes", "artifact"),
    (("based_on", "basis"), "based_on", "artifact"),
    (("affects",), "affects", "concept"),
    (("supports",), "supports", "concept"),
    (("contradicts",), "contradicts", "concept"),
    (("about", "topics"), "about", "topic"),
)

# Passive fields are normalized by reversing their endpoints.
_REVERSE_SUBJECT_FIELDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("blocked_by",), "blocks", "work_item"),
    (("superseded_by", "replaced_by"), "supersedes", "artifact"),
    (("supported_by",), "supports", "concept"),
    (("affected_by",), "affects", "concept"),
    (("referenced_by",), "references", "artifact"),
    (("has_parts", "has_part"), "part_of", "concept"),
)


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _entity_ref(value: Any, *, default_type: str, path: str, errors: list[str]) -> MetadataEntityRef | None:
    if isinstance(value, str):
        name = value.strip()
        if not name:
            errors.append(f"{path}: entity name must not be empty")
            return None
        entity_type = re.sub(r"[^a-z0-9_]+", "_", str(default_type).strip().casefold()).strip("_")
        if not _ENTITY_TYPE_RE.fullmatch(entity_type):
            errors.append(f"{path}: invalid entity type")
            return None
        return MetadataEntityRef(name=name, entity_type=entity_type, metadata_path=path)
    if not isinstance(value, dict):
        errors.append(f"{path}: entity must be a string or object")
        return None

    raw_name = value.get("name", value.get("title", value.get("display_name")))
    name = str(raw_name).strip() if raw_name is not None else None
    raw_id = value.get("entity_id")
    entity_id = str(raw_id).strip() if raw_id is not None else None
    if not name and not entity_id:
        errors.append(f"{path}: entity requires name or entity_id")
        return None

    entity_type = str(value.get("type") or value.get("entity_type") or default_type).strip().casefold()
    entity_type = re.sub(r"[^a-z0-9_]+", "_", entity_type).strip("_")
    if not _ENTITY_TYPE_RE.fullmatch(entity_type):
        errors.append(f"{path}: invalid entity type")
        return None

    aliases = tuple(
        dict.fromkeys(
            str(alias).strip()
            for alias in _values(value.get("aliases"))
            if alias is not None and str(alias).strip()
        )
    )
    try:
        confidence = float(value.get("confidence", 1.0))
    except (TypeError, ValueError):
        errors.append(f"{path}: confidence must be numeric")
        return None
    if not 0.0 <= confidence <= 1.0:
        errors.append(f"{path}: confidence must be between 0 and 1")
        return None
    raw_external = value.get("external_ids") or {}
    if not isinstance(raw_external, dict):
        errors.append(f"{path}: external_ids must be an object")
        return None
    external_ids = {
        str(key): str(item)
        for key, item in raw_external.items()
        if key is not None and item is not None
    }
    return MetadataEntityRef(
        name=name,
        entity_id=entity_id,
        entity_type=entity_type,
        aliases=aliases,
        confidence=confidence,
        external_ids=external_ids,
        metadata_path=path,
    )


def _field_refs(
    profile: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    default_type: str,
    errors: list[str],
) -> list[tuple[MetadataEntityRef, str]]:
    refs: list[tuple[MetadataEntityRef, str]] = []
    for field_name in aliases:
        if field_name not in profile:
            continue
        for index, value in enumerate(_values(profile[field_name])):
            path = f"{KNOWLEDGE_WORK_METADATA_KEY}.{field_name}[{index}]"
            ref = _entity_ref(value, default_type=default_type, path=path, errors=errors)
            if ref is not None:
                refs.append((ref, path))
    return refs


def extract_metadata_relations(metadata: dict[str, Any] | None) -> MetadataRelationExtraction:
    """Normalize a reserved knowledge-work metadata profile into relation candidates.

    Missing metadata is a no-op.  Malformed profile entries are reported per
    entry so one bad connector field cannot discard otherwise valid assertions.
    """

    result = MetadataRelationExtraction()
    if not metadata or KNOWLEDGE_WORK_METADATA_KEY not in metadata:
        return result
    profile = metadata.get(KNOWLEDGE_WORK_METADATA_KEY)
    if not isinstance(profile, dict):
        result.errors.append(f"{KNOWLEDGE_WORK_METADATA_KEY}: profile must be an object")
        return result

    subject = _entity_ref(
        profile.get("subject"),
        default_type=str(profile.get("subject_type") or "concept"),
        path=f"{KNOWLEDGE_WORK_METADATA_KEY}.subject",
        errors=result.errors,
    )
    if subject is None:
        return result

    for aliases, relationship, default_type in _ACTOR_FIELDS:
        for actor, path in _field_refs(profile, aliases, default_type=default_type, errors=result.errors):
            result.candidates.append(MetadataRelationCandidate(actor, subject, relationship, path))

    for aliases, relationship, default_type in _SUBJECT_FIELDS:
        for target, path in _field_refs(profile, aliases, default_type=default_type, errors=result.errors):
            result.candidates.append(MetadataRelationCandidate(subject, target, relationship, path))

    for aliases, relationship, default_type in _REVERSE_SUBJECT_FIELDS:
        for source, path in _field_refs(profile, aliases, default_type=default_type, errors=result.errors):
            result.candidates.append(MetadataRelationCandidate(source, subject, relationship, path))

    # Escape hatch for connector-native fields that do not yet have a shorthand.
    # ``"$subject"`` keeps the common endpoint concise while still requiring a
    # registered relation type at the relation-service boundary.
    for index, value in enumerate(_values(profile.get("relations"))):
        path = f"{KNOWLEDGE_WORK_METADATA_KEY}.relations[{index}]"
        if not isinstance(value, dict):
            result.errors.append(f"{path}: relation must be an object")
            continue
        relationship = re.sub(
            r"[^a-z0-9_]+",
            "_",
            str(value.get("relationship") or "").strip().casefold(),
        ).strip("_")
        if not relationship:
            result.errors.append(f"{path}: relationship is required")
            continue

        def endpoint(field_name: str) -> MetadataEntityRef | None:
            raw = value.get(field_name)
            if raw == "$subject":
                return subject
            return _entity_ref(
                raw,
                default_type="concept",
                path=f"{path}.{field_name}",
                errors=result.errors,
            )

        source = endpoint("source")
        target = endpoint("target")
        if source is not None and target is not None:
            result.candidates.append(MetadataRelationCandidate(source, target, relationship, path))

    # Connectors occasionally supply synonymous fields together.  Preserve the
    # first source path deterministically and avoid duplicate evidence attempts.
    deduplicated: list[MetadataRelationCandidate] = []
    seen: set[tuple[tuple[str, str], tuple[str, str], str]] = set()
    for candidate in result.candidates:
        key = (candidate.source.key, candidate.target.key, candidate.relationship)
        if key not in seen:
            seen.add(key)
            deduplicated.append(candidate)
    result.candidates = deduplicated
    return result


__all__ = (
    "KNOWLEDGE_WORK_METADATA_KEY",
    "MetadataEntityRef",
    "MetadataRelationCandidate",
    "MetadataRelationExtraction",
    "extract_metadata_relations",
)
