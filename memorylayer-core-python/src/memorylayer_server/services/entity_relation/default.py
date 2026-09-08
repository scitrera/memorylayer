"""Deterministic entity-relation resolution, evidence writes, and traversal."""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass

from ...models.entity_relation import (
    EntityRelation,
    EntityRelationEvidence,
    EntityRelationInput,
    EntityRelationPath,
    EntityRelationWriteResult,
    RelationEvidenceKind,
)
from ...models.memory import Memory
from ...utils import generate_id
from ..entity_registry._normalize import normalize_entity_name
from ..memory.relation_intent import classify_relation_intent
from ..ontology.base import BASE_ONTOLOGY
from ..storage import StorageBackend, StorageCapabilityError
from .metadata import KNOWLEDGE_WORK_METADATA_KEY, MetadataEntityRef, extract_metadata_relations

_STRUCTURAL_RELATIONSHIPS: dict[str, tuple[str, bool]] = {
    "employed_by": ("employed_by", False),
    "employs": ("employed_by", True),
    "invested_in": ("invested_in", False),
    "investor_of": ("invested_in", True),
    "advises": ("advises", False),
    "advised_by": ("advises", True),
    "attended": ("attended", False),
    "attended_by": ("attended", True),
    "founded": ("founded", False),
    "founded_by": ("founded", True),
    "owns": ("owns", False),
    "owned_by": ("owns", True),
    "responsible_for": ("responsible_for", False),
    "assigned_to": ("responsible_for", True),
    "responsibility_of": ("responsible_for", True),
    "authored": ("authored", False),
    "author_of": ("authored", False),
    "authored_by": ("authored", True),
    "contributed_to": ("contributed_to", False),
    "has_contributor": ("contributed_to", True),
    "reviewed": ("reviewed", False),
    "reviewed_by": ("reviewed", True),
    "approved": ("approved", False),
    "approved_by": ("approved", True),
    "decided": ("decided", False),
    "decided_by": ("decided", True),
    "member_of": ("member_of", False),
    "has_member": ("member_of", True),
    "part_of": ("part_of", False),
    "has_part": ("part_of", True),
    "parent_of": ("parent_of", False),
    "child_of": ("parent_of", True),
    "sibling_of": ("sibling_of", False),
    "located_in": ("located_in", False),
    "location_of": ("located_in", True),
    "depends_on": ("depends_on", False),
    "dependency_of": ("depends_on", True),
    "depended_on_by": ("depends_on", True),
    "blocks": ("blocks", False),
    "blocked_by": ("blocks", True),
    "references": ("references", False),
    "referenced_by": ("references", True),
    "supersedes": ("supersedes", False),
    "superseded_by": ("supersedes", True),
    "replaces": ("supersedes", False),
    "replaced_by": ("supersedes", True),
    "based_on": ("based_on", False),
    "basis_for": ("based_on", True),
    "affects": ("affects", False),
    "affected_by": ("affects", True),
    "supports": ("supports", False),
    "supported_by": ("supports", True),
    "about": ("about", False),
    "subject_of": ("about", True),
    "sent_to": ("sent_to", False),
    "received_from": ("sent_to", True),
}
for _relationship, _metadata in BASE_ONTOLOGY.items():
    _STRUCTURAL_RELATIONSHIPS.setdefault(_relationship, (_relationship, False))

_NAMED_ENTITY = r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3})"
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\b{_NAMED_ENTITY}\s+works? (?:at|for)\s+{_NAMED_ENTITY}\b"), "employed_by"),
    (re.compile(rf"\b{_NAMED_ENTITY}\s+(?:is|serves as)\s+an?\s+advis(?:or|er)\s+(?:at|to)\s+{_NAMED_ENTITY}\b"), "advises"),
    (re.compile(rf"\b{_NAMED_ENTITY}\s+invested in\s+{_NAMED_ENTITY}\b"), "invested_in"),
    (re.compile(rf"\b{_NAMED_ENTITY}\s+founded\s+{_NAMED_ENTITY}\b"), "founded"),
    (re.compile(rf"\b{_NAMED_ENTITY}\s+attended\s+{_NAMED_ENTITY}\b"), "attended"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+owns\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3})\b"), "owns"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+is (?:a )?member of\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3})\b"), "member_of"),
    (re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})\s+depends on\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3})\b"), "depends_on"),
)


@dataclass
class RelationRecallResult:
    memories: list[Memory]
    paths: list[EntityRelationPath]
    unresolved_seed: bool = False


class EntityRelationService:
    def __init__(self, storage: StorageBackend, metrics=None, entity_registry=None):
        self.storage = storage
        self.metrics = metrics
        self.entity_registry = entity_registry

    def _record_write_metrics(self, result: EntityRelationWriteResult) -> None:
        if self.metrics is None:
            return
        try:
            for outcome, value in (
                ("resolved", result.resolved),
                ("unresolved", result.unresolved),
                ("rejected", result.rejected),
                ("duplicate", result.duplicate),
            ):
                if value:
                    self.metrics.counter(
                        "memorylayer_entity_relation_inputs_total",
                        value,
                        labels={"outcome": outcome},
                    )
            if result.resolved:
                self.metrics.counter(
                    "memorylayer_entity_relation_evidence_writes_total",
                    result.resolved,
                    labels={"outcome": "stored"},
                )
            if result.duplicate:
                self.metrics.counter(
                    "memorylayer_entity_relation_evidence_writes_total",
                    result.duplicate,
                    labels={"outcome": "duplicate"},
                )
        except Exception:
            return

    async def _resolve_endpoint(
        self,
        workspace_id: str,
        entity_id: str | None,
        entity_name: str | None,
    ) -> str | None:
        if entity_id:
            entity = await self.storage.get_entity(workspace_id, entity_id)
            if entity and entity.get("status") == "active":
                return entity_id
            return None
        normalized = normalize_entity_name(entity_name or "")
        if not normalized:
            return None
        exact = await self.storage.find_entities_by_normalized_name_any_type(
            workspace_id,
            normalized,
        )
        if exact:
            return exact[0]["id"]
        aliases = await self.storage.find_entities_by_normalized_alias(
            workspace_id,
            normalized,
        )
        return aliases[0]["id"] if aliases else None

    @staticmethod
    def _canonicalize(value: EntityRelationInput, source_id: str, target_id: str) -> tuple[str, str, str]:
        relationship = re.sub(r"[^a-z0-9_]+", "_", value.relationship.strip().casefold()).strip("_")
        if relationship not in _STRUCTURAL_RELATIONSHIPS:
            raise ValueError(f"unregistered relationship type: {relationship}")
        canonical, reverse = _STRUCTURAL_RELATIONSHIPS[relationship]
        if reverse:
            source_id, target_id = target_id, source_id
        if canonical == "sibling_of" and target_id < source_id:
            source_id, target_id = target_id, source_id
        if source_id == target_id:
            raise ValueError("self-relations are not allowed")
        return source_id, target_id, canonical

    @staticmethod
    def pattern_relations(content: str) -> list[EntityRelationInput]:
        results: list[EntityRelationInput] = []
        for pattern, relationship in _PATTERNS:
            for match in pattern.finditer(content):
                results.append(
                    EntityRelationInput(
                        source_entity_name=match.group(1),
                        target_entity_name=match.group(2),
                        relationship=relationship,
                        confidence=0.95,
                        source_span_start=match.start(),
                        source_span_end=match.end(),
                        evidence_kind=RelationEvidenceKind.PATTERN,
                        extraction_method=f"pattern:{relationship}",
                    )
                )
        return results

    async def _materialize_metadata_relations(
        self,
        memory: Memory,
    ) -> tuple[list[EntityRelationInput], list[str], list[str]]:
        """Resolve the reserved metadata profile through the canonical registry.

        Returns ``(inputs, malformed_errors, resolution_errors)``.  Entity
        creation is intentionally limited to typed, authoritative metadata; raw
        text patterns retain their conservative resolve-only behavior.
        """

        extraction = extract_metadata_relations(memory.metadata)
        if not extraction.candidates:
            return [], extraction.errors, []

        entity_ids: dict[tuple[str, str], str | None] = {}
        resolution_errors: list[str] = []

        async def resolve(ref: MetadataEntityRef) -> str | None:
            metadata_path = ref.metadata_path or KNOWLEDGE_WORK_METADATA_KEY
            if ref.key in entity_ids:
                return entity_ids[ref.key]
            if ref.entity_id:
                entity_ids[ref.key] = ref.entity_id
                return ref.entity_id
            if not ref.name:
                entity_ids[ref.key] = None
                return None
            if self.entity_registry is None:
                resolved = await self._resolve_endpoint(memory.workspace_id, None, ref.name)
                entity_ids[ref.key] = resolved
                if resolved is None:
                    resolution_errors.append(f"{metadata_path}: entity registry unavailable for {ref.name!r}")
                return resolved
            try:
                entity = await self.entity_registry.upsert(
                    memory.workspace_id,
                    ref.name,
                    ref.entity_type,
                    aliases=list(ref.aliases) or None,
                    confidence=ref.confidence,
                    provenance={
                        "source": "knowledge_work_metadata",
                        "source_memory_id": memory.id,
                        "metadata_path": metadata_path,
                        **({"external_ids": ref.external_ids} if ref.external_ids else {}),
                    },
                    representative_memory_id=memory.id,
                )
            except Exception as exc:  # one invalid connector entity must not discard the memory
                resolution_errors.append(f"{metadata_path}: could not resolve {ref.name!r}: {exc}")
                entity_ids[ref.key] = None
                return None
            entity_ids[ref.key] = entity.id
            return entity.id

        inputs: list[EntityRelationInput] = []
        for candidate in extraction.candidates:
            source_id = await resolve(candidate.source)
            target_id = await resolve(candidate.target)
            if source_id is None or target_id is None:
                continue
            inputs.append(
                EntityRelationInput(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship=candidate.relationship,
                    confidence=min(candidate.source.confidence, candidate.target.confidence),
                    evidence_kind=RelationEvidenceKind.ADAPTER,
                    extraction_method=f"metadata:{candidate.metadata_path}"[:128],
                )
            )
        return inputs, extraction.errors, resolution_errors

    async def write_for_memory(
        self,
        memory: Memory,
        inputs: list[EntityRelationInput],
        *,
        include_patterns: bool = True,
        include_metadata: bool = True,
    ) -> EntityRelationWriteResult:
        capability_check = getattr(self.storage, "supports_capability", None)
        supported = bool(
            capability_check and not inspect.iscoroutinefunction(capability_check) and capability_check("entity_relations") is True
        )
        if inputs and not supported:
            raise StorageCapabilityError("entity_relations")
        if not supported:
            return EntityRelationWriteResult()
        values = list(inputs)
        malformed_errors: list[str] = []
        resolution_errors: list[str] = []
        if include_metadata:
            metadata_values, malformed_errors, resolution_errors = await self._materialize_metadata_relations(memory)
            values.extend(metadata_values)
        if include_patterns:
            values.extend(self.pattern_relations(memory.content))
        result = EntityRelationWriteResult(
            rejected=len(malformed_errors),
            unresolved=len(resolution_errors),
            errors=[*malformed_errors, *resolution_errors],
        )
        for value in values:
            source_id = await self._resolve_endpoint(
                memory.workspace_id,
                value.source_entity_id,
                value.source_entity_name,
            )
            target_id = await self._resolve_endpoint(
                memory.workspace_id,
                value.target_entity_id,
                value.target_entity_name,
            )
            if not source_id or not target_id:
                result.unresolved += 1
                result.errors.append(f"unresolved endpoints for {value.relationship}")
                continue
            try:
                source_id, target_id, relationship = self._canonicalize(value, source_id, target_id)
            except ValueError as exc:
                result.rejected += 1
                result.errors.append(str(exc))
                continue
            start = value.source_span_start
            end = value.source_span_end
            if start is not None and (end is None or end > len(memory.content)):
                result.rejected += 1
                result.errors.append(f"invalid source span for {relationship}")
                continue
            excerpt = memory.content[start:end] if start is not None and end is not None else memory.content
            relation = EntityRelation(
                id=generate_id("erel"),
                workspace_id=memory.workspace_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship=relationship,
                confidence=value.confidence,
            )
            evidence = EntityRelationEvidence(
                id=generate_id("eev"),
                workspace_id=memory.workspace_id,
                relation_id=relation.id,
                source_memory_id=memory.id,
                evidence_kind=value.evidence_kind,
                source_span_start=start,
                source_span_end=end,
                excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
                confidence=value.confidence,
                extraction_method=value.extraction_method,
            )
            stored, duplicate = await self.storage.upsert_entity_relation(relation, evidence)
            if duplicate:
                result.duplicate += 1
            else:
                result.resolved += 1
            if stored.id not in result.relation_ids:
                result.relation_ids.append(stored.id)
        self._record_write_metrics(result)
        return result

    async def recall(
        self,
        workspace_id: str,
        query: str,
        *,
        max_edges: int = 40,
        max_memories: int = 20,
    ) -> RelationRecallResult:
        capability_check = getattr(self.storage, "supports_capability", None)
        supported = bool(
            capability_check
            and not inspect.iscoroutinefunction(capability_check)
            and capability_check("entity_relations") is True
        )
        if not supported:
            return RelationRecallResult([], [])
        intent = classify_relation_intent(query)
        if not intent.is_relation_query or intent.confidence < 0.65:
            return RelationRecallResult([], [])
        seeds: list[str] = []
        unresolved = False
        for phrase in intent.seed_phrases[:4]:
            resolved = await self._resolve_endpoint(workspace_id, None, phrase)
            if resolved:
                seeds.append(resolved)
            else:
                unresolved = True
        if not seeds:
            return RelationRecallResult([], [], unresolved_seed=True)
        compatible: list[str] | None = []
        for relationship in intent.relationship_types:
            canonical = _STRUCTURAL_RELATIONSHIPS.get(relationship)
            if canonical and canonical[0] not in compatible:
                compatible.append(canonical[0])
        paths = await self.storage.traverse_entity_relations(
            workspace_id,
            seeds,
            relationships=compatible or None,
            direction=intent.direction,
            max_hops=intent.max_hops,
            max_edges=max_edges,
        )
        memory_ids = list(dict.fromkeys(memory_id for path in paths for memory_id in path.evidence_memory_ids))[:max_memories]
        memories = await self.storage.get_memories_by_ids(workspace_id, memory_ids)
        by_id = {memory.id: memory for memory in memories}
        ordered = [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]
        for memory in ordered:
            memory.match_signals = sorted(set(memory.match_signals or []) | {"relational"})
        return RelationRecallResult(ordered, paths, unresolved_seed=unresolved)
