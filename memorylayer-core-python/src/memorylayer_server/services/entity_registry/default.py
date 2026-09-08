"""Default (relational) Entity Registry Service implementation (slice 1).

``DefaultEntityRegistryService`` resolves surface names to canonical entity
nodes directly over the relational storage primitives — no embeddings, no LLM,
no graph database. It is the OSS default and the PARITY GOLDEN that the
enterprise PostgreSQL backend is conformance-tested against: both backends MUST
return identical normalized DTOs.

Resolution (``resolve``) is deterministic:
  normalize(name) -> exact match (workspace, type, normalized_name)
                  -> alias match (normalized alias)
                  -> create (only when allow_create)

All persistence goes through the ``StorageBackend`` entity methods so SQLite,
in-memory, and PostgreSQL share the same control flow. Member accretion
(``add_member``) and ``merge`` (member + alias reassignment, alias
carry-forward, source tombstoning) are likewise pure relational operations.
"""

import logging

from scitrera_app_framework import Variables, get_extension, get_logger

from ...models.entity_registry import Entity, EntityResolution, EntityType, Member, RelatedEntity
from ...utils import utc_now_iso
from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import EntityRegistryServicePluginBase
from ._normalize import normalize_entity_name
from .base import EntityRegistryService


def _etype_str(entity_type: "EntityType | str") -> str:
    """Coerce an EntityType enum or bare string to the storage str value.

    Centralised so ``resolve``, ``upsert``, and ``find_by_alias`` cannot drift.
    """
    return entity_type.value if isinstance(entity_type, EntityType) else str(entity_type)


def _entity_from_dict(d: dict) -> Entity:
    """Map a storage entity dict to the ``Entity`` DTO."""
    return Entity(
        id=d["id"],
        workspace_id=d["workspace_id"],
        # Kept as the stored string (may be a core EntityType value OR an ontology
        # domain type like "equipment"); NOT coerced to EntityType, which would
        # raise for domain types.
        entity_type=d["entity_type"],
        canonical_name=d["canonical_name"],
        normalized_name=d["normalized_name"],
        aliases=list(d.get("aliases") or []),
        confidence=d.get("confidence", 1.0),
        provenance=dict(d.get("provenance") or {}),
        representative_memory_id=d.get("representative_memory_id"),
        status=d.get("status", "active"),
        merged_into=d.get("merged_into"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


class DefaultEntityRegistryService(EntityRegistryService):
    """Entity registry over relational storage primitives (OSS default)."""

    PROVIDER_NAME = "default"

    def __init__(self, storage: StorageBackend, v: Variables):
        self._storage = storage
        self.logger = get_logger(v, name="EntityRegistryService")

    async def resolve(
        self,
        workspace_id: str,
        name: str,
        entity_type: EntityType,
        *,
        source_memory_id: str | None = None,
        observer_id: str | None = None,
        allow_create: bool = True,
        promote: bool = False,
    ) -> EntityResolution:
        etype = _etype_str(entity_type)
        normalized = normalize_entity_name(name)

        # 0. Name-first resolution (accretion path only; default OFF).
        #
        # Default-typed resolution keys on (workspace, type, normalized_name),
        # which fragments a single real-world person who is BOTH a speaker
        # (PERSON, role="self") and a third-party mention (CONCEPT, role="mention")
        # into two registry nodes — so get_representation(observer, subject) with
        # the default subject_type=PERSON queries the empty PERSON node. When the
        # caller opts into ``promote`` (accretion does), a given normalized name
        # maps to ONE canonical entity within the workspace, regardless of type:
        #   * an existing entity of the requested type → use it;
        #   * an existing CONCEPT when PERSON is requested → PROMOTE it to PERSON
        #     (keep id/members/aliases) so prior mentions co-locate with self
        #     turns. If a PERSON node ALSO already exists, merge the CONCEPT INTO
        #     the PERSON (reassign members/aliases, tombstone) — no duplicate;
        #   * an existing PERSON when CONCEPT is requested → use the PERSON (a
        #     mention of a known person IS that person).
        # Tradeoff: name-first within a workspace cannot distinguish "Apple" the
        # org from "Apple" the fruit — the enterprise GLiNER2/fuzzy/LLM tier
        # handles that; this deterministic OSS path optimizes for the
        # dialogue/person-perspective case (same name as speaker AND mention = the
        # same person — safe to unify). Only PERSON⟷CONCEPT for the SAME
        # normalized name is unified; two different PERSON entities are never
        # merged here.
        if promote:
            promoted = await self._resolve_name_first(
                workspace_id, name, normalized, etype, source_memory_id, observer_id
            )
            if promoted is not None:
                return promoted

        # 1. Exact match on (workspace, type, normalized_name).
        exact = await self._storage.find_entity_by_normalized_name(workspace_id, etype, normalized)
        if exact is not None:
            return EntityResolution(entity=_entity_from_dict(exact), matched_via="exact", score=1.0)

        # 2. Alias match on the normalized alias (restricted to the same type).
        # When multiple entities share the alias (unusual but possible if aliases
        # were added independently), we take the first by entity id (deterministic
        # tie-break; documented in base.py). Log at DEBUG so operators can spot
        # ambiguous aliases and merge them.
        alias_hits = await self._storage.find_entities_by_normalized_alias(
            workspace_id, normalized, entity_type=etype
        )
        if alias_hits:
            if len(alias_hits) > 1:
                self.logger.debug(
                    "Ambiguous alias %r matched %d entities in workspace %s (type=%s) — "
                    "using lowest entity id; consider merging duplicates",
                    name, len(alias_hits), workspace_id, etype,
                )
            entity = _entity_from_dict(alias_hits[0])
            return EntityResolution(entity=entity, matched_via="alias", score=entity.confidence)

        # 3. Create (only when allowed).
        if not allow_create:
            raise LookupError(
                f"No {etype} entity matching {name!r} in workspace {workspace_id} (allow_create=False)"
            )

        from .provenance import build_provenance

        _now = utc_now_iso()
        provenance = build_provenance(
            "entity.create",
            generated_at=_now,
            matched_via="created",
            created_at=_now,  # back-compat key (existing readers)
            source_name=name,
            source_memory_id=source_memory_id,
            observer_id=observer_id,
        )

        stored = await self._storage.store_entity(
            {
                "workspace_id": workspace_id,
                "entity_type": etype,
                "canonical_name": name,
                "normalized_name": normalized,
                "confidence": 1.0,
                "provenance": provenance,
                "representative_memory_id": source_memory_id,
                "status": "active",
            }
        )
        # Race tolerance: ``store_entity`` handles the partial-unique-index
        # collision and returns the winner's row when two concurrent callers
        # both missed the exact-match check. Detect this by comparing the
        # stored normalized_name with what we tried to create — if it matches
        # but the entity id already existed before our call, treat it as an
        # "exact" hit so the caller can still ``add_member`` safely.
        # In practice the only difference from "created" is the matched_via
        # label; the entity is valid either way.
        if stored["normalized_name"] == normalized and stored.get("provenance", {}).get("matched_via") != "created":
            # The winner had different provenance — we lost the race.
            matched_via = "exact"
        else:
            matched_via = "created"
        return EntityResolution(entity=_entity_from_dict(stored), matched_via=matched_via, score=1.0)

    async def _resolve_name_first(
        self,
        workspace_id: str,
        name: str,
        normalized: str,
        etype: str,
        source_memory_id: str | None,
        observer_id: str | None,
    ) -> EntityResolution | None:
        """Name-first (type-agnostic) resolution with PERSON promotion.

        Returns an ``EntityResolution`` when an existing same-name entity is
        reused/promoted/merged, or ``None`` to fall through to the standard
        typed exact/alias/create path (no existing entity, or nothing to unify).

        Unification is scoped to PERSON⟷CONCEPT for the SAME normalized name:
          * existing entity of the requested type → reuse it ("exact");
          * requested PERSON, existing is CONCEPT → PROMOTE the CONCEPT to PERSON
            in place (keep id/members/aliases). If a PERSON node also exists,
            merge the CONCEPT INTO that PERSON instead of promoting (avoids a
            duplicate), returning the surviving PERSON;
          * requested CONCEPT, existing is PERSON → reuse the PERSON (a mention of
            a known person is that person).
        """
        candidates = await self._storage.find_entities_by_normalized_name_any_type(
            workspace_id, normalized
        )
        if not candidates:
            return None

        # Candidates are PERSON-first, then by id (storage contract).
        person = next((c for c in candidates if c["entity_type"] == EntityType.PERSON.value), None)
        concept = next((c for c in candidates if c["entity_type"] == EntityType.CONCEPT.value), None)

        if etype == EntityType.PERSON.value and concept is not None:
            # Resolving as PERSON while a CONCEPT for the same name exists: unify.
            # Handled BEFORE the generic same-type shortcut so that, when a PERSON
            # node ALSO exists, the CONCEPT (and its mention members) is merged in
            # rather than silently left behind on a dangling CONCEPT node.
            if person is not None:
                # Both nodes exist: fold the CONCEPT (its mention members/aliases)
                # into the PERSON so self + mention co-locate on one PERSON node.
                merged = await self.merge(
                    workspace_id,
                    concept["id"],
                    person["id"],
                    reason="name-first unify: mention CONCEPT merged into speaker PERSON",
                )
                return EntityResolution(entity=merged, matched_via="exact", score=1.0)
            # Only a CONCEPT exists; PROMOTE it to PERSON in place (keep its id,
            # members, aliases) so the prior mentions now live on the PERSON node.
            prov = dict(concept.get("provenance") or {})
            prov["promoted_from"] = EntityType.CONCEPT.value
            prov["promoted_reason"] = "name-first: name recognized as PERSON (speaker/known person)"
            if source_memory_id is not None:
                prov.setdefault("promoted_source_memory_id", source_memory_id)
            if observer_id is not None:
                prov.setdefault("promoted_observer_id", observer_id)
            updated = await self._storage.update_entity(
                workspace_id,
                concept["id"],
                entity_type=EntityType.PERSON.value,
                provenance=prov,
            )
            self.logger.debug(
                "Promoted CONCEPT entity %s (%r) to PERSON in workspace %s",
                concept["id"], name, workspace_id,
            )
            return EntityResolution(entity=_entity_from_dict(updated), matched_via="exact", score=1.0)

        if etype == EntityType.CONCEPT.value and person is not None:
            # A mention of a known person resolves to the PERSON node.
            return EntityResolution(entity=_entity_from_dict(person), matched_via="exact", score=1.0)

        # No PERSON⟷CONCEPT unification applies (e.g. only an unrelated-type node
        # exists). Fall through to the standard typed path.
        return None

    async def upsert(
        self,
        workspace_id: str,
        name: str,
        entity_type: EntityType,
        *,
        aliases: list[str] | None = None,
        confidence: float = 1.0,
        provenance: dict | None = None,
        representative_memory_id: str | None = None,
    ) -> Entity:
        etype = _etype_str(entity_type)
        normalized = normalize_entity_name(name)

        existing = await self._storage.find_entity_by_normalized_name(workspace_id, etype, normalized)
        if existing is not None:
            entity_id = existing["id"]
            updates: dict = {"confidence": confidence}
            if provenance is not None:
                merged_prov = {**(existing.get("provenance") or {}), **provenance}
                updates["provenance"] = merged_prov
            if representative_memory_id is not None:
                updates["representative_memory_id"] = representative_memory_id
            await self._storage.update_entity(workspace_id, entity_id, **updates)
            for alias in aliases or []:
                await self._storage.add_entity_alias(
                    workspace_id, entity_id, alias, normalize_entity_name(alias), source="upsert"
                )
            updated = await self._storage.get_entity(workspace_id, entity_id)
            return _entity_from_dict(updated)

        created = await self._storage.store_entity(
            {
                "workspace_id": workspace_id,
                "entity_type": etype,
                "canonical_name": name,
                "normalized_name": normalized,
                "confidence": confidence,
                "provenance": provenance or {},
                "representative_memory_id": representative_memory_id,
                "status": "active",
                "aliases": aliases or [],
            }
        )
        return _entity_from_dict(created)

    async def get(self, workspace_id: str, entity_id: str) -> Entity | None:
        d = await self._storage.get_entity(workspace_id, entity_id)
        return _entity_from_dict(d) if d else None

    async def list_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 100,
    ) -> list[Entity]:
        """List canonical entities in a workspace (deterministic order by id).

        Thin delegate over ``storage.list_workspace_entities``; defaults to
        ``status="active"`` so merged tombstones are excluded."""
        rows = await self._storage.list_workspace_entities(workspace_id, status=status, limit=limit)
        return [_entity_from_dict(r) for r in rows]

    async def clear_workspace(self, workspace_id: str) -> int:
        """Wipe the workspace's entity registry (entities + aliases + members).

        Thin delegate over ``storage.delete_workspace_entities``; returns the count
        of entities deleted. Used to rebuild the entity graph from scratch.
        """
        return await self._storage.delete_workspace_entities(workspace_id)

    async def enrich_entities(
        self,
        workspace_id: str,
        linker,
        *,
        limit: int = 500,
        eligible_types: set[str] | None = None,
        source: str = "wikidata",
        overwrite: bool = False,
    ) -> dict:
        """Link workspace entities to an external KB and fold the result into provenance.

        For each active entity (optionally filtered to ``eligible_types``), calls
        ``linker.link(name, type)`` and, on a returned link, writes
        ``provenance.external_ids[source] = external_id`` plus a richer
        ``provenance.external_links[source]`` record via ``storage.update_entity``.
        Idempotent: an entity that already carries ``source`` is skipped unless
        ``overwrite``. Best-effort — a ``None`` linker, a link failure, or a store
        failure never aborts the batch. Returns ``{checked, enriched, skipped}``.
        """
        result = {"checked": 0, "enriched": 0, "skipped": 0}
        if linker is None:
            return result
        entities = await self.list_entities(workspace_id, status="active", limit=limit)
        for ent in entities:
            etype = getattr(ent.entity_type, "value", ent.entity_type)
            if eligible_types is not None and etype not in eligible_types:
                continue
            result["checked"] += 1
            provenance = dict(ent.provenance or {})
            existing_ids = dict(provenance.get("external_ids") or {})
            if not overwrite and source in existing_ids:
                result["skipped"] += 1
                continue
            try:
                link = await linker.link(ent.canonical_name, str(etype))
            except Exception as e:  # noqa: BLE001 - linking is best-effort
                self.logger.debug("Entity linking failed for %s: %s", ent.canonical_name, e)
                link = None
            if link is None:
                result["skipped"] += 1
                continue
            existing_ids[link.source] = link.external_id
            provenance["external_ids"] = existing_ids
            links = dict(provenance.get("external_links") or {})
            links[link.source] = {
                "id": link.external_id,
                "label": link.label,
                "description": link.description,
                "url": link.url,
                "score": link.score,
            }
            provenance["external_links"] = links
            try:
                await self._storage.update_entity(workspace_id, ent.id, provenance=provenance)
                result["enriched"] += 1
            except Exception as e:  # noqa: BLE001 - one store failure never aborts the batch
                self.logger.debug("Entity enrichment store failed for %s: %s", ent.id, e)
                result["skipped"] += 1
        self.logger.info(
            "Entity enrichment for %s (source=%s): checked=%d enriched=%d skipped=%d",
            workspace_id, source, result["checked"], result["enriched"], result["skipped"],
        )
        return result

    async def related_entities(
        self, workspace_id: str, entity_id: str, *, limit: int = 10, min_shared: int = 1
    ) -> list[RelatedEntity]:
        """Entities related to ``entity_id`` by co-occurrence in shared member memories.

        A lightweight, interpretable graph analytic: entities mentioned in the same
        memories are related. Ranked by shared-memory overlap; ``score`` is that
        overlap as a fraction of the source entity's members. Reads the workspace
        membership edges once (``list_workspace_entity_members``) and computes the
        co-occurrence in memory. Returns [] when the entity has no members.
        """
        from collections import Counter

        rows = await self._storage.list_workspace_entity_members(workspace_id)
        target_memories = {
            r["memory_id"] for r in rows
            if r.get("entity_id") == entity_id and r.get("memory_id")
        }
        if not target_memories:
            return []

        co: Counter = Counter()
        for r in rows:
            eid, mid = r.get("entity_id"), r.get("memory_id")
            if eid and eid != entity_id and mid in target_memories:
                co[eid] += 1

        related: list[RelatedEntity] = []
        denom = len(target_memories)
        for eid, shared in co.most_common():
            if shared < min_shared:
                break
            entity = await self.get(workspace_id, eid)
            if entity is not None and entity.status == "active":
                related.append(
                    RelatedEntity(entity=entity, shared_memories=shared, score=round(shared / denom, 3))
                )
            if len(related) >= limit:
                break
        return related

    async def cooccurrence_map(
        self, workspace_id: str, *, min_shared: int = 1, max_entities_per_memory: int = 50
    ) -> dict[str, dict[str, int]]:
        """Full pairwise entity co-occurrence for a workspace, computed ONCE.

        Returns ``{entity_id: {other_entity_id: shared_memory_count}}`` — the whole
        relatedness graph in a single pass over the membership edges, for callers
        (e.g. KB generation) that need every entity's neighborhood at once instead
        of one ``related_entities`` query per entity. Memories that mention more than
        ``max_entities_per_memory`` entities are skipped (noise + quadratic blowup).
        """
        from collections import defaultdict

        rows = await self._storage.list_workspace_entity_members(workspace_id)
        mem_to_entities: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            eid, mid = r.get("entity_id"), r.get("memory_id")
            if eid and mid:
                mem_to_entities[mid].add(eid)

        co: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entities in mem_to_entities.values():
            if len(entities) < 2 or len(entities) > max_entities_per_memory:
                continue
            ordered = sorted(entities)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a, b = ordered[i], ordered[j]
                    co[a][b] += 1
                    co[b][a] += 1

        return {
            eid: {o: c for o, c in others.items() if c >= min_shared}
            for eid, others in co.items()
            if any(c >= min_shared for c in others.values())
        }

    async def find_by_alias(
        self,
        workspace_id: str,
        alias: str,
        *,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        etype = _etype_str(entity_type) if entity_type is not None else None
        normalized = normalize_entity_name(alias)
        hits = await self._storage.find_entities_by_normalized_alias(
            workspace_id, normalized, entity_type=etype
        )
        return [_entity_from_dict(h) for h in hits]

    async def list_members(
        self,
        workspace_id: str,
        entity_id: str,
        *,
        role: str | None = None,
        limit: int = 100,
    ) -> list[Member]:
        rows = await self._storage.list_entity_members(workspace_id, entity_id, role=role, limit=limit)
        return [
            Member(
                entity_id=r["entity_id"],
                memory_id=r["memory_id"],
                role=r["role"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    async def add_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        *,
        role: str = "mention",
        confidence: float = 1.0,
    ) -> Member:
        row = await self._storage.add_entity_member(
            workspace_id, entity_id, memory_id, role=role, confidence=confidence
        )
        return Member(
            entity_id=row["entity_id"],
            memory_id=row["memory_id"],
            role=row["role"],
            confidence=row["confidence"],
        )

    async def merge(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
        *,
        reason: str,
    ) -> Entity:
        # Self-merge guard: merging an entity into itself is always a no-op.
        # Return the entity unchanged rather than raising so callers (including
        # the adjudication handler) stay idempotent without a special-case check.
        if source_id == target_id:
            self.logger.debug(
                "merge(%s -> %s) is a self-merge; returning entity unchanged",
                source_id, target_id,
            )
            entity = await self._storage.get_entity(workspace_id, source_id)
            if entity is None:
                raise LookupError(
                    f"merge: self-merge entity ({source_id}) does not exist"
                )
            return _entity_from_dict(entity)

        source = await self._storage.get_entity(workspace_id, source_id)
        target = await self._storage.get_entity(workspace_id, target_id)
        if source is None or target is None:
            raise LookupError(
                f"merge requires both source ({source_id}) and target ({target_id}) to exist"
            )

        # Active-status guard: a merge is irreversible. If either entity is
        # already merged/inactive, the merge has already happened (or the entity
        # is in an invalid state). Return the surviving target without re-running
        # any of the destructive steps so TOCTOU re-runs are safe no-ops.
        if source.get("status") != "active":
            self.logger.debug(
                "merge(%s -> %s): source status=%r (not active); no-op (idempotent)",
                source_id, target_id, source.get("status"),
            )
            return _entity_from_dict(target)
        if target.get("status") != "active":
            self.logger.debug(
                "merge(%s -> %s): target status=%r (not active); no-op (idempotent)",
                source_id, target_id, target.get("status"),
            )
            return _entity_from_dict(target)

        # Reassign members from source to target (collisions dropped by storage).
        await self._storage.reassign_entity_members(workspace_id, source_id, target_id)
        supports_capability = getattr(self._storage, "supports_capability", None)
        if supports_capability and supports_capability("entity_relations"):
            await self._storage.reconcile_entity_relations_after_merge(
                workspace_id,
                source_id,
                target_id,
            )

        # Alias carry-forward: fold the source's canonical name + every source
        # alias into the target as aliases.
        carry: list[str] = [source["canonical_name"], *source.get("aliases", [])]
        for alias in carry:
            await self._storage.add_entity_alias(
                workspace_id, target_id, alias, normalize_entity_name(alias), source="merge"
            )

        # Tombstone the source: status="merged", merged_into=target_id, record reason.
        source_prov = {**(source.get("provenance") or {}), "merged_reason": reason, "merged_into": target_id}
        await self._storage.update_entity(
            workspace_id,
            source_id,
            status="merged",
            merged_into=target_id,
            provenance=source_prov,
        )

        merged_target = await self._storage.get_entity(workspace_id, target_id)
        return _entity_from_dict(merged_target)


class DefaultEntityRegistryServicePlugin(EntityRegistryServicePluginBase):
    """Plugin for the relational (OSS default) entity registry service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> DefaultEntityRegistryService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultEntityRegistryService(storage=storage, v=v)
