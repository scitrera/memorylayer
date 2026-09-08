"""Entity Registry Service — base interface (entity registry slice 1).

``EntityRegistryService`` is the canonical-entity layer over the workspace
memory set. It resolves surface names to stable entity nodes (exact → alias →
create), accretes ``Member`` rows as memories mention an entity, and supports
merging duplicate entities. The method signatures here are the cross-backend
contract: every parameter after the positional ``workspace_id`` (+ key args)
is keyword-only so the contract stays additive/stable as options are added, and
the return DTOs (``models.entity_registry``) are identical across the OSS
relational backend and the enterprise Postgres backend.

Slice 1 is deterministic and relational only — no embeddings, no LLM, no graph
database. ``resolve`` is exact + alias + create; semantic / fuzzy resolution is
a follow-on.

Conventions shared by all methods:
  * ``workspace_id`` is the first positional arg and the hard isolation
    boundary — no operation ever crosses workspaces.
  * Normalization is done via ``_normalize.normalize_entity_name`` so both
    backends agree byte-for-byte on the match key.
"""

from abc import ABC, abstractmethod

from ...config import (
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
    MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
)
from ...models.entity_registry import Entity, EntityResolution, EntityType, Member, SeedEntity
from .._constants import EXT_ENTITY_REGISTRY_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base


class EntityRegistryService(ABC):
    """Interface for canonical entity resolution + member accretion + merge."""

    @abstractmethod
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
        """Resolve a surface ``name`` to a canonical entity within a workspace.

        Resolution order (deterministic, relational): normalize ``name`` →
        exact match on ``(workspace_id, entity_type, normalized_name)`` →
        alias match on the normalized alias → create a new entity (only when
        ``allow_create`` is True).

        Returns an ``EntityResolution`` whose ``matched_via`` reports the path
        (``"exact"`` / ``"alias"`` / ``"created"``). Raises ``LookupError`` when
        nothing matched and ``allow_create`` is False.

        When ``promote`` is True (the accretion path opts in), resolution becomes
        NAME-FIRST: a given normalized name maps to ONE canonical entity within
        the workspace regardless of type, and a CONCEPT is PROMOTED to PERSON (or
        merged into an existing PERSON of the same name) when PERSON is requested.
        This keeps a person who both speaks (PERSON/self) and is mentioned
        (CONCEPT/mention) on a single node so perspective queries resolve. Default
        ``promote=False`` preserves the strict typed semantics for callers that
        pass an explicit type (e.g. ``get_representation`` typed mode).
        """
        ...

    @abstractmethod
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
        """Create or update a canonical entity, returning the stored row.

        If an active entity already exists for
        ``(workspace_id, entity_type, normalized_name)`` it is updated
        (aliases merged in, optional fields refreshed); otherwise a new entity
        is created.
        """
        ...

    @abstractmethod
    async def get(self, workspace_id: str, entity_id: str) -> Entity | None:
        """Return the entity by id, or ``None`` if it does not exist."""
        ...

    async def list_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 100,
    ) -> list[Entity]:
        """List canonical entities in a workspace (deterministic order by id).

        Non-abstract registry-wide enumeration over the workspace's entity set,
        defaulting to ``status="active"`` (merged tombstones excluded). Backends
        override with their storage seam; the base raises so an unsupported
        backend fails loudly rather than silently returning nothing.
        """
        raise NotImplementedError("Entity registry backend does not implement list_entities")

    async def clear_workspace(self, workspace_id: str) -> int:
        """Delete ALL canonical entities (+ their aliases/members) in a workspace.

        Returns the number of entities deleted. A registry-wide wipe used to
        rebuild the entity graph from scratch — e.g. a backfill reset after
        changing the extractor. Backends override with their storage seam; the base
        raises so an unsupported backend fails loudly rather than silently no-op'ing.
        """
        raise NotImplementedError("Entity registry backend does not implement clear_workspace")

    async def dedupe_workspace(
        self, workspace_id: str, *, threshold: float | None = None, limit: int = 1000
    ) -> dict:
        """Batch-dedupe: cluster transitively-similar entities and merge each cluster
        into its most-established representative (most members > aliases > confidence).

        Complements resolve-time fuzzy dedup by catching duplicates that already
        coexist (e.g. a seed + an extracted surface form, or entities created before
        the fuzzy tier). Requires a similarity backend (name embeddings); the base
        raises so an unsupported backend fails loudly. Returns ``{clusters, merged}``.
        """
        raise NotImplementedError("Entity registry backend does not implement dedupe_workspace")

    async def seed_entities(self, workspace_id: str, entities: list[SeedEntity]) -> dict:
        """Seed a curated catalog of canonical entities (create-or-update).

        A thin batch over ``upsert`` — the entity authority stays the registry, so
        this is not a parallel store. Each seed's ``description`` + ``external_ids``
        are folded into ``provenance`` (``source="seed"``), the same place a future
        Wikidata/external-KB enrichment writes into. Idempotent (``upsert`` merges
        aliases + refreshes fields on an existing active entity), and per-row
        failures never abort the batch.

        Callers SHOULD validate ``entity_type`` against the ontology entity-type
        vocabulary before calling; this method trusts the given types.

        Returns ``{"seeded": int, "failed": [name, ...]}``.
        """
        from .provenance import build_provenance

        seeded = 0
        failed: list[str] = []
        for e in entities:
            provenance = build_provenance(
                "entity.seed",
                source="seed",
                description=e.description,
                external_ids=e.external_ids or None,
            )
            try:
                await self.upsert(
                    workspace_id,
                    e.name,
                    e.entity_type,
                    aliases=e.aliases or None,
                    confidence=e.confidence,
                    provenance=provenance,
                )
                seeded += 1
            except Exception:  # noqa: BLE001 - one bad row must not abort the catalog load
                failed.append(e.name)
        return {"seeded": seeded, "failed": failed}

    @abstractmethod
    async def find_by_alias(
        self,
        workspace_id: str,
        alias: str,
        *,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """Return active entities whose aliases include the normalized ``alias``.

        Optionally restricted to a single ``entity_type``. Results are ordered
        by entity id (deterministic tie-break). When ``resolve`` encounters
        multiple alias matches it takes ``[0]`` — the entity with the lowest
        id — and logs at DEBUG so operators can spot and merge duplicates.
        """
        ...

    @abstractmethod
    async def list_members(
        self,
        workspace_id: str,
        entity_id: str,
        *,
        role: str | None = None,
        limit: int = 100,
    ) -> list[Member]:
        """List membership edges for an entity, optionally filtered by ``role``."""
        ...

    @abstractmethod
    async def add_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        *,
        role: str = "mention",
        confidence: float = 1.0,
    ) -> Member:
        """Attach a memory to an entity as a member (idempotent on
        ``(entity_id, memory_id, role)``)."""
        ...

    @abstractmethod
    async def merge(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
        *,
        reason: str,
    ) -> Entity:
        """Merge ``source_id`` into ``target_id`` and return the surviving target.

        Reassigns the source's members and aliases to the target, folds the
        source's canonical name in as a target alias (alias carry-forward), and
        tombstones the source (``status="merged"``, ``merged_into=target_id``).
        """
        ...


EntityRegistryServicePluginBase = make_service_plugin_base(
    ext_name=EXT_ENTITY_REGISTRY_SERVICE,
    config_key=MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_PROVIDER,
    # Slice 1 deps are intentionally minimal — just storage. Embedding /
    # extraction / contradiction deps are follow-ons.
    dependencies=(EXT_STORAGE_BACKEND,),
)
