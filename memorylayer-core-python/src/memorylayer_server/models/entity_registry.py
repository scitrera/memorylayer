"""Entity registry domain models for MemoryLayer (entity registry slice 1).

These are the backend-agnostic DTOs returned by ``EntityRegistryService`` — the
canonical-entity layer that sits above raw memories. An ``Entity`` is a stable,
workspace-scoped node (a person/org/project/...) that accretes ``Member`` rows
as memories mention it. ``EntityResolution`` reports how a name was resolved
(exact match, alias match, or freshly created).

The SHAPES here are the contract: the OSS relational ``default`` backend and the
enterprise Postgres backend MUST return identical structures so a caller can
swap backends transparently. Slice 1 is deterministic and relational only — no
embeddings, no LLM, no graph database. (Those are follow-ons.)

Modeled after the ``models/graph_query.py`` DTO style: minimal pydantic models,
explicit ``Field`` descriptions, plain ``str`` ids.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Coarse entity category. Deliberately small for slice 1; type refinement
    (sub-types, LLM-assisted typing) is a follow-on. ``CONCEPT`` is the catch-all
    default for regex-extracted spans that are not clearly a person/org/etc.
    """

    PERSON = "person"
    ORG = "org"
    PROJECT = "project"
    PLACE = "place"
    CONCEPT = "concept"
    EVENT = "event"


class Entity(BaseModel):
    """A canonical, workspace-scoped entity node.

    ``normalized_name`` is the deterministic key (see ``_normalize``): exact
    resolution matches on ``(workspace_id, entity_type, normalized_name)``.
    ``aliases`` are alternate surface forms folded in over time. ``provenance``
    records how/when the entity came to exist (e.g. ``matched_via`` history,
    source memory ids). ``status`` is ``"active"`` or ``"merged"``; a merged
    entity carries ``merged_into`` pointing at its surviving target.
    """

    id: str
    workspace_id: str
    # A string, not the EntityType enum: the core types (person/org/...) are the
    # reserved set with special resolution semantics, but the ontology entity-type
    # vocabulary is extensible (domain types like "equipment"), so the DTO must
    # carry any validated string. Core EntityType values are still valid inputs
    # (EntityType is a str-Enum, so comparisons like ``entity_type == EntityType.PERSON``
    # continue to work).
    entity_type: str = Field(..., description="Entity type (core EntityType value or an ontology domain type)")
    canonical_name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list, description="Alternate surface forms")
    confidence: float = Field(1.0, description="Resolution/creation confidence (0.0-1.0)")
    provenance: dict = Field(default_factory=dict, description="How/when this entity was created or matched")
    representative_memory_id: str | None = Field(
        None, description="Optional canonical memory exemplar for this entity"
    )
    status: str = Field("active", description="'active' or 'merged'")
    merged_into: str | None = Field(None, description="Surviving entity id when status == 'merged'")
    created_at: datetime
    updated_at: datetime


class Member(BaseModel):
    """A membership edge: a memory that mentions / belongs to an entity.

    Mirrors the association-edge style. ``role`` describes the relationship
    (default ``"mention"``); ``confidence`` is the accretion confidence.
    """

    entity_id: str
    memory_id: str
    role: str = Field("mention", description="Membership role, e.g. 'mention'")
    confidence: float = Field(1.0, description="Membership confidence (0.0-1.0)")


class EntityResolution(BaseModel):
    """Result of resolving a surface name to a canonical entity.

    ``matched_via`` reports the resolution path: an existing exact/alias hit, a
    freshly created node, or an ``"embedding"`` (semantic-fuzzy) match. The OSS
    ``default`` backend only ever emits ``exact``/``alias``/``created``; the
    enterprise embedding-fuzzy backend additionally emits ``"embedding"`` when a
    surface form resolves to an existing entity by name-embedding similarity.
    ``score`` is the match strength (1.0 for exact; entity confidence for alias;
    cosine similarity for embedding matches).
    """

    entity: Entity
    matched_via: Literal["exact", "alias", "created", "embedding"]
    score: float = Field(1.0, description="Match strength (1.0 exact; entity confidence otherwise)")


class SeedEntity(BaseModel):
    """One canonical entity to seed into the registry (a curated catalog row).

    Seeding is create-or-update via ``upsert`` — a high-precision way to
    pre-canonicalize entities so that later extracted mentions attach to the seed
    (and its aliases/description) instead of fragmenting into new nodes. ``name``
    is the canonical surface form; ``entity_type`` is validated against the
    ontology entity-type vocabulary. ``description`` and ``external_ids`` (e.g.
    ``{"wikidata": "Q95"}``) are folded into the entity ``provenance`` — the same
    place a future external-KB enrichment (Wikidata linking) writes into.
    """

    name: str = Field(..., min_length=1, description="Canonical surface name")
    entity_type: str = Field("concept", description="Entity type (validated against the ontology vocabulary)")
    aliases: list[str] = Field(default_factory=list, description="Alternate surface forms")
    description: str | None = Field(None, description="Short human description (stored in provenance)")
    external_ids: dict = Field(default_factory=dict, description="External-KB ids, e.g. {'wikidata': 'Q95'}")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Seed confidence")


class EntityProvenance(BaseModel):
    """A normalized, PROV-O-flavored lineage view of a canonical entity.

    Derived (tolerantly) from the entity's raw ``provenance`` dict + fields, which
    different producers populate with different keys (resolve/create, name-first
    promotion, seed, external-KB enrichment). This turns that ad-hoc bag into one
    auditable shape: how the entity ORIGINATED, which memories first surfaced it,
    and which external knowledge bases it links to. The raw dict is included for
    completeness.
    """

    entity_id: str
    origin: str = Field("unknown", description="'created' | 'matched' | 'promoted' | 'seeded' | 'unknown'")
    activity: str | None = Field(None, description="PROV-O activity that produced it, e.g. 'entity.create'")
    agent: str | None = Field(None, description="PROV-O agent (who recorded it)")
    description: str | None = Field(None, description="Human description (from seed or an external link)")
    external_links: dict = Field(default_factory=dict, description="{source: {id, label, description, url, score}}")
    source_memory_ids: list[str] = Field(default_factory=list, description="Memories that first surfaced this entity")
    generated_at: str | None = Field(None, description="When the entity was created/first recorded (ISO 8601)")
    raw: dict = Field(default_factory=dict, description="The full underlying provenance dict")


class RelatedEntity(BaseModel):
    """A canonical entity related to another by co-occurrence in shared memories.

    Two entities are "related" when they are mentioned together — i.e. accrete
    from the same member memories. ``shared_memories`` is that overlap count;
    ``score`` is the overlap as a fraction of the source entity's members (0.0-1.0).
    A lightweight, interpretable graph analytic that surfaces the entity neighborhood.
    """

    entity: Entity
    shared_memories: int = Field(..., description="Member memories shared with the source entity")
    score: float = Field(..., description="Overlap / source-entity member count (0.0-1.0)")


class ExternalEntityLink(BaseModel):
    """A resolved link from a canonical entity to an external knowledge base.

    Returned by an ``EntityLinkerService`` (e.g. Wikidata). Folded into the entity
    ``provenance`` by the enrichment path: ``external_ids[source] = external_id``
    plus a richer ``external_links[source]`` record. ``score`` is the linker's
    confidence (1.0 = exact label match).
    """

    source: str = Field(..., description="External KB name, e.g. 'wikidata'")
    external_id: str = Field(..., description="External id, e.g. 'Q95'")
    label: str = Field(..., description="Canonical label in the external KB")
    description: str | None = Field(None, description="Short external-KB description")
    url: str | None = Field(None, description="Canonical URL for the external record")
    score: float = Field(1.0, ge=0.0, le=1.0, description="Link confidence")
