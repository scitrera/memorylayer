"""Representation domain models for MemoryLayer (P3 perspective slice 1).

These are the backend-agnostic DTOs returned by ``RepresentationService`` — the
perspective-assembly surface that answers "what observer O understands about
subject S". A ``Representation`` is a tight, deterministic (observer, subject)
view: the OBSERVER's own memories that pertain to the SUBJECT (or, when O == S,
the subject's self-reports).

This is a SCOPED ASSEMBLY surface, NOT a recall channel. The service never calls
``recall()`` and never injects members into recall. Its value is the tight
(observer, subject) scope via an INTERSECTION of the observer's authored
memories with the subject's mentions — which is leakage-safe (it excludes turns
about the subject authored by *other* observers).

The SHAPES here are the contract: the OSS relational ``default`` backend and the
enterprise backend (which adds LLM-derived beliefs + consolidation as follow-ons)
MUST return identical structures so a caller can swap backends transparently.
Slice 1 is deterministic — no embeddings, no LLM, no graph database, no recall.
``derived_beliefs`` is present-but-empty in slice 1 (kept in the shape for
contract stability; the enterprise LLM-derivation backend populates it).

Modeled after the ``models/entity_registry.py`` / ``models/graph_query.py`` DTO
style: minimal pydantic models, explicit ``Field`` descriptions, plain ``str``
ids. ``Entity`` is imported from ``models/entity_registry.py`` so the resolved
observer/subject nodes share one canonical shape.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .entity_registry import Entity


class Observation(BaseModel):
    """A single memory that contributes to the assembled representation.

    ``role`` records WHY this memory is in scope: ``"self"`` (authored by the
    observer about itself — the O == S self-report case, or the observer's own
    authored side of the intersection), ``"mention"`` (the memory mentions the
    subject), or ``"subject_tagged"`` (reserved for a future explicit
    subject-tagging path). ``relevance`` / ``match_signals`` mirror the recall
    evidence fields but are populated WITHOUT calling recall — they are
    carried straight off the source memory when present.
    """

    memory_id: str
    content: str
    role: Literal["self", "mention", "subject_tagged"]
    relevance: float | None = None
    event_time: datetime | None = None
    match_signals: list[str] | None = None


class RepresentationProfile(BaseModel):
    """A compact profile assembled deterministically (no LLM).

    Built from a PROFILE-subtype memory when one is in scope (``derived=False``,
    ``summary`` = that memory's content), else synthesized from the top
    observation contents (still ``derived=False`` — slice 1 never uses an LLM,
    so the "profile" is a deterministic concatenation, not a generated summary).
    ``source_memory_ids`` records which memories backed it.
    """

    summary: str | None = None
    source_memory_ids: list[str] = Field(default_factory=list)
    derived: bool = False


class DerivedBelief(BaseModel):
    """An LLM-derived belief about the subject (enterprise follow-on).

    Present in the shape for contract stability; slice 1 always returns an empty
    ``derived_beliefs`` list. The enterprise backend populates these via LLM
    derivation + consolidation over the assembled observations.
    """

    statement: str
    confidence: float
    support_memory_ids: list[str]
    contradicted: bool = False


class Representation(BaseModel):
    """What observer O understands about subject S.

    ``is_self`` is True when the observer and subject resolve to the same
    canonical entity (a self-report view). ``observations`` is the deterministic,
    scoped, time-ordered memory set. ``profile`` is an optional compact summary.
    ``derived_beliefs`` is empty in slice 1 (enterprise follow-on). ``provenance``
    records the resolution paths, the scoping mode (``self`` | ``intersection``),
    counts, and whether the observation set was truncated.
    """

    observer: Entity
    subject: Entity
    is_self: bool
    observations: list[Observation]
    profile: RepresentationProfile | None = None
    derived_beliefs: list[DerivedBelief] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)


class UserRepresentation(BaseModel):
    """A USER-SCOPE self-representation: who a user is, across workspaces.

    The cross-workspace analogue of ``Representation`` with observer == subject
    == the user — a user's preferences/traits/personality assembled from their
    USER-scope (``_global_user``) memories ("personality follows the user").
    It is NOT keyed on an (observer, subject) entity pair: the
    user is identified by ``user_id`` (the partition inside ``_global_user``),
    so there is no resolved ``Entity`` here.

    CARDINAL LEAKAGE PROPERTY: ``observations`` is assembled ONLY from the
    ``_global_user`` workspace constrained by a FORCED ``user_id`` filter — one
    user's preferences can NEVER appear in another user's representation. The
    ``origin_workspace_ids`` provenance records the source workspaces the
    user-scope memories were promoted from (each row carries
    ``metadata['origin_workspace_id']``), evidencing the cross-workspace span.

    Shape parity with ``Representation``: ``observations`` / ``profile`` /
    ``derived_beliefs`` / ``provenance`` are identical so a caller can consume a
    user-representation with the same code that consumes a perspective
    representation. ``derived_beliefs`` is empty in the OSS deterministic path
    (the enterprise backend populates it via the same LLM belief layer)."""

    user_id: str
    observations: list[Observation]
    profile: RepresentationProfile | None = None
    derived_beliefs: list[DerivedBelief] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
