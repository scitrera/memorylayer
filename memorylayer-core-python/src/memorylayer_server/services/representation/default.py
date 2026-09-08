"""Default (relational) Representation Service implementation (P3 slice 1).

``DefaultRepresentationService`` assembles "what observer O understands about
subject S" deterministically over the entity registry + relational storage — no
embeddings, no LLM, no graph database, and (critically) NO ``recall()`` call and
NO injection of members into recall.

The assembly is a SCOPED INTERSECTION, not a retrieval:

  * self (O == S): the subject's ``role="self"`` members (self-reports).
  * other: ``{observer's role="self" members} ∩ {subject's role="mention"
    members}`` — i.e. memories the OBSERVER authored that MENTION the subject.
    This intersection is the leakage-safe perspective scope: a turn that mentions
    the subject but was authored by a DIFFERENT observer is EXCLUDED.

Scoped memories are fetched with ``track_access=False`` (decay is not
perturbed), filtered (drop None/archived/deleted), deduped, ordered by
``event_time`` desc then ``created_at`` desc, and truncated to ``limit``.

``derived_beliefs`` is always empty in slice 1 (kept in the DTO shape for
contract stability; enterprise LLM derivation is a follow-on).
"""

import logging

from scitrera_app_framework import Variables, get_extension, get_logger

from ...config import GLOBAL_USER_WORKSPACE_ID
from ...models.entity_registry import Entity, EntityType
from ...models.memory import Memory, MemoryStatus, MemorySubtype
from ...models.representation import (
    Observation,
    Representation,
    RepresentationProfile,
    UserRepresentation,
)
from ...utils import utc_now
from .._constants import EXT_ENTITY_REGISTRY_SERVICE, EXT_STORAGE_BACKEND
from ..entity_registry.base import EntityRegistryService
from ..storage import StorageBackend
from . import RepresentationServicePluginBase
from .base import RepresentationService

# Number of synthesized-profile observation contents to fold into the fallback
# profile summary when no PROFILE-subtype memory is in scope. Small + fixed so
# the profile stays compact and deterministic.
_PROFILE_FALLBACK_TOP_N = 3


def _placeholder_entity(workspace_id: str, name: str, entity_type: EntityType | None) -> Entity:
    """Build a best-effort, non-persisted ``Entity`` for an unresolved side.

    Used only when ``resolve(allow_create=False)`` misses so the returned
    ``Representation`` can still carry the observer/subject the caller asked for
    (with an empty/sentinel id) instead of crashing. ``provenance`` flags it as
    unresolved so callers can tell it is not a real registry node.
    """
    now = utc_now()
    return Entity(
        id="",
        workspace_id=workspace_id,
        entity_type=entity_type or EntityType.PERSON,
        canonical_name=name,
        normalized_name=name,
        provenance={"unresolved": True},
        created_at=now,
        updated_at=now,
    )


class DefaultRepresentationService(RepresentationService):
    """Deterministic (observer, subject) perspective assembly (OSS default)."""

    PROVIDER_NAME = "default"

    def __init__(self, registry: EntityRegistryService, storage: StorageBackend, v: Variables):
        self._registry = registry
        self._storage = storage
        self.logger = get_logger(v, name="RepresentationService")

    async def get_representation(
        self,
        workspace_id: str,
        observer: str,
        subject: str,
        *,
        observer_type: EntityType | None = None,
        subject_type: EntityType | None = None,
        limit: int = 20,
        include_profile: bool = True,
    ) -> Representation:
        # Degrade gracefully if no registry is available (mirrors the registry
        # pattern: the surface returns an empty rep rather than raising).
        if self._registry is None:
            self.logger.debug(
                "get_representation: no entity registry available; returning empty representation"
            )
            return Representation(
                observer=_placeholder_entity(workspace_id, observer, observer_type),
                subject=_placeholder_entity(workspace_id, subject, subject_type),
                is_self=False,
                observations=[],
                profile=None,
                derived_beliefs=[],
                provenance={"error": "no_registry", "scoping_mode": None},
            )

        # 1. Resolve observer + subject (NEVER create — allow_create=False).
        obs_res = None
        subj_res = None
        unresolved: list[str] = []
        try:
            obs_res = await self._registry.resolve(
                workspace_id, observer, observer_type or EntityType.PERSON, allow_create=False
            )
        except LookupError:
            unresolved.append("observer")
        try:
            subj_res = await self._registry.resolve(
                workspace_id, subject, subject_type or EntityType.PERSON, allow_create=False
            )
        except LookupError:
            unresolved.append("subject")

        if unresolved:
            # One or both sides did not resolve — return an EMPTY representation
            # with best-effort entities and provenance noting the unresolved side.
            observer_entity = (
                obs_res.entity if obs_res is not None
                else _placeholder_entity(workspace_id, observer, observer_type)
            )
            subject_entity = (
                subj_res.entity if subj_res is not None
                else _placeholder_entity(workspace_id, subject, subject_type)
            )
            return Representation(
                observer=observer_entity,
                subject=subject_entity,
                is_self=False,
                observations=[],
                profile=None,
                derived_beliefs=[],
                provenance={
                    "unresolved": unresolved,
                    "scoping_mode": None,
                    "observer_matched_via": obs_res.matched_via if obs_res else None,
                    "subject_matched_via": subj_res.matched_via if subj_res else None,
                },
            )

        observer_entity = obs_res.entity
        subject_entity = subj_res.entity

        # 2. Self-detection.
        is_self = observer_entity.id == subject_entity.id

        # 3. Scope the memory id set (deterministic; no recall).
        if is_self:
            scoping_mode = "self"
            self_members = await self._registry.list_members(
                workspace_id, subject_entity.id, role="self", limit=limit * 4
            )
            scoped_ids = [m.memory_id for m in self_members]
            id_roles = {mid: "self" for mid in scoped_ids}
        else:
            scoping_mode = "intersection"
            # Memories the OBSERVER authored that MENTION the subject. The
            # intersection is the leakage guard: a turn mentioning S authored by
            # a DIFFERENT observer is excluded (it is in subj_mentions but not in
            # obs_authored), and a turn O authored that does NOT mention S is
            # excluded (in obs_authored but not in subj_mentions).
            subj_mentions = await self._registry.list_members(
                workspace_id, subject_entity.id, role="mention", limit=limit * 8
            )
            obs_authored = await self._registry.list_members(
                workspace_id, observer_entity.id, role="self", limit=limit * 8
            )
            mention_ids = {m.memory_id for m in subj_mentions}
            authored_ids = {m.memory_id for m in obs_authored}
            scoped_ids = [mid for mid in mention_ids & authored_ids]
            # In the intersection these memories are the observer's own authored
            # turns; tag them "self" (authored-by-observer) consistent with the
            # role they carry in the registry.
            id_roles = {mid: "self" for mid in scoped_ids}

        # 4. Fetch each scoped memory without perturbing decay; filter + dedup.
        fetched: dict[str, Memory] = {}
        for mid in scoped_ids:
            if mid in fetched:
                continue
            mem = await self._storage.get_memory(workspace_id, mid, track_access=False)
            if mem is None:
                continue
            if mem.status in (MemoryStatus.ARCHIVED, MemoryStatus.DELETED):
                continue
            fetched[mid] = mem

        ordered, truncated = self._order_and_truncate(list(fetched.values()), limit)
        observations = self._to_observations(ordered, id_roles)

        # 5. Profile (deterministic; no LLM).
        profile = None
        if include_profile:
            profile = self._build_profile(ordered)

        # 6. Provenance.
        provenance = {
            "scoping_mode": scoping_mode,
            "observer_matched_via": obs_res.matched_via,
            "subject_matched_via": subj_res.matched_via,
            "scoped_id_count": len(scoped_ids),
            "observation_count": len(observations),
            "truncated": truncated,
        }

        return Representation(
            observer=observer_entity,
            subject=subject_entity,
            is_self=is_self,
            observations=observations,
            profile=profile,
            derived_beliefs=[],  # slice 1: always empty (enterprise follow-on)
            provenance=provenance,
        )

    def _order_and_truncate(self, mems: list[Memory], limit: int) -> tuple[list[Memory], bool]:
        """Sort by event_time desc then created_at desc, truncate to ``limit``.

        Shared by ``get_representation`` and ``get_user_representation`` so both
        surfaces order deterministically the same way (no LLM ranking). Memories
        without an ``event_time`` fall back to ``created_at`` for the primary key
        so they interleave sensibly with timestamped ones. Returns the truncated
        list and whether truncation occurred.
        """
        def _sort_key(m: Memory):
            evt = m.event_time or m.created_at
            return (evt, m.created_at)

        ordered = sorted(mems, key=_sort_key, reverse=True)
        truncated = len(ordered) > limit
        return ordered[:limit], truncated

    def _to_observations(
        self, ordered: list[Memory], id_roles: dict[str, str]
    ) -> list[Observation]:
        """Map ordered memories to ``Observation`` DTOs (shared assembly).

        ``id_roles`` records why each memory is in scope ("self"/"mention");
        defaults to "self" for the user-scope self-assembly (the user's own
        user-global memories)."""
        return [
            Observation(
                memory_id=m.id,
                content=m.content,
                role=id_roles.get(m.id, "self"),
                relevance=m.relevance_score,
                event_time=m.event_time,
                match_signals=m.match_signals,
            )
            for m in ordered
        ]

    async def get_user_representation(
        self,
        user_id: str,
        *,
        limit: int = 20,
        include_profile: bool = True,
    ) -> UserRepresentation:
        # FORCED user_id filter: an empty/falsy user_id is unfilterable (a
        # user-global read with no partition would return every user's
        # preferences). Refuse it with an empty, fail-safe representation — this
        # is the cross-user leakage guard at the entry of the path.
        if not user_id:
            self.logger.debug(
                "get_user_representation: no user_id; returning empty representation"
            )
            return UserRepresentation(
                user_id=user_id or "",
                observations=[],
                profile=None,
                derived_beliefs=[],
                provenance={"error": "no_user_id", "scoping_mode": None},
            )

        # Assemble ONLY from the _global_user workspace, constrained by the
        # FORCED user_id filter — the SAME cross-user boundary the recall fan-out
        # uses. No entity resolution / intersection: the user's user-scope
        # memories ARE their self-knowledge (observer == subject == the user).
        # Fail-safe: any storage error -> empty representation, never raises.
        try:
            mems = await self._storage.search_memories_by_filter(
                GLOBAL_USER_WORKSPACE_ID,
                user_id=user_id,
                status=MemoryStatus.ACTIVE.value,
                limit=max(limit * 4, limit),
            )
        except Exception:  # noqa: BLE001 - user-representation must be fail-safe
            self.logger.exception(
                "get_user_representation: storage read failed for user_id=%s; "
                "returning empty representation", user_id,
            )
            return UserRepresentation(
                user_id=user_id,
                observations=[],
                profile=None,
                derived_beliefs=[],
                provenance={"error": "storage_error", "scoping_mode": "user"},
            )

        # Defense-in-depth: re-assert the user_id partition in-process so even a
        # backend that ignored the filter cannot leak another user's rows, and
        # drop archived/deleted (status filter is belt-and-suspenders).
        scoped = [
            m
            for m in mems
            if m.user_id == user_id
            and m.status not in (MemoryStatus.ARCHIVED, MemoryStatus.DELETED)
        ]

        ordered, truncated = self._order_and_truncate(scoped, limit)
        observations = self._to_observations(ordered, {})

        profile = self._build_profile(ordered) if include_profile else None

        # Provenance: the origin workspaces these user-scope memories were
        # promoted from (evidences the cross-workspace span) + counts.
        origin_workspace_ids = sorted(
            {
                origin
                for m in ordered
                if (origin := (m.metadata or {}).get("origin_workspace_id"))
            }
        )
        provenance = {
            "scoping_mode": "user",
            "user_id": user_id,
            "scoped_id_count": len(scoped),
            "observation_count": len(observations),
            "origin_workspace_ids": origin_workspace_ids,
            "truncated": truncated,
        }

        return UserRepresentation(
            user_id=user_id,
            observations=observations,
            profile=profile,
            derived_beliefs=[],  # OSS deterministic path (enterprise populates)
            provenance=provenance,
        )

    def _build_profile(self, ordered: list[Memory]) -> RepresentationProfile | None:
        """Build a deterministic profile from the scoped memories (no LLM).

        Prefers a PROFILE-subtype memory in scope (``derived=False``, summary =
        its content). Falls back to synthesizing a compact summary from the top
        observation contents (still ``derived=False`` — slice 1 never uses an
        LLM). Returns ``None`` only when there are no scoped memories at all.
        """
        if not ordered:
            return None

        for m in ordered:
            if m.subtype == MemorySubtype.PROFILE.value:
                return RepresentationProfile(
                    summary=m.content,
                    source_memory_ids=[m.id],
                    derived=False,
                )

        top = ordered[:_PROFILE_FALLBACK_TOP_N]
        summary = " ".join(m.content for m in top)
        return RepresentationProfile(
            summary=summary,
            source_memory_ids=[m.id for m in top],
            derived=False,
        )


class DefaultRepresentationServicePlugin(RepresentationServicePluginBase):
    """Plugin for the relational (OSS default) representation service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> DefaultRepresentationService:
        registry: EntityRegistryService = get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return DefaultRepresentationService(registry=registry, storage=storage, v=v)
