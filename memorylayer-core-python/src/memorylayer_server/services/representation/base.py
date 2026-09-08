"""Representation Service — base interface (P3 perspective slice 1).

``RepresentationService`` assembles "what observer O understands about subject S"
as a tight, deterministic ``Representation``. The method signature here is the
cross-backend contract: every parameter after the positional
``workspace_id, observer, subject`` is keyword-only so the contract stays
additive/stable as options are added, and the return DTO
(``models.representation``) is identical across the OSS relational ``default``
backend and the enterprise (LLM-derivation) backend.

This is a SCOPED ASSEMBLY surface, NOT a recall channel. Implementations MUST
NOT call ``recall()`` and MUST NOT inject members into recall — the whole value
is the tight (observer, subject) scope via an INTERSECTION of the observer's
self-authored memories with the subject's mentions (leakage-safe: excludes
turns about the subject authored by *other* observers).

Conventions:
  * ``workspace_id`` is the first positional arg and the hard isolation
    boundary — no operation ever crosses workspaces.
  * Resolution goes through the injected ``EntityRegistryService`` with
    ``allow_create=False`` — the surface NEVER creates entities.
"""

from abc import ABC, abstractmethod

from ...config import (
    DEFAULT_MEMORYLAYER_REPRESENTATION_PROVIDER,
    MEMORYLAYER_REPRESENTATION_PROVIDER,
)
from ...models.entity_registry import EntityType
from ...models.representation import Representation, UserRepresentation
from .._constants import (
    EXT_ENTITY_REGISTRY_SERVICE,
    EXT_REPRESENTATION_SERVICE,
    EXT_STORAGE_BACKEND,
)
from .._plugin_factory import make_service_plugin_base


class RepresentationService(ABC):
    """Interface for deterministic (observer, subject) perspective assembly."""

    @abstractmethod
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
        """Assemble what ``observer`` understands about ``subject``.

        Resolves both surface names via the entity registry
        (``allow_create=False``). When either side does not resolve, returns an
        EMPTY ``Representation`` (best-effort observer/subject placeholders,
        ``observations=[]``) with provenance noting the unresolved side — never
        raises and never creates entities.

        Scoping (deterministic, no recall):
          * self (observer == subject): the subject's ``role="self"`` members.
          * other: the INTERSECTION of the observer's ``role="self"`` members
            with the subject's ``role="mention"`` members — i.e. memories the
            OBSERVER authored that MENTION the subject. This intersection is the
            leakage-safe perspective scope (NOT "all turns mentioning S").

        Scoped memories are fetched without perturbing decay
        (``track_access=False``), filtered (drop None/archived/deleted), deduped,
        sorted by ``event_time`` desc then ``created_at`` desc (no LLM ranking),
        and truncated to ``limit``.

        Args:
            limit: max number of observations to return.
            include_profile: when True, attach a deterministic
                ``RepresentationProfile`` (from a PROFILE-subtype memory if one
                is in scope, else synthesized from the top observations).
        """
        ...

    @abstractmethod
    async def get_user_representation(
        self,
        user_id: str,
        *,
        limit: int = 20,
        include_profile: bool = True,
    ) -> UserRepresentation:
        """Assemble a USER-SCOPE self-representation for ``user_id``.

        This is the cross-workspace user-scope analogue of
        ``get_representation`` with observer == subject == the user: a user's
        preferences/traits/personality assembled from their USER-scope
        (``_global_user``) memories, "personality follows the user". It lifts the
        within-workspace, leakage-safe self-assembly to a within-``_global_user``,
        ``user_id``-filtered self-assembly — reusing the same filter/sort/profile
        machinery, NOT a recall change (the recall fan-out already surfaces these
        additively via ``include_global_user``).

        CARDINAL LEAKAGE PROPERTY (cross-user isolation): the observation set is
        assembled ONLY from the ``_global_user`` workspace constrained by a
        FORCED, non-optional ``user_id`` filter — the same forced filter the
        recall fan-out uses. User B's preferences can NEVER appear in user A's
        user-representation. This is the user-scope analogue of perspective
        leakage-0.

        Fail-safe + dark: ANY failure (no/empty ``user_id``, no
        ``_global_user`` rows, storage error, LLM down in the enterprise belief
        layer) yields an EMPTY ``UserRepresentation`` (``observations=[]``,
        ``derived_beliefs=[]``) — it NEVER raises into the caller.

        Args:
            user_id: the user-scope partition. REQUIRED + forced — an
                empty/falsy ``user_id`` yields an empty representation (a
                user-global read with no partition would be unfilterable).
            limit: max number of observations to return.
            include_profile: when True, attach a deterministic
                ``RepresentationProfile`` (from a PROFILE-subtype memory if one
                is in scope, else synthesized from the top observations).
        """
        ...


RepresentationServicePluginBase = make_service_plugin_base(
    ext_name=EXT_REPRESENTATION_SERVICE,
    config_key=MEMORYLAYER_REPRESENTATION_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_REPRESENTATION_PROVIDER,
    # Assembly depends on the entity registry (resolution + member scoping) and
    # storage (memory fetch). It deliberately does NOT depend on recall.
    dependencies=(EXT_ENTITY_REGISTRY_SERVICE, EXT_STORAGE_BACKEND),
)
