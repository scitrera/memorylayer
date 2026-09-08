"""Graph Query Service — base interface (P2 Track A).

``GraphQueryService`` is the recall/RAG-facing READ contract over the
workspace association graph. The method signatures here are the cross-backend
contract: every parameter after ``workspace_id`` is keyword-only so the
contract stays stable as options are added, and the return DTOs
(``models.graph_query``) are identical across the relational and AGE backends.

Conventions shared by all methods:
  * ``workspace_id`` is the first positional arg and the hard isolation
    boundary — no query ever crosses workspaces.
  * ``direction`` is one of ``"both"`` / ``"outgoing"`` / ``"incoming"`` and
    follows the stored association direction (source→target).
  * ``relationship_types`` (when given) filters edges to those types.
  * Node/result limits cause ``truncated=True`` rather than an error.
"""

from abc import ABC, abstractmethod

from ...models.graph_query import (
    EntityNeighborhood,
    FragmentResult,
    NeighborResult,
    PathResult,
    PatternMatch,
    RelationshipRollup,
    SubgraphResult,
)
from ...config import DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER, MEMORYLAYER_GRAPH_QUERY_PROVIDER
from .._constants import EXT_GRAPH_QUERY_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base

# Maximum allowed depth / max_hops for traversal methods. Both the OSS
# relational backend and the enterprise AGE backend enforce this bound and
# raise ``ValueError`` for values outside ``[1, MAX_GRAPH_TRAVERSAL_HOPS]``.
# The AGE backend additionally requires this as the upper bound on the literal
# var-length range ``[:ASSOC*1..N]`` (stored in ``_cypher.MAX_TRAVERSAL_HOPS``
# which is set to the same value). Any change here must be mirrored there.
MAX_GRAPH_TRAVERSAL_HOPS: int = 6


def validate_graph_hops(value: int, *, kind: str = "depth") -> int:
    """Validate a traversal depth / max_hops value for ``GraphQueryService``.

    Enforces ``1 <= value <= MAX_GRAPH_TRAVERSAL_HOPS`` and raises
    ``ValueError`` for out-of-range or non-integer inputs. Both the OSS
    relational backend and the AGE backend call this so callers see the same
    error regardless of which backend is active.

    Raises:
        ValueError: if ``value`` is not an int, is a bool, or is out of
            ``[1, MAX_GRAPH_TRAVERSAL_HOPS]``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Invalid %s %r — must be an int." % (kind, value))
    if value < 1 or value > MAX_GRAPH_TRAVERSAL_HOPS:
        raise ValueError(
            "Invalid %s %d — must be 1..%d." % (kind, value, MAX_GRAPH_TRAVERSAL_HOPS)
        )
    return value


class GraphQueryService(ABC):
    """Interface for localized reads over the workspace association graph."""

    @abstractmethod
    async def neighbors(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 50,
    ) -> NeighborResult:
        """Return the depth-bounded neighborhood (ego graph) around ``memory_id``.

        ``depth=1`` returns direct neighbors; ``depth>1`` expands transitively
        up to ``depth`` hops. ``limit`` caps the number of nodes returned
        (sets ``truncated=True`` when exceeded).

        Args:
            depth: traversal depth, must be ``1 <= depth <= MAX_GRAPH_TRAVERSAL_HOPS``.

        Raises:
            ValueError: if ``depth`` is out of ``[1, MAX_GRAPH_TRAVERSAL_HOPS]``.
        """
        ...

    @abstractmethod
    async def k_hop_subgraph(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        depth: int = 2,
        relationship_types: list[str] | None = None,
        direction: str = "both",
        node_limit: int = 200,
    ) -> SubgraphResult:
        """Return the bounded ``depth``-hop subgraph induced by ``memory_ids``.

        The union of the k-hop neighborhoods of every root in ``memory_ids``,
        capped at ``node_limit`` nodes (sets ``truncated=True`` when exceeded).

        Args:
            depth: traversal depth, must be ``1 <= depth <= MAX_GRAPH_TRAVERSAL_HOPS``.

        Raises:
            ValueError: if ``depth`` is out of ``[1, MAX_GRAPH_TRAVERSAL_HOPS]``.
        """
        ...

    @abstractmethod
    async def shortest_path(
        self,
        workspace_id: str,
        src_id: str,
        dst_id: str,
        *,
        max_hops: int = 5,
        relationship_types: list[str] | None = None,
    ) -> PathResult:
        """Return the shortest path from ``src_id`` to ``dst_id``.

        Searches up to ``max_hops`` edges. ``found=False`` (empty path) when no
        path within ``max_hops`` exists.

        Tie-break (multiple equal-length paths): both backends select the path
        whose UUID-normalized node-id sequence is lexicographically smallest.
        This is enforced by expanding BFS frontiers in sorted order (relational)
        and by ``ORDER BY`` on path node-id sequences (AGE).

        Args:
            max_hops: upper bound on path length,
                must be ``1 <= max_hops <= MAX_GRAPH_TRAVERSAL_HOPS``.

        Raises:
            ValueError: if ``max_hops`` is out of ``[1, MAX_GRAPH_TRAVERSAL_HOPS]``.
        """
        ...

    @abstractmethod
    async def typed_pattern(
        self,
        workspace_id: str,
        *,
        relationship: str,
        limit: int = 100,
    ) -> list[PatternMatch]:
        """Return up to ``limit`` edges whose relationship == ``relationship``."""
        ...

    @abstractmethod
    async def relationship_rollup(
        self,
        workspace_id: str,
    ) -> list[RelationshipRollup]:
        """Return per-relationship edge counts for the workspace."""
        ...

    @abstractmethod
    async def entity_neighborhood(
        self,
        workspace_id: str,
        entity_id: str,
        *,
        hops: int = 1,
        memory_limit: int = 100,
        entity_limit: int = 50,
    ) -> EntityNeighborhood:
        """Return the neighborhood of a canonical entity.

        The cross-source / relationship-traversal primitive over the
        canonical-entity layer (P4 channel-G groundwork + perspective lookup —
        NOT a chat-recall expansion):

          * ``memories_for_entity``: the memories the entity MENTIONS (the
            entity's member memories — which include memories linked via aliases
            and, on the enterprise tier, fuzzy/LLM-merged surface forms), capped
            at ``memory_limit``;
          * ``entities_co_mentioned``: entities reachable via
            ``Entity -> MENTIONS -> Memory <- MENTIONS <- Entity`` (sharing a
            mentioned memory), ranked by shared-memory count desc then id asc and
            capped at ``entity_limit``.

        ``hops`` reserved for future multi-hop entity walks; only ``hops=1``
        (direct mentions + 1 shared-memory co-mention) is implemented in this
        slice. An empty/disabled registry yields an empty neighborhood (no error).

        Tier note: the enterprise AGE backend answers this over the C1
        Entity/MENTIONS vertices+edges (rich); the OSS relational backend answers
        the contract-equivalent set over the registry tables (parity relaxed only
        under truncation per the OSS/enterprise parity bar).
        """
        ...

    @abstractmethod
    async def fragments_for_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int = 100,
    ) -> FragmentResult:
        """Return the fact fragments DERIVED_FROM a source memory.

        The fragment-traversal primitive over the decomposed-fact layer: the
        ``subtype="fact"`` memories whose ``metadata["source_id"]`` is
        ``memory_id`` (the facts decomposed from that memory), ordered by
        fragment_id asc and capped at ``limit`` (sets ``truncated=True`` when
        exceeded).

        DARK + MEASURABLE — NOT wired into recall. Whether fragment-traversal
        beats the existing fact channel (E) is a later measurement.

        Tier note: the enterprise AGE backend answers this over the P4.5
        Fragment/DERIVED_FROM vertices+edges (rich, gated behind
        ``MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED``); the OSS relational
        backend answers the contract-equivalent set over the fact memories +
        their ``source_id`` directly (no flag, no graph database). An empty/
        disabled fragment layer yields an empty result — never an error.
        """
        ...


GraphQueryServicePluginBase = make_service_plugin_base(
    ext_name=EXT_GRAPH_QUERY_SERVICE,
    config_key=MEMORYLAYER_GRAPH_QUERY_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)
