"""Incremental KB rendering differ (Lever 3).

Pure, I/O-free helpers used by ``DefaultKnowledgebaseService.generate()`` to avoid
re-summarizing unchanged communities/entities on every fire and to detect orphaned
(stale) articles for garbage collection.

Two responsibilities, both deterministic and testable in isolation:

1. **Community-stability matching** (:func:`match_communities`). Louvain community ids
   are *not* stable across runs, so a current community is matched to a prior community
   by Jaccard member-set overlap. A matched current community inherits the PRIOR article
   id (``community-{prior_id}``), decoupling the stored id from the volatile Louvain idx.

2. **Per-article content hashing** (:func:`community_content_key`,
   :func:`entity_content_key`). A ``sha256`` over the article's rendered inputs. When the
   recomputed key equals the key stored in the prior article's ``metadata["content_key"]``,
   the prior article is reused verbatim with no LLM call.

The prior ``GraphAnalysis`` (stored as ``model_dump(mode="json")``) is parsed into
:class:`~memorylayer_server.models.graph_analysis.Community` models by the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ...models.graph_analysis import Community
from .linkcheck import RENDER_FORMAT_VERSION

# Key used inside Article.metadata to persist the content hash (free-form JSONB).
CONTENT_KEY_FIELD = "content_key"


@dataclass
class CommunityMatch:
    """Result of matching one current community against the prior run.

    Attributes:
        community: The current-run community (its ``id`` is the volatile Louvain idx).
        article_id: The stable article id this community should be stored under.
            For a matched community this is the *prior* ``community-{prior_id}``;
            for a new community it is ``community-{current_id}``.
        prior_id: The prior community id this matched to, or ``None`` if new.
        prior_label: The prior community label to carry forward, or ``None``.
        is_new: True if no prior community matched at/above the threshold.
    """

    community: Community
    article_id: str
    prior_id: int | None
    prior_label: str | None
    is_new: bool


@dataclass
class MatchResult:
    """Full outcome of community matching for one ``generate()`` run."""

    matches: list[CommunityMatch] = field(default_factory=list)
    # Prior community ids with no current match -> their articles are orphans.
    dissolved_prior_ids: list[int] = field(default_factory=list)


def _community_article_id(community_id: int) -> str:
    return f"community-{community_id}"


def match_communities(
    current: list[Community],
    prior: list[Community],
    threshold: float = 0.5,
) -> MatchResult:
    """Greedy max-weight one-to-one Jaccard matching of current vs prior communities.

    For each current community ``a`` and prior community ``b`` with member sets
    ``Sa``/``Sb``, the Jaccard score is ``|Sa & Sb| / |Sa | Sb|``. Intersection sizes
    are computed in O(|Sa|) via an inverted ``memory_id -> prior_idx`` index. A match
    requires ``J >= threshold``. Each current/prior community is matched at most once.

    Tie-breaking for equal Jaccard: (a) larger absolute overlap ``|Sa & Sb|``, then
    (b) smaller absolute size delta ``||Sa| - |Sb||``, then (c) lower prior id.

    Args:
        current: Communities detected in this run.
        prior: Communities from the previous run (empty -> all current are new).
        threshold: Minimum Jaccard to treat two communities as the same.

    Returns:
        A :class:`MatchResult` classifying every current community (matched/new) and
        listing prior community ids that dissolved (their articles are orphans).
    """
    # Pre-compute member sets and an inverted index over prior membership.
    prior_sets: list[set[str]] = [set(c.memory_ids) for c in prior]
    inverted: dict[str, list[int]] = {}
    for p_idx, members in enumerate(prior_sets):
        for mem_id in members:
            inverted.setdefault(mem_id, []).append(p_idx)

    # Build candidate (current_idx, prior_idx, jaccard, overlap, size_delta, prior_id) tuples.
    candidates: list[tuple[int, int, float, int, int, int]] = []
    for c_idx, community in enumerate(current):
        cur_set = set(community.memory_ids)
        if not cur_set:
            continue
        overlap_counts: dict[int, int] = {}
        for mem_id in cur_set:
            for p_idx in inverted.get(mem_id, ()):  # O(|Sa|) total
                overlap_counts[p_idx] = overlap_counts.get(p_idx, 0) + 1
        for p_idx, overlap in overlap_counts.items():
            union = len(cur_set) + len(prior_sets[p_idx]) - overlap
            if union <= 0:
                continue
            jaccard = overlap / union
            if jaccard < threshold:
                continue
            size_delta = abs(len(cur_set) - len(prior_sets[p_idx]))
            candidates.append((c_idx, p_idx, jaccard, overlap, size_delta, prior[p_idx].id))

    # Greedy: sort by jaccard desc, then overlap desc, then size_delta asc, then prior id asc.
    candidates.sort(key=lambda t: (-t[2], -t[3], t[4], t[5]))

    matched_current: set[int] = set()
    matched_prior: set[int] = set()
    matches_by_current: dict[int, CommunityMatch] = {}

    for c_idx, p_idx, _jaccard, _overlap, _delta, prior_id in candidates:
        if c_idx in matched_current or p_idx in matched_prior:
            continue
        matched_current.add(c_idx)
        matched_prior.add(p_idx)
        matches_by_current[c_idx] = CommunityMatch(
            community=current[c_idx],
            article_id=_community_article_id(prior_id),
            prior_id=prior_id,
            prior_label=prior[p_idx].label,
            is_new=False,
        )

    matches: list[CommunityMatch] = []
    for c_idx, community in enumerate(current):
        match = matches_by_current.get(c_idx)
        if match is None:
            match = CommunityMatch(
                community=community,
                article_id=_community_article_id(community.id),
                prior_id=None,
                prior_label=None,
                is_new=True,
            )
        matches.append(match)

    dissolved = [prior[p_idx].id for p_idx in range(len(prior)) if p_idx not in matched_prior]
    return MatchResult(matches=matches, dissolved_prior_ids=dissolved)


def community_content_key(
    community: Community,
    content_versions: dict[str, str],
) -> str:
    """Compute the content hash for a community article.

    The key folds in each member's ``content_version`` (so an edited member busts the
    cache), the community ``size``, and the sorted ``central_node_ids`` (so a membership
    reshuffle that preserves the set but changes centrality still busts the cache).

    Args:
        community: The community whose article inputs are being hashed.
        content_versions: Map of ``memory_id -> content version string`` for members.
            A missing member contributes an empty version.

    Returns:
        Hex sha256 digest string.
    """
    member_lines = "\n".join(
        f"{mem_id}:{content_versions.get(mem_id, '')}" for mem_id in sorted(community.memory_ids)
    )
    central = ",".join(sorted(community.central_node_ids))
    # fmt= busts the cache when the RENDERER changes, not just the inputs. Without
    # it a rendering fix never reaches an unchanged article: the key still matches,
    # so the prior article is reused verbatim and keeps its stale output forever.
    payload = (
        f"{member_lines}\nsize={community.size}\ncentral={central}"
        f"\nfmt={RENDER_FORMAT_VERSION}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def entity_content_key(
    memory_id: str,
    content_version: str,
    connections: list[dict],
) -> str:
    """Compute the content hash for a god-node entity article.

    Includes the node's content version and the sorted set of its connection tuples
    ``(target_id, relationship, strength)`` — the connections the entity article renders.

    Args:
        memory_id: The entity node's (stable) memory id.
        content_version: The node's content version string.
        connections: List of ``{"target_id", "relationship", "strength"}`` dicts.

    Returns:
        Hex sha256 digest string.
    """
    conn_tuples = sorted(
        f"{c.get('target_id', '')}|{c.get('relationship', '')}|{c.get('strength', '')}"
        for c in connections
    )
    # See community_content_key: fmt= makes a renderer change invalidate reuse.
    payload = (
        f"{memory_id}:{content_version}:fmt={RENDER_FORMAT_VERSION}:"
        + "\n".join(conn_tuples)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_prior_communities(analysis_json: dict | None) -> list[Community]:
    """Parse a cached ``GraphAnalysis`` JSON dict into prior ``Community`` models.

    Tolerant of a missing/empty cache (first ever run) -> returns an empty list.
    """
    if not analysis_json:
        return []
    raw_communities = analysis_json.get("communities") or []
    parsed: list[Community] = []
    for raw in raw_communities:
        try:
            parsed.append(Community(**raw))
        except (TypeError, ValueError):
            continue
    return parsed
