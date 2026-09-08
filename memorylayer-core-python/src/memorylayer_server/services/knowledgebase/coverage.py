"""How much of the corpus the knowledgebase actually covers.

A generated KB is a FILTERED view of the association graph: communities below
``min_community_size`` are dropped, the rest are capped at ``max_communities``, and
central nodes are capped at ``max_god_nodes``. Every one of those cuts leaves
memories with no article to their name -- and nothing reported it, so a KB that
covered a third of a workspace looked exactly like one that covered all of it.

This is the blind-spot number: of the memories in the association graph, the share
that reached at least one article. Read it alongside ``dropped_*`` counts, which say
WHICH cut did the dropping, so a low ratio is actionable (raise a cap, lower the
min size) rather than merely alarming.

Pure and I/O-free: the caller supplies the covered id set and the totals.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CoverageReport", "compute_coverage"]


@dataclass
class CoverageReport:
    """Share of the association graph that reached an article."""

    graph_nodes: int = 0
    covered: int = 0
    dropped_small_communities: int = 0
    dropped_capped_communities: int = 0

    @property
    def uncovered(self) -> int:
        # Clamped: `covered` is a set of ids that reached an article, while
        # graph_nodes comes from the analysis stats. A god node or entity member can
        # legitimately sit outside the community partition the node count was taken
        # from, and a negative "uncovered" would be nonsense.
        return max(self.graph_nodes - self.covered, 0)

    @property
    def ratio(self) -> float:
        """Covered share in [0.0, 1.0]; 1.0 when there is nothing to cover."""
        if self.graph_nodes <= 0:
            return 1.0
        return min(self.covered / self.graph_nodes, 1.0)

    def as_metadata(self) -> dict:
        """Compact form for article metadata / quality dashboards."""
        return {
            "coverage_ratio": round(self.ratio, 4),
            "coverage_covered": self.covered,
            "coverage_graph_nodes": self.graph_nodes,
            "coverage_uncovered": self.uncovered,
            "coverage_dropped_small_communities": self.dropped_small_communities,
            "coverage_dropped_capped_communities": self.dropped_capped_communities,
        }


def compute_coverage(
    graph_nodes: int,
    covered_memory_ids: set[str],
    dropped_small_communities: int = 0,
    dropped_capped_communities: int = 0,
) -> CoverageReport:
    """Build a :class:`CoverageReport` from a run's covered-id set.

    Args:
        graph_nodes: Memories in the association graph (the denominator).
        covered_memory_ids: Every memory that reached at least one article --
            full community membership, not just the members rendered in the body.
        dropped_small_communities: Communities cut by ``min_community_size``.
        dropped_capped_communities: Communities cut by ``max_communities``.
    """
    return CoverageReport(
        graph_nodes=max(graph_nodes, 0),
        covered=len(covered_memory_ids),
        dropped_small_communities=dropped_small_communities,
        dropped_capped_communities=dropped_capped_communities,
    )
