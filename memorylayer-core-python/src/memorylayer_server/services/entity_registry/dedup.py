"""Entity dedup helpers — union-find clustering + representative selection.

Pure, backend-agnostic helpers for a batch dedup pass: group transitively-similar
entities into clusters (union-find), then pick the best-established representative
of each cluster to merge the rest into. The similarity/pair-finding is a backend
concern (the enterprise registry uses its pgvector ANN); these helpers own the
clustering + selection logic so it stays consistent and testable.
"""

from collections import defaultdict


class UnionFind:
    """Disjoint-set over hashable ids (path-compressed) for transitive grouping."""

    def __init__(self):
        self._parent: dict = {}

    def _find(self, x):
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    def add(self, x) -> None:
        self._parent.setdefault(x, x)

    def groups(self) -> list[list]:
        """All disjoint sets as lists (each id sorted for determinism)."""
        buckets: dict = defaultdict(list)
        for x in self._parent:
            buckets[self._find(x)].append(x)
        return [sorted(ids) for ids in buckets.values()]


def pick_representative(entities: list, member_counts: dict):
    """Choose the entity to merge a cluster INTO — the most established one.

    Ranked by: most members (most-referenced), then most aliases (most surface
    forms folded in), then highest confidence, then lowest id (stable tie-break).
    ``member_counts`` maps entity id -> member count. Returns the winning entity.
    """
    return sorted(
        entities,
        key=lambda e: (
            -member_counts.get(e.id, 0),
            -len(e.aliases or []),
            -float(getattr(e, "confidence", 1.0) or 0.0),
            e.id,
        ),
    )[0]
