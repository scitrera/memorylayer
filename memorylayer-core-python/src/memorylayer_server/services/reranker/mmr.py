"""Maximal Marginal Relevance (MMR) result diversification.

A pure, greedy re-ordering that trades a little query-relevance for result-set
diversity, so recall does not spend its top-k budget on near-duplicate memories.
Adapted from Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking
for Reordering Documents and Producing Summaries" (SIGIR 1998); it mirrors the
redundancy penalty Microsoft Memora bakes into its retrieval reward
(``rl/trajectory_scorer.py``), but as a deterministic selection step rather than
a learned policy.

Kept dependency-light and side-effect free so it can be unit-tested in isolation
and reused by any recall path. The one dependency is the shared cosine helper.
"""

from __future__ import annotations

from ...utils import cosine_similarity


def mmr_select(
    relevance: list[float],
    embeddings: list[list[float] | None],
    lambda_param: float = 0.5,
    k: int | None = None,
) -> list[int]:
    """Greedy Maximal Marginal Relevance ordering.

    Returns the selected *original* indices, best-first. Each pick maximises::

        lambda * relevance[i] - (1 - lambda) * max_{j in selected} cos(i, j)

    Args:
        relevance: Query-relevance score per candidate (higher is better). Any
            real scale works; typically the post-boost recall score.
        embeddings: Per-candidate embedding, index-aligned with ``relevance``. A
            ``None`` entry disables the diversity term for that candidate (it is
            treated as maximally novel), so a partially-embedded pool degrades
            gracefully instead of raising.
        lambda_param: Balance in ``[0, 1]``. ``1.0`` is pure relevance (no
            diversification, i.e. a plain relevance sort); ``0.0`` is pure
            diversity. Values are clamped to the range.
        k: Number of items to select. ``None`` (or a value exceeding the pool)
            selects all candidates, yielding a full diversified ordering.

    Returns:
        List of selected indices into the input lists, in selection order.
    """
    n = len(relevance)
    if n == 0:
        return []
    lambda_param = max(0.0, min(1.0, lambda_param))
    if k is None or k > n:
        k = n

    selected: list[int] = []
    remaining = set(range(n))

    while remaining and len(selected) < k:
        best_idx: int | None = None
        best_score: float | None = None
        for i in remaining:
            if not selected:
                # First pick is the most relevant candidate (no diversity term
                # yet); ties resolve to the lowest index for determinism.
                score = relevance[i]
            else:
                emb_i = embeddings[i] if i < len(embeddings) else None
                if emb_i is None:
                    max_sim = 0.0
                else:
                    max_sim = 0.0
                    for j in selected:
                        emb_j = embeddings[j] if j < len(embeddings) else None
                        if emb_j is not None:
                            sim = cosine_similarity(emb_i, emb_j)
                            if sim > max_sim:
                                max_sim = sim
                score = lambda_param * relevance[i] - (1.0 - lambda_param) * max_sim
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        # best_idx is always assigned while remaining is non-empty.
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.discard(best_idx)

    return selected
