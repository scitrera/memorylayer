"""Pure, dependency-free retrieval metric functions.

These mirror the formulas used by gbrain's eval harness (MIT,
https://github.com/gtanai/gbrain — design ported, not code) so that
MemoryLayer's retrieval quality is scored with the same well-understood
definitions. Every function operates on plain lists of opaque string keys
(e.g. a corpus document's ``eval_key``); they are deliberately ignorant of
how those keys were produced so they can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant.

    Divides by ``k`` (the cutoff), not by the number actually returned, so a
    query that returns fewer than ``k`` results is penalized for the misses.
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items captured within the top-k results."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none found)."""
    if not retrieved or not relevant:
        return 0.0
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def first_relevant_hit(retrieved: list[str], relevant: set[str]) -> int:
    """1 if the top result is relevant, else 0."""
    if not retrieved or not relevant:
        return 0
    return 1 if retrieved[0] in relevant else 0


def expected_top1_hit(retrieved: list[str], expected_top1: str) -> int:
    """1 if the top result is exactly the expected item (stricter than relevance)."""
    if not retrieved:
        return 0
    return 1 if retrieved[0] == expected_top1 else 0


def ndcg_at_k(retrieved: list[str], grades: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain over graded relevance.

    ``grades`` maps key -> graded relevance (>= 0). Binary relevance is the
    special case where every relevant key has grade 1. Uses a log2 discount
    starting at rank 1 and normalizes by the ideal ordering of all graded items.
    """
    if k <= 0 or not retrieved or not grades:
        return 0.0

    dcg = 0.0
    for i, key in enumerate(retrieved[:k]):
        grade = grades.get(key, 0.0)
        if grade:
            dcg += grade / math.log2(i + 2)

    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero).

    Inlined rather than imported so this module stays dependency-free (see the
    module docstring); the definition matches ``utils.cosine_similarity``.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def redundancy(embeddings: list[list[float]]) -> float:
    """Mean pairwise cosine similarity of a result set's embeddings.

    A diversity penalty: higher means the returned memories are more alike (the
    result set wastes its top-k budget on near-duplicates), lower means the set
    covers more distinct facets. This is Memora's redundancy term
    (``rl/trajectory_scorer.py``) adapted as an offline eval metric — no RL.

    Returns 0.0 for fewer than two embeddings (no pair to compare). Negative
    cosines are clamped to 0 so the metric stays in ``[0, 1]`` for interpretable
    aggregation across queries.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += max(0.0, _cosine(embeddings[i], embeddings[j]))
            pairs += 1
    return total / pairs if pairs else 0.0


def retrieval_cost(steps: int, requeries: int = 0, requery_penalty: float = 0.2, max_steps: int = 5) -> float:
    """Normalized retrieval effort in ``[0, 1]`` (lower is cheaper).

    Mirrors Memora's cost term (``rl/trajectory_scorer.py``): a normalized step
    count plus a per-requery surcharge, since a RE_QUERY (a fresh search hop) is
    more expensive than an EXPAND (a cheap graph walk). Single-shot retrieval is
    ``steps=1, requeries=0``. Used to grade the agentic recall loop (Phase 2) on
    frugality, not just recall@k; ``max_steps`` normalizes to the loop's bound.

    The raw cost ``steps + requery_penalty * requeries`` is divided by
    ``max_steps`` and clamped to 1.0.
    """
    if max_steps <= 0:
        return 0.0
    raw = steps + requery_penalty * requeries
    return min(1.0, max(0.0, raw / max_steps))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Set-Jaccard overlap of two result sets (1.0 when both are empty)."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 if union == 0 else intersection / union


def mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
