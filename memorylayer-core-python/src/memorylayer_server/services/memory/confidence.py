"""Deterministic retrieval-confidence calculation."""

from __future__ import annotations

from ...models.memory import Memory


def retrieval_confidence(
    query: str,
    memories: list[Memory],
    *,
    unresolved_entity: bool = False,
) -> tuple[str, list[str]]:
    """Describe retrieval evidence, never the truth of recalled content."""
    if not memories:
        return "weak", ["empty_result"]
    top = memories[0]
    signals = set(top.match_signals or [])
    normalized_query = query.strip().casefold()
    if normalized_query in {top.id.casefold(), (top.logical_key or "").casefold()}:
        return "strong", ["exact_identifier_match"]
    if "exact" in signals or "alias" in signals:
        return "strong", ["exact_entity_or_alias_match"]
    independent = signals & {"keyword", "vector", "alias", "relational"}
    if len(independent) >= 2:
        return "strong", ["multiple_independent_retrieval_signals"]
    if "relational" in signals:
        return "moderate", ["typed_relation_with_active_evidence"]
    if "keyword" in signals:
        return "moderate", ["strong_keyword_evidence"]
    reasons: list[str] = []
    if unresolved_entity:
        reasons.append("unresolved_query_entity")
    if signals == {"vector"} or not signals:
        reasons.append("uncalibrated_vector_only")
    scores = [score for score in (getattr(memory, "boosted_score", None) for memory in memories[:2]) if score is not None]
    if len(scores) > 1 and scores[0] - scores[1] < 0.03:
        reasons.append("low_fused_margin")
    if top.trust_score is not None and top.trust_score < 0.5:
        reasons.append("low_trust_candidate")
    return "weak", sorted(set(reasons or ["weak_retrieval_evidence"]))
