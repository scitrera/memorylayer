"""Shared conservative token estimation and hard recall packing."""

from __future__ import annotations

from ...models.context_pack import BudgetSummary
from ...models.memory import DetailLevel, Memory

ESTIMATOR_NAME = "chars_div_3_conservative_v1"


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate used by deterministic APIs."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def _representations(memory: Memory, detail_level: DetailLevel) -> list[str]:
    values: list[str | None]
    if detail_level == DetailLevel.FULL:
        values = [memory.content, memory.overview, memory.abstract]
    elif detail_level == DetailLevel.OVERVIEW:
        values = [memory.overview, memory.abstract]
    else:
        values = [memory.abstract]
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def pack_recall_memories(
    memories: list[Memory],
    budget_tokens: int | None,
    detail_level: DetailLevel,
) -> tuple[list[Memory], BudgetSummary]:
    """Choose honest stored/extractive representations without exceeding budget."""
    if budget_tokens is None:
        used = sum(estimate_tokens(memory.content) for memory in memories)
        return memories, BudgetSummary(
            requested=None,
            used=used,
            estimator=ESTIMATOR_NAME,
        )
    packed: list[Memory] = []
    used = 0
    truncated = 0
    omitted = 0
    for memory in memories:
        remaining = budget_tokens - used
        if remaining <= 0:
            omitted += 1
            continue
        selected: str | None = None
        for representation in _representations(memory, detail_level):
            if estimate_tokens(representation) <= remaining:
                selected = representation
                break
        if selected is None:
            max_chars = remaining * 3
            if max_chars >= 24:
                source = memory.abstract or memory.overview or memory.content
                selected = source[:max_chars].rstrip()
                while selected and estimate_tokens(selected) > remaining:
                    selected = selected[:-1]
                if selected:
                    truncated += 1
        if not selected:
            omitted += 1
            continue
        selected_tokens = estimate_tokens(selected)
        if selected != memory.content:
            memory = memory.model_copy(deep=True, update={"content": selected})
        packed.append(memory)
        used += selected_tokens
    return packed, BudgetSummary(
        requested=budget_tokens,
        used=used,
        estimator=ESTIMATOR_NAME,
        truncated_items=truncated,
        omitted_items=omitted,
    )
