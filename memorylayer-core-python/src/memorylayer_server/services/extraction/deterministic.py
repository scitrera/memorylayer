"""Pure deterministic classification, segmentation, and narrow extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...models.memory import MemoryType


@dataclass(frozen=True)
class ClassificationResult:
    memory_type: MemoryType
    subtype: str | None
    confidence: str
    matched_rules: tuple[str, ...]
    explicitly_classified: bool


@dataclass(frozen=True)
class DeterministicSegment:
    content: str
    source_span_start: int
    source_span_end: int
    method: str
    memory_type: MemoryType = MemoryType.EPISODIC
    subtype: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractiveTier:
    content: str
    source_spans: tuple[tuple[int, int], ...]
    method: str = "extractive"


@dataclass(frozen=True)
class MarkerCandidate:
    kind: str
    content: str
    source_span_start: int
    source_span_end: int
    rule_id: str


_PROCEDURAL = re.compile(r"\b(how to|steps?|procedure|process|method|workflow|run|execute)\b", re.I)
_EPISODIC = re.compile(r"\b(yesterday|today|tomorrow|occurred|happened|we (?:met|decided|fixed)|at that time)\b", re.I)
_WORKING = re.compile(r"\b(currently|working on|in progress|right now|blocked|next(?: step)?)\b", re.I)
_DIRECTIVE = re.compile(r"\b(always|never|must|prefer|do not|don't)\b", re.I)
_MARKER = re.compile(r"(?im)^\s*(DECISION|TODO|NEXT|BLOCKED|RESOLVED)\s*[:\-]\s*(.+?)\s*$")
_HEADING_OR_LIST = re.compile(r"(?m)^(?:#{1,6}\s+.+|\s*(?:[-*+] |\d+[.)]\s+).+)$")
_CHAT_LINE = re.compile(r"(?m)^\s*(?:user|assistant|system|tool|human|agent)\s*:\s*.+$", re.I)
_SENTENCE = re.compile(r"[^\n.!?]+(?:[.!?]+(?=\s|$)|$)")


def classify_content(
    content: str,
    *,
    explicit_type: MemoryType | None = None,
    explicit_subtype: str | None = None,
) -> ClassificationResult:
    """Classify without generation; explicit caller classification always wins."""
    if explicit_type is not None:
        return ClassificationResult(
            explicit_type,
            explicit_subtype,
            "strong",
            ("explicit_type",),
            True,
        )
    rules: list[tuple[str, MemoryType, re.Pattern[str]]] = [
        ("procedural_language", MemoryType.PROCEDURAL, _PROCEDURAL),
        ("working_state", MemoryType.WORKING, _WORKING),
        ("episodic_time_or_event", MemoryType.EPISODIC, _EPISODIC),
    ]
    matches = [(rule, memory_type) for rule, memory_type, pattern in rules if pattern.search(content)]
    if not matches:
        return ClassificationResult(
            MemoryType.EPISODIC,
            explicit_subtype,
            "weak",
            ("neutral_episodic_fallback",),
            False,
        )
    selected = matches[0][1]
    matched_rules = tuple(rule for rule, memory_type in matches if memory_type == selected)
    confidence = "strong" if len(matched_rules) > 1 else "moderate"
    return ClassificationResult(selected, explicit_subtype, confidence, matched_rules, False)


def _bounded_chunks(content: str, start: int, max_chars: int, overlap_chars: int) -> list[DeterministicSegment]:
    chunks: list[DeterministicSegment] = []
    cursor = 0
    while cursor < len(content):
        end = min(len(content), cursor + max_chars)
        if end < len(content):
            boundary = content.rfind(" ", cursor + max_chars // 2, end)
            if boundary > cursor:
                end = boundary
        text = content[cursor:end].strip()
        if text:
            local = content.find(text, cursor, end + 1)
            chunks.append(
                DeterministicSegment(
                    text,
                    start + local,
                    start + local + len(text),
                    "bounded_character_split",
                )
            )
        if end >= len(content):
            break
        cursor = max(cursor + 1, end - overlap_chars)
    return chunks


def segment_content(
    content: str,
    *,
    structured_fields: list[tuple[str, str]] | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 80,
) -> list[DeterministicSegment]:
    """Split content on authoritative boundaries while preserving exact spans."""
    if not content.strip():
        return []
    candidates: list[tuple[int, int, str]] = []
    if structured_fields:
        search_from = 0
        for name, value in structured_fields:
            if not value.strip():
                continue
            pos = content.find(value, search_from)
            if pos < 0:
                pos = content.find(value)
            if pos >= 0:
                candidates.append((pos, pos + len(value), f"structured_field:{name}"))
                search_from = pos + len(value)
    if not candidates:
        chat = list(_CHAT_LINE.finditer(content))
        if chat:
            candidates = [(m.start(), m.end(), "chat_boundary") for m in chat]
    if not candidates:
        structural = list(_HEADING_OR_LIST.finditer(content))
        if structural:
            starts = [m.start() for m in structural]
            starts.append(len(content))
            candidates = [(starts[index], starts[index + 1], "heading_or_list") for index in range(len(starts) - 1)]
    if not candidates:
        cursor = 0
        for block in re.split(r"\n\s*\n", content):
            if not block.strip():
                cursor += len(block)
                continue
            start = content.find(block, cursor)
            candidates.append((start, start + len(block), "paragraph"))
            cursor = start + len(block)
    segments: list[DeterministicSegment] = []
    for start, end, method in candidates:
        raw = content[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        adjusted_start = start + raw.find(stripped)
        if len(stripped) <= max_chars:
            segments.append(DeterministicSegment(stripped, adjusted_start, adjusted_start + len(stripped), method))
            continue
        sentence_matches = [m for m in _SENTENCE.finditer(stripped) if m.group(0).strip()]
        if len(sentence_matches) > 1:
            group_start = sentence_matches[0].start()
            group_end = group_start
            for match in sentence_matches:
                if match.end() - group_start > max_chars and group_end > group_start:
                    text = stripped[group_start:group_end].strip()
                    local = stripped.find(text, group_start, group_end + 1)
                    segments.append(
                        DeterministicSegment(
                            text,
                            adjusted_start + local,
                            adjusted_start + local + len(text),
                            "sentence_group",
                        )
                    )
                    group_start = max(match.start(), group_end - overlap_chars)
                group_end = match.end()
            text = stripped[group_start:group_end].strip()
            if text:
                local = stripped.find(text, group_start, group_end + 1)
                segments.append(
                    DeterministicSegment(
                        text,
                        adjusted_start + local,
                        adjusted_start + local + len(text),
                        "sentence_group",
                    )
                )
        else:
            segments.extend(_bounded_chunks(stripped, adjusted_start, max_chars, overlap_chars))
    unique: dict[tuple[str, int, int], DeterministicSegment] = {}
    for segment in segments:
        normalized = " ".join(segment.content.casefold().split())
        unique[(normalized, segment.source_span_start, segment.source_span_end)] = segment
    return list(unique.values())


def extract_marker_candidates(content: str) -> list[MarkerCandidate]:
    """Extract only explicit markers and bounded directive statements."""
    candidates = [
        MarkerCandidate(
            kind=match.group(1).lower(),
            content=match.group(2).strip(),
            source_span_start=match.start(2),
            source_span_end=match.end(2),
            rule_id=f"marker_{match.group(1).lower()}",
        )
        for match in _MARKER.finditer(content)
    ]
    for match in _SENTENCE.finditer(content):
        sentence = match.group(0).strip()
        if sentence and _DIRECTIVE.search(sentence):
            start = match.start() + match.group(0).find(sentence)
            candidates.append(MarkerCandidate("directive", sentence, start, start + len(sentence), "imperative_directive"))
    candidates.sort(key=lambda item: (item.source_span_start, item.source_span_end, item.kind))
    return candidates


def extractive_tiers(
    content: str,
    *,
    abstract_chars: int = 200,
    overview_chars: int = 500,
) -> tuple[ExtractiveTier, ExtractiveTier]:
    """Select complete source sentences for bounded abstract and overview tiers."""
    if len(content) <= abstract_chars:
        tier = ExtractiveTier(content, ((0, len(content)),), "source_reuse")
        return tier, tier
    sentences: list[tuple[str, int, int]] = []
    for match in _SENTENCE.finditer(content):
        sentence = match.group(0).strip()
        if sentence:
            start = match.start() + match.group(0).find(sentence)
            sentences.append((sentence, start, start + len(sentence)))
    if not sentences:
        abstract = content[:abstract_chars].rstrip()
        overview = content[:overview_chars].rstrip()
        return (
            ExtractiveTier(abstract, ((0, len(abstract)),), "bounded_source_excerpt"),
            ExtractiveTier(overview, ((0, len(overview)),), "bounded_source_excerpt"),
        )
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", content.casefold())
    frequencies: dict[str, int] = {}
    for term in terms:
        frequencies[term] = frequencies.get(term, 0) + 1
    scored = []
    for index, (sentence, start, end) in enumerate(sentences):
        sentence_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", sentence.casefold()))
        repeated = sum(max(0, frequencies.get(term, 0) - 1) for term in sentence_terms)
        heading = 2.0 if start == 0 or content[max(0, start - 2) : start].endswith("\n\n") else 0.0
        position = 3.0 / (index + 1)
        length_penalty = max(0.0, (len(sentence) - 320) / 160)
        scored.append((heading + position + repeated * 0.2 - length_penalty, index))
    order = [index for _score, index in sorted(scored, key=lambda item: (-item[0], item[1]))]

    def select(limit: int) -> ExtractiveTier:
        chosen: list[int] = []
        used = 0
        for index in order:
            sentence = sentences[index][0]
            additional = len(sentence) + (1 if chosen else 0)
            if used + additional <= limit:
                chosen.append(index)
                used += additional
        if not chosen:
            sentence, start, end = sentences[0]
            excerpt = sentence[:limit].rstrip()
            return ExtractiveTier(excerpt, ((start, start + len(excerpt)),), "bounded_source_excerpt")
        chosen.sort()
        return ExtractiveTier(
            " ".join(sentences[index][0] for index in chosen),
            tuple((sentences[index][1], sentences[index][2]) for index in chosen),
        )

    overview = ExtractiveTier(content, ((0, len(content)),), "source_reuse") if len(content) <= overview_chars else select(overview_chars)
    return select(abstract_chars), overview
