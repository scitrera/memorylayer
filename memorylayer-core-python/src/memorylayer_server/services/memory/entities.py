"""Lightweight regex entity extraction for the entity-anchored retrieval channel.

Dialogue memories carry a perfectly-reliable structural entity — the *speaker*
(``[timestamp] Name:``). Proper-noun spans (capitalized tokens) provide the
mention set. Both are folded into memory metadata at ingest and matched against
query entities to drive a rank-disciplined RRF fusion channel in recall.

This is a no-LLM, no-embed extractor (text-only), validated offline on LoCoMo
(``.slop/kg_retrieval_lab.py``). Speaker-anchored fusion lifts recall@5 over the
pure-vector baseline without crowding (RRF discipline). See
``.slop/INCREMENTAL_KG_DESIGN.md`` for the design rationale.
"""
import re

# Speaker of a dialogue turn: ``[timestamp] Name:`` at line start.
SPEAKER_RE = re.compile(r"^\[[^\]]*\]\s*([A-Z][a-zA-Z]+):")
# Proper-noun spans: capitalized tokens of length >= 3.
CAP_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
# Question / stop words to strip from query entity extraction so interrogatives
# and common capitalized leading words don't masquerade as entities.
QWORDS = {
    "What", "When", "Where", "Who", "Why", "How", "Which", "Did", "Does", "Do", "Is",
    "Are", "Was", "Were", "Will", "Would", "Could", "Should", "The", "A", "An", "In",
    "On", "At", "Of", "To", "For", "And", "Or", "But", "Her", "His", "She", "He", "They",
    "I", "We", "You", "It", "That", "This", "There", "Their",
}


def extract_entities(content: str) -> dict:
    """Extract the speaker and proper-noun entity set from a memory's content.

    Returns ``{"speaker": str | None, "entities": list[str], "entity_types":
    dict[str, str]}`` where ``entities`` is the sorted set of capitalized
    proper-noun spans (the speaker, when present, is always included) and
    ``entity_types`` maps each entity name to an ``EntityType`` value
    (``models/entity_registry.py``). The regex extractor cannot infer types, so
    ``entity_types`` is always ``{}`` here — typed providers (e.g. GLiNER2 NER)
    populate it; downstream accretion falls back to the
    speaker-is-PERSON / else-CONCEPT heuristic when a name is absent from the
    map. The speaker is parsed structurally from a leading ``[timestamp] Name:``
    prefix; the proper-noun spans are parsed from the body (the timestamp/speaker
    prefix is stripped first so it isn't double-counted).
    """
    if not content:
        return {"speaker": None, "entities": [], "entity_types": {}}

    m = SPEAKER_RE.match(content)
    speaker = m.group(1) if m else None

    body = re.sub(r"^\[[^\]]*\]\s*", "", content)
    ents = set(CAP_RE.findall(body))
    if speaker:
        ents.add(speaker)

    return {"speaker": speaker, "entities": sorted(ents), "entity_types": {}}


def extract_query_entities(query: str) -> list[str]:
    """Extract candidate entity tokens from a query.

    Capitalized proper-noun spans minus the question/stop-word set (``QWORDS``).
    Returns a sorted list so callers get a deterministic ordering.
    """
    if not query:
        return []
    return sorted({t for t in CAP_RE.findall(query) if t not in QWORDS})
