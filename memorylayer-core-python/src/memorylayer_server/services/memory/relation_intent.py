"""Bounded pure relation-query intent classification."""

from __future__ import annotations

import re

from ...models.entity_relation import RelationIntent

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...], str], ...] = (
    (
        re.compile(r"\b(?:work(?:s|ed)? (?:at|for)|employ(?:s|ed|ment)|employer|found(?:ed|er|ers|ing))\b", re.I),
        ("employed_by", "employs", "founded", "founded_by"),
        "both",
    ),
    (re.compile(r"\b(?:invest(?:s|ed|ment)? in|investor)\b", re.I), ("invested_in", "investor_of"), "both"),
    (re.compile(r"\b(?:advis(?:e|es|ed|er|or|ory)(?: at| to)?)\b", re.I), ("advises", "advised_by"), "both"),
    (re.compile(r"\b(?:attend(?:s|ed|ance|ee|ees)?)\b", re.I), ("attended", "attended_by"), "both"),
    (re.compile(r"\b(?:own(?:s|ed|ership)?|belongs to)\b", re.I), ("owns", "owned_by"), "both"),
    (
        re.compile(r"\b(?:responsible for|assigned to|assignee|assignment)\b", re.I),
        ("responsible_for", "assigned_to"),
        "both",
    ),
    (re.compile(r"\b(?:author(?:ed|ship)?|wrote|written by)\b", re.I), ("authored", "authored_by"), "both"),
    (re.compile(r"\b(?:contribut(?:e|es|ed|or|ors|ion|ions) to)\b", re.I), ("contributed_to", "has_contributor"), "both"),
    (re.compile(r"\b(?:review(?:s|ed|er|ers)?|reviewed by)\b", re.I), ("reviewed", "reviewed_by"), "both"),
    (re.compile(r"\b(?:approv(?:e|es|ed|al)|approved by)\b", re.I), ("approved", "approved_by"), "both"),
    (re.compile(r"\b(?:decid(?:e|es|ed)|decision made by|decided by)\b", re.I), ("decided", "decided_by"), "both"),
    (re.compile(r"\b(?:member of|membership|belongs to)\b", re.I), ("member_of", "has_member"), "both"),
    (re.compile(r"\b(?:parent|child|mother|father|sibling|family)\b", re.I), ("parent_of", "child_of", "sibling_of"), "both"),
    (re.compile(r"\b(?:located in|based in|location|headquartered)\b", re.I), ("located_in", "location_of"), "both"),
    (re.compile(r"\b(?:depends? on|dependency|requires?|required by)\b", re.I), ("depends_on", "dependency_of"), "both"),
    (re.compile(r"\b(?:blocks?|blocked by)\b", re.I), ("blocks", "blocked_by"), "both"),
    (re.compile(r"\b(?:references?|referenced by|cites?|cited by)\b", re.I), ("references", "referenced_by"), "both"),
    (
        re.compile(r"\b(?:supersed(?:e|es|ed)|superseded by|replac(?:e|es|ed)|replaced by)\b", re.I),
        ("supersedes", "superseded_by"),
        "both",
    ),
    (re.compile(r"\b(?:based on|basis for)\b", re.I), ("based_on", "basis_for"), "both"),
    (re.compile(r"\b(?:affects?|affected by|impact(?:s|ed)?)\b", re.I), ("affects", "affected_by"), "both"),
    (re.compile(r"\b(?:supports?|supported by)\b", re.I), ("supports", "supported_by"), "both"),
    (re.compile(r"\b(?:contradicts?|contradicted by)\b", re.I), ("contradicts",), "both"),
    (re.compile(r"^\s*(?:what|which)\s+.+?\s+(?:is\s+)?about\b", re.I), ("about", "subject_of"), "both"),
    (re.compile(r"\b(?:part of|belongs to (?:the )?(?:project|program|portfolio|organization|team))\b", re.I), ("part_of", "has_part"), "both"),
    (
        re.compile(r"\b(?:sent (?:an? )?(?:email|message)? to|received (?:an? )?(?:email|message)? from)\b", re.I),
        ("sent_to", "received_from"),
        "both",
    ),
    (re.compile(r"\b(?:connected|related) to\b", re.I), (), "both"),
)
_CAPITALIZED = re.compile(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*){0,3}\b")
_QUOTED = re.compile(r"['\"]([^'\"]{2,80})['\"]")
_RELATION_SEEDS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:who|what|which)\s+(?:works? (?:at|for)|(?:has )?invested in|advises?|attended|founded|owns?|"
        r"authored|wrote|contributed to|reviewed|approved|decided)\s+(.+?)\??\s*$",
        re.I,
    ),
    re.compile(r"^\s*who\s+(?:is\s+)?(?:responsible for|assigned to|a member of|member of)\s+(.+?)\??\s*$", re.I),
    re.compile(r"^\s*(?:what|which)\s+(?:project|program|portfolio|organization|team)\s+is\s+(.+?)\s+part of\??\s*$", re.I),
    re.compile(
        r"^\s*(?:what|which)\s+does\s+(.+?)\s+(?:depend on|block|reference|cite|supersede|replace|affect|impact|support|contradict)\??\s*$",
        re.I,
    ),
    re.compile(r"^\s*(?:what|which)\s+is\s+(.+?)\s+(?:based on|part of|about)\??\s*$", re.I),
    re.compile(
        r"^\s*(?:what|which)\s+(?:is\s+)?(?:blocked by|supports?|contradicts?|supersed(?:e|es|ed)|replac(?:e|es|ed)|references?)\s+(.+?)\??\s*$",
        re.I,
    ),
)


def classify_relation_intent(query: str) -> RelationIntent:
    relationships: list[str] = []
    direction = "both"
    matches = 0
    for pattern, types, pattern_direction in _RELATION_PATTERNS:
        if pattern.search(query):
            matches += 1
            relationships.extend(types)
            direction = pattern_direction
    specific_seed = next((pattern.match(query) for pattern in _RELATION_SEEDS if pattern.match(query)), None)
    seeds = ([specific_seed.group(1).strip()] if specific_seed else []) + _QUOTED.findall(query) + _CAPITALIZED.findall(query)
    seeds = [seed.strip() for seed in seeds if seed.casefold() not in {"who", "what", "where", "which"}]
    return RelationIntent(
        is_relation_query=matches > 0,
        confidence=min(1.0, 0.65 + 0.15 * matches) if matches else 0.0,
        seed_phrases=list(dict.fromkeys(seeds))[:4],
        relationship_types=list(dict.fromkeys(relationships)),
        direction=direction,
        max_hops=1,
    )
