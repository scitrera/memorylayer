"""Entity provenance/lineage summarization (PROV-O-flavored).

Turns an entity's ad-hoc ``provenance`` dict — populated with different keys by
different producers (resolve/create, name-first promotion, seed catalog, external-KB
enrichment) — into one normalized, auditable ``EntityProvenance`` view. Pure and
I/O-free; tolerant of any shape (missing/unknown keys degrade gracefully).
"""

from ...models.entity_registry import Entity, EntityProvenance

# The agent that records provenance by default (PROV-O ``wasAttributedTo``).
DEFAULT_PROVENANCE_AGENT = "memorylayer"


def build_provenance(
    activity: str,
    *,
    agent: str = DEFAULT_PROVENANCE_AGENT,
    source: str | None = None,
    generated_at: str | None = None,
    **extra,
) -> dict:
    """Construct a normalized (PROV-O-flavored) provenance dict for an entity writer.

    Every provenance record carries a consistent core — ``activity`` (what
    happened, e.g. ``entity.create`` / ``entity.seed``), ``agent`` (who), and
    optionally ``source`` + ``generated_at`` — while producer-specific keys pass
    through via ``extra`` (``None`` values dropped). This unifies the SHAPE across
    producers without discarding the keys existing logic reads (``matched_via``,
    ``promoted_from``, ``fuzzy_candidates``, …).
    """
    prov: dict = {"activity": activity, "agent": agent}
    if source is not None:
        prov["source"] = source
    if generated_at is not None:
        prov["generated_at"] = generated_at
    for key, value in extra.items():
        if value is not None:
            prov[key] = value
    return prov


def summarize_entity_provenance(entity: Entity) -> EntityProvenance:
    """Normalize an entity's raw provenance into an auditable lineage view."""
    prov = dict(entity.provenance or {})

    # Origin: how the entity came to exist. Seed and promotion are explicit; the
    # matched_via key (resolve/create) distinguishes a fresh create from a match.
    if prov.get("source") == "seed":
        origin = "seeded"
    elif prov.get("promoted_from"):
        origin = "promoted"
    elif prov.get("matched_via") == "created":
        origin = "created"
    elif prov.get("matched_via") in ("exact", "alias", "embedding"):
        origin = "matched"
    else:
        origin = "unknown"

    external_links = dict(prov.get("external_links") or {})

    # Description: prefer an explicit seed description, else the first external link's.
    description = prov.get("description")
    if not description and external_links:
        first = next(iter(external_links.values()), {}) or {}
        description = first.get("description")

    # Source memories that first surfaced this entity (from any producer's keys).
    candidate_ids = [
        prov.get("source_memory_id"),
        prov.get("promoted_source_memory_id"),
        entity.representative_memory_id,
    ]
    source_memory_ids = sorted({mid for mid in candidate_ids if mid})

    generated_at = prov.get("created_at") or prov.get("generated_at")

    return EntityProvenance(
        entity_id=entity.id,
        origin=origin,
        activity=prov.get("activity"),
        agent=prov.get("agent"),
        description=description,
        external_links=external_links,
        source_memory_ids=source_memory_ids,
        generated_at=generated_at,
        raw=prov,
    )
