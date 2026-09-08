# Design: Unifying Contradiction Detection and Auto-Association

**Status:** Draft / proposal — needs sign-off before implementation.
**Scope:** OSS core (`services/contradiction`, `services/association`, `services/ontology`, `services/memory`), with automatic pickup by the enterprise AGE graph materializer.

## Problem

Two independent subsystems detect the same "these two memories conflict"
relationship, on the same trigger, over the same candidate set, and never
reconcile:

1. **`ContradictionService.check_new_memory`** (`services/contradiction/default.py`)
   — deterministic (negation word-pairs + regex value-triples), writes
   `ContradictionRecord`s to a standalone `contradictions` table, and owns the
   only **resolution** machinery (`resolve()` → soft-delete / merge / keep).
2. **`AssociationService.auto_associate` → `classify_relationship`**
   (`services/association/default.py`, `services/ontology/default.py`) — the LLM
   picks from the ~60-type ontology and can emit a `contradicts` (or
   `supersedes`) **graph edge** into `memory_associations`.

Both run in the same `MemoryService._post_store_pipeline`
(`services/memory/default.py:1103` and `:1112`), and **each issues its own
`search_memories(..., min_relevance=0.7 / 0.6)`** KNN over the same neighborhood.

Consequences:

- **Duplicated KNN** per stored memory (two similar-memory searches).
- **Split truth:** an LLM `contradicts` edge is invisible to the contradiction
  table's `get_unresolved` / `resolve` flow; a deterministic contradiction
  record never becomes a graph edge. A contradiction can exist in one system and
  not the other.
- **Only one path can act:** resolution (soft-delete / merge) operates solely on
  table records. LLM-detected `contradicts` / `supersedes` edges are inert.
- The **cross-encoder ontology provider** (an optional NLI-based `OntologyService`
  implementation supplied by the enterprise distribution)
  already sidesteps this by NOT emitting `contradicts` — but that is a
  workaround, not a resolution. The NLI cross-encoder is in fact the natural
  place to *converge* the two detectors.

## Principle

**One contradiction authority.** `ContradictionService` detects and records
contradictions and supersessions; the association graph is a *sink* that mirrors
those decisions as edges so graph consumers (traversal, backlink boost, AGE) see
the same truth. The relationship classifier gets out of the contradiction
business entirely.

This keeps the deterministic cheap path, keeps resolution in one place, and
gives a single seam to upgrade detection quality (regex → NLI cross-encoder)
without touching the sink contract.

## Target architecture

```
                   post-store pipeline (per new memory)
                              │
                shared KNN candidate neighborhood
                     ┌────────┴─────────┐
                     ▼                  ▼
        ContradictionService     AssociationService.auto_associate
        (detect: regex→NLI)      (type NON-contradiction edges only:
             │                    duplicate_of / similar_to / supports / …)
     ┌───────┴────────┐                    │
     ▼                ▼                     ▼
 contradictions   mirror edge        memory_associations
 table (resolve)  contradicts /       (similarity + typed graph)
                  supersedes ────────────► same table, tagged
                                            {source: "contradiction",
                                             contradiction_id: …}
```

- The **classifier's output menu excludes** the `learning`-category conflict
  types `contradicts`, `supersedes`, `superseded_by` (and the `refinement`
  `replaces`/`replaced_by` overlap). Those are produced only by the
  contradiction authority. The LLM classifier maps them to `related_to` if the
  model returns them; the cross-encoder already declines to emit `contradicts`.
- On `create_contradiction`, the contradiction service **also writes a
  `contradicts` edge** (symmetric) into `memory_associations`, tagged
  `metadata={"source": "contradiction", "contradiction_id": <id>,
  "detection_method": <m>}`. When `newer_memory_id` is set (temporal
  supersession), additionally write a directional **`supersedes`** edge
  newer → older. Idempotent via the existing `ON CONFLICT DO NOTHING`.
- **Resolution** can then act on either representation: `resolve()` continues to
  work on the record and, via the `contradiction_id` backref, can drop/annotate
  the mirrored edge.

## Phasing (each phase independently shippable, dark-safe)

**Phase 0 — stop the double-write (no new behavior).**
Remove the classifier's ability to emit conflict edges: filter `contradicts` /
`supersedes` / `superseded_by` (and optionally `replaces`/`replaced_by`) out of
the type menu built in `classify_relationship` / `classify_relationships_batch`,
mapping any such model output to `related_to`. Net effect: the graph loses
unresolvable LLM `contradicts` edges (which no consumer could act on anyway).
Contradiction detection is unchanged. **This is the safe first commit and
removes the duplication of *effort* immediately.**

**Phase 1 — mirror records to edges.**
In `ContradictionService`, after `create_contradiction`, write the
`contradicts` (and temporal `supersedes`) edge via the association/storage
layer, tagged with `contradiction_id`. Graph traversal and backlink boost now
reflect real contradictions. (Note: the backlink boost already *excludes*
`similar_to` and *counts* `contradicts` — so mirrored edges start feeding
hub-salience; confirm that's desired, likely yes.)

**Phase 2 — share the candidate KNN.**
`_post_store_pipeline` computes the similar-memory neighborhood ONCE and passes
it to both the contradiction check and `auto_associate`, removing the second
`search_memories`. Requires widening `check_new_memory` to accept an optional
pre-fetched candidate list (fall back to its own search when absent, preserving
the standalone task/scan paths).

**Phase 3 — upgrade the detector to the cross-encoder.**
Add an NLI-backed `ContradictionService` provider that reuses the same endpoint
as the `cross_encoder` ontology provider: `contradiction` label with score ≥
threshold → `ContradictionRecord(detection_method="nli_cross_encoder")`. Replaces
the negation-pair / regex heuristics with a real model while keeping the record +
resolution + mirror contract. This is where the (b) cross-encoder work converges:
**one NLI call per new memory feeds both coarse relationship typing and
contradiction detection** over the shared candidate set.

**Phase 4 — resolution over edges (optional).**
Expose resolution that starts from a graph edge (via `contradiction_id`), and
have `resolve()` annotate/soft-delete the mirrored edge so a resolved
contradiction visibly disappears from graph consumers.

## Open questions / risks

- **`supersedes` semantics need recency, not just entailment.** NLI gives
  entailment/contradiction; "which is newer" comes from `event_time` /
  `created_at` (`_determine_newer_memory` already exists). Supersession = signal
  + temporal order. Keep that combination in the contradiction authority, not the
  content-only classifier.
- **Symmetric vs directional.** `contradicts` is symmetric; `supersedes` is
  directional (newer → older). The mirror writer must set direction from
  `newer_memory_id`.
- **Backlink in-degree.** Mirroring contradiction edges will raise the target's
  in-degree and thus its salience boost. Probably desirable (contested memories
  are salient) but should be validated against the retrieval eval
  (`benchmarks/assoc_typed_edge_eval.py`) before enabling by default.
- **Enterprise AGE.** The materializer projects `memory_associations` rows into
  AGE automatically, so mirrored edges appear in Cypher with no extra work.
- **Does typed conflict edge value survive eval?** Gate Phase 1's default-on
  behind the typed-edge eval on a discriminating corpus (LoCoMo / LongMemEval).
  Graph typing stays flag-gated dark until it demonstrably moves retrieval.

## Non-goals

- Rewriting the resolution strategies or the `contradictions` table schema.
- Changing recall's contradiction-aware ranking (separate concern).
- Forcing the cross-encoder on by default (stays opt-in until eval-justified).

## Related

- Cost-reduction batching/deterministic gating: committed
  `perf(association): batch + deterministically gate ...`.
- Cross-encoder ontology provider: an optional NLI-based `OntologyService`
  implementation supplied by the enterprise distribution.
- Typed-edge eval harness: `benchmarks/assoc_typed_edge_eval.py`.
- Type-sensitive retrieval: `MemoryService.apply_backlink_boost` (excludes `similar_to`).
