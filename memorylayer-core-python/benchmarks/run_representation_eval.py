"""End-to-end eval for ``RepresentationService.get_representation(observer, subject)``.

This drives the REAL pipeline — ingest -> entity-registry accretion (self/mention
member roles) -> registry resolution -> deterministic representation assembly —
on a small, hand-built, multi-observer dialogue corpus, then scores precision /
recall / leakage / attribution per (observer, subject) pair.

Why this exists ("eval before believing"): the unit test
(``tests/unit/test_representation_service.py``) proves the scoped-intersection
contract against in-memory fakes. THIS proves the same property survives the real
accretion path: that role assignment (speaker -> self, other proper-noun spans ->
mention), real resolution, and the intersection assembly together return the
OBSERVER's perspective on the SUBJECT and EXCLUDE other observers' statements
about that same subject (leakage ~= 0).

Backend: SQLite + the deterministic ``hash`` embedding provider (no PG / GPU /
LLM). Fully deterministic + CI-stable. Representation assembly never calls
``recall()`` and never embeds — the ``hash`` provider here is only what the
ingest path requires; it does not influence the assembled scope.

Run::

    .venv/bin/python benchmarks/run_representation_eval.py

Exits 0 when the gate passes (overall leakage == 0 AND mean recall >= 0.99 AND
attribution == 1.0), 1 otherwise. The importable ``run()`` coroutine returns the
structured results so ``tests/integration/test_representation_eval.py`` can assert
the same gate as a regression guard.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from scitrera_app_framework import get_extension

from memorylayer_server.eval.harness import _EVAL_WORKSPACE_ID, _eval_metadata, bootstrap
from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.models.memory import MemoryType, RememberInput
from memorylayer_server.services.entity_registry import EXT_ENTITY_REGISTRY_SERVICE
from memorylayer_server.services.representation import get_representation_service
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

# Gate thresholds. Leakage MUST be exactly 0 (the headline property). Recall is
# held near-perfect because the corpus is tiny and exact; attribution must be
# total (every returned observation authored by the requested observer).
_GATE_MAX_LEAKAGE = 0.0
_GATE_MIN_RECALL = 0.99
_GATE_MIN_ATTRIBUTION = 1.0


# --------------------------------------------------------------------------- #
# Scripted multi-observer corpus.
#
# Format: ``[ts] Speaker: body`` so the speaker-regex + accretion treat the
# speaker as role="self" and every other capitalized proper-noun span (>=3 chars)
# in the body as role="mention". Bodies are otherwise lowercase so the only
# PERSON entities are the three speakers (extra CONCEPT spans would not affect the
# self/mention intersection, but keeping bodies clean keeps the gold obvious).
#
# Each turn has a stable ``key``; we record the gold mapping (which keys are the
# observer's statements about the subject) right alongside the turns below.
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    key: str
    content: str


CORPUS: list[Turn] = [
    # --- Alice ABOUT Bob (gold for (Alice, Bob)) --------------------------- #
    Turn("a_bob_1", "[2026-01-02 09:00] Alice: Bob is reliable and ships features on time."),
    Turn("a_bob_2", "[2026-01-03 09:00] Alice: Bob reviewed my pull request carefully yesterday."),
    Turn("a_bob_3", "[2026-01-04 09:00] Alice: Bob mentored two junior developers this quarter."),
    # --- Carol ABOUT Bob (DISTRACTORS for (Alice, Bob); leakage guard) ----- #
    Turn("c_bob_1", "[2026-01-02 11:00] Carol: Bob often misses deadlines on the billing work."),
    Turn("c_bob_2", "[2026-01-03 11:00] Carol: Bob skipped the standup meeting again today."),
    # --- Alice ABOUT Carol (gold for (Alice, Carol); also a distractor for
    #     (Alice, Bob) since it is Alice-authored but does NOT mention Bob) -- #
    Turn("a_carol_1", "[2026-01-05 09:00] Alice: Carol leads the analytics team effectively."),
    # --- Alice self-report only (no subject mention; excluded everywhere) -- #
    Turn("a_self_1", "[2026-01-06 09:00] Alice: today was a quiet day for me."),
    # --- Bob self-reports (gold for (Bob, Bob)) ---------------------------- #
    Turn("b_self_1", "[2026-01-02 14:00] Bob: I prefer working on backend infrastructure."),
    Turn("b_self_2", "[2026-01-03 14:00] Bob: I shipped the caching layer this week."),
    # --- Bob ABOUT Carol (gold for (Bob, Carol)) --------------------------- #
    Turn("b_carol_1", "[2026-01-04 14:00] Bob: Carol gave thorough feedback on my design doc."),
]


# (observer, subject) -> set of gold corpus keys (the observer's OWN statements
# about the subject). For the self case the gold is the subject's self-reports.
GOLD: dict[tuple[str, str], set[str]] = {
    ("Alice", "Bob"): {"a_bob_1", "a_bob_2", "a_bob_3"},
    # Self case: the documented scope is the subject's role="self" members — i.e.
    # EVERY turn the subject authored, not only first-person statements. Bob
    # authored b_self_1/2 (first-person) AND b_carol_1 (a turn about Carol), so all
    # three are in the self-report scope. (b_carol_1 is also gold for (Bob, Carol).)
    ("Bob", "Bob"): {"b_self_1", "b_self_2", "b_carol_1"},
    ("Bob", "Carol"): {"b_carol_1"},
    ("Alice", "Carol"): {"a_carol_1"},
}

# For each subject, all keys that are SOMEONE's statement about that subject,
# tagged by the authoring observer. Used to compute leakage: a returned key whose
# author is NOT the requested observer (but who DID author a statement about the
# subject) is leakage. Derived from the corpus speakers + mentions explicitly so
# the gold is independent of the code under test.
SUBJECT_STATEMENTS: dict[str, dict[str, str]] = {
    # subject -> {key: authoring observer}
    "Bob": {
        "a_bob_1": "Alice",
        "a_bob_2": "Alice",
        "a_bob_3": "Alice",
        "c_bob_1": "Carol",
        "c_bob_2": "Carol",
    },
    "Carol": {
        "a_carol_1": "Alice",
        "b_carol_1": "Bob",
    },
}


@dataclass
class PairResult:
    observer: str
    subject: str
    mode_label: str  # "default(PERSON)" or "typed(CONCEPT)"
    is_self: bool
    returned_keys: list[str]
    gold: set[str]
    precision: float
    recall: float
    leakage: float
    attribution: float
    scoping_mode: str | None
    notes: list[str] = field(default_factory=list)


async def _ingest(v) -> dict[str, str]:
    """Ingest the corpus via the clean storage path + drive accretion inline.

    Mirrors ``eval.harness.ingest_corpus(clean=True, build_entity_registry=True)``
    but captures the storage ``memory_id -> corpus key`` mapping (the harness does
    not return ids), which the scorer needs to map returned observations back to
    gold keys. Same accretion call (``_accrete_entities``) the harness uses, so the
    real self/mention role assignment is exercised.
    """
    from memorylayer_server.services.embedding import EXT_EMBEDDING_PROVIDER
    from memorylayer_server.services.memory import EXT_MEMORY_SERVICE

    storage = get_extension(EXT_STORAGE_BACKEND, v)
    provider = get_extension(EXT_EMBEDDING_PROVIDER, v)
    memory_service = get_extension(EXT_MEMORY_SERVICE, v)

    contents = [t.content for t in CORPUS]
    embeddings = await provider.embed_batch(contents)

    id_to_key: dict[str, str] = {}
    for turn, embedding in zip(CORPUS, embeddings, strict=True):
        doc = _DocShim(turn.key, turn.content)
        memory = await storage.create_memory(
            _EVAL_WORKSPACE_ID,
            RememberInput(
                content=turn.content,
                type=MemoryType.SEMANTIC,
                importance=0.5,
                tags=[],
                metadata=_eval_metadata(doc),
                context_id="_default",
            ),
        )
        await storage.update_memory(_EVAL_WORKSPACE_ID, memory.id, embedding=embedding)
        # Drive registry accretion inline (clean path bypasses remember()'s
        # _inline_auto_enrich where it normally fires) — same call the harness uses.
        await memory_service._accrete_entities(_EVAL_WORKSPACE_ID, memory)
        id_to_key[memory.id] = turn.key

    return id_to_key


@dataclass
class _DocShim:
    """Minimal stand-in for ``eval.models.CorpusDoc`` so ``_eval_metadata`` can
    read ``.key`` + ``.content`` without importing the whole corpus loader."""

    key: str
    content: str


async def _confirm_registry_populated(v) -> dict:
    """Inventory each name's entities (PERSON + CONCEPT) and member roles.

    Probes BOTH types deliberately: name-first accretion
    (``resolve(promote=True)``) unifies a name seen as both a speaker (PERSON,
    role="self") and a third-party mention (CONCEPT, role="mention") onto a SINGLE
    PERSON node — so a person who both speaks AND is mentioned now shows
    ``person=(self=N,mention=M)`` and ``concept=<none>`` (no fork). Surfacing both
    types confirms the fix: the prior PERSON/CONCEPT split that broke the default
    ``get_representation`` path is gone.
    """
    registry = get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)
    summary: dict = {}
    for name in ("Alice", "Bob", "Carol"):
        per_type: dict = {}
        for etype in (EntityType.PERSON, EntityType.CONCEPT):
            try:
                res = await registry.resolve(
                    _EVAL_WORKSPACE_ID, name, etype, allow_create=False
                )
            except LookupError:
                per_type[etype.value] = None
                continue
            self_members = await registry.list_members(
                _EVAL_WORKSPACE_ID, res.entity.id, role="self", limit=100
            )
            mention_members = await registry.list_members(
                _EVAL_WORKSPACE_ID, res.entity.id, role="mention", limit=100
            )
            per_type[etype.value] = {
                "entity_id": res.entity.id,
                "self": len(self_members),
                "mention": len(mention_members),
            }
        summary[name] = per_type
    return summary


def _score_pair(
    observer: str,
    subject: str,
    mode_label: str,
    is_self: bool,
    returned_keys: list[str],
    scoping_mode: str | None,
) -> PairResult:
    gold = GOLD[(observer, subject)]
    returned_set = set(returned_keys)
    notes: list[str] = []

    # Attribution / leakage are evaluated per returned observation using the
    # independent SUBJECT_STATEMENTS authorship map.
    statements = SUBJECT_STATEMENTS.get(subject, {})

    if is_self:
        # Self case: "observer-authored-about-subject" == subject's self-report.
        # Gold IS the self-report set; a returned key is correctly attributed when
        # it is in gold (a self-report by the subject). Leakage in the self case is
        # any returned key that is another observer's statement about the subject.
        correct = returned_set & gold
        precision = len(correct) / len(returned_set) if returned_set else 1.0
        recall = len(correct) / len(gold) if gold else 1.0
        # The authoring observer of a self-report is the subject itself, so any
        # returned key that is another observer's statement about the subject is
        # leakage.
        leak_keys = {k for k in returned_keys if k in statements and statements[k] != subject}
        leakage = len(leak_keys) / len(returned_set) if returned_set else 0.0
        attributed = {k for k in returned_keys if k in gold}
        attribution = len(attributed) / len(returned_set) if returned_set else 1.0
    else:
        # observer-authored-about-subject: the returned key must be a statement
        # about the subject AND authored by the requested observer.
        correct = {k for k in returned_keys if statements.get(k) == observer}
        precision = len(correct) / len(returned_set) if returned_set else 1.0
        recall = len(returned_set & gold) / len(gold) if gold else 1.0
        # Leakage: returned keys that ARE a statement about the subject but were
        # authored by a DIFFERENT observer (the key metric — e.g. Carol's
        # Bob-statements appearing in get_representation(Alice, Bob)).
        leak_keys = {k for k in returned_keys if k in statements and statements[k] != observer}
        leakage = len(leak_keys) / len(returned_set) if returned_set else 0.0
        # Attribution: fraction of returned observations whose author IS the
        # requested observer. A returned key that is not in the statements map at
        # all (i.e. not about the subject) counts against attribution.
        attributed = {k for k in returned_keys if statements.get(k) == observer}
        attribution = len(attributed) / len(returned_set) if returned_set else 1.0

    if leak_keys:
        notes.append(f"LEAKED: {sorted(leak_keys)}")
    missing = gold - returned_set
    if missing:
        notes.append(f"MISSED gold: {sorted(missing)}")
    spurious = returned_set - gold - set(statements)
    if spurious:
        notes.append(f"SPURIOUS (not about subject): {sorted(spurious)}")

    return PairResult(
        observer=observer,
        subject=subject,
        mode_label=mode_label,
        is_self=is_self,
        returned_keys=returned_keys,
        gold=gold,
        precision=precision,
        recall=recall,
        leakage=leakage,
        attribution=attribution,
        scoping_mode=scoping_mode,
        notes=notes,
    )


async def _eval_pair(rep_service, id_to_key, observer, subject, mode_label, *, observer_type, subject_type) -> PairResult:
    rep = await rep_service.get_representation(
        _EVAL_WORKSPACE_ID,
        observer,
        subject,
        observer_type=observer_type,
        subject_type=subject_type,
        limit=20,
    )
    returned_keys = [
        id_to_key.get(obs.memory_id, f"<unknown:{obs.memory_id}>")
        for obs in rep.observations
    ]
    return _score_pair(
        observer,
        subject,
        mode_label,
        rep.is_self,
        returned_keys,
        rep.provenance.get("scoping_mode"),
    )


async def run() -> dict:
    """Bootstrap, ingest, score every gold pair in two resolution modes.

    Name-first accretion (``resolve(promote=True)``) unifies a person who both
    SPEAKS (PERSON/self) and is MENTIONED (CONCEPT/mention) onto a SINGLE PERSON
    node, so the two modes now report the SAME numbers — the type-split bug is
    gone. Both are kept to lock that in:

      * DEFAULT mode — ``get_representation(O, S)`` with no type args (the real
        consumer path; subject defaults to PERSON). This is what a caller gets and
        is now leakage-safe AND complete: the unified PERSON node carries the
        subject's mention members, so the cross-observer intersection is non-empty.
        This was the bug (recall 0.25); the GATE now asserts on it.
      * TYPED mode — resolve the subject explicitly as PERSON (where BOTH self and
        mention members now live after unification). Confirms the typed API still
        works and matches the default path post-fix.

    The GATE asserts on DEFAULT mode (the consumer path the bug broke). Importable
    so the integration test asserts the same gate. Tears down the service stack on
    the way out.
    """
    from memorylayer_server.dependencies import shutdown_services

    v = await bootstrap(embedding_provider="hash", storage_backend="sqlite")
    try:
        id_to_key = await _ingest(v)
        registry_summary = await _confirm_registry_populated(v)
        rep_service = get_representation_service(v)

        default_pairs: list[PairResult] = []
        typed_pairs: list[PairResult] = []
        for (observer, subject) in GOLD:
            # DEFAULT: no type hints (subject -> PERSON default) — the consumer path.
            default_pairs.append(
                await _eval_pair(
                    rep_service, id_to_key, observer, subject, "default(PERSON-subj)",
                    observer_type=None, subject_type=None,
                )
            )
            # TYPED: subject resolved explicitly as PERSON — where BOTH self and
            # mention members now live after name-first unification (no CONCEPT
            # fork). Matches the default path post-fix.
            typed_pairs.append(
                await _eval_pair(
                    rep_service, id_to_key, observer, subject, "typed(PERSON-subj)",
                    observer_type=EntityType.PERSON, subject_type=EntityType.PERSON,
                )
            )

        default_overall = _aggregate(default_pairs)
        typed_overall = _aggregate(typed_pairs)
        return {
            "registry_summary": registry_summary,
            "default_pairs": default_pairs,
            "default_overall": default_overall,
            "typed_pairs": typed_pairs,
            "typed_overall": typed_overall,
            # The gate is the headline claim: the leakage-safe intersection works
            # end-to-end through the real pipeline on the DEFAULT consumer path
            # (the path the type-split bug broke).
            "gate_pass": _gate_pass(default_overall),
        }
    finally:
        await shutdown_services(v)


def _aggregate(pairs: list[PairResult]) -> dict:
    n = len(pairs) or 1
    return {
        "mean_precision": sum(p.precision for p in pairs) / n,
        "mean_recall": sum(p.recall for p in pairs) / n,
        "mean_leakage": sum(p.leakage for p in pairs) / n,
        "max_leakage": max((p.leakage for p in pairs), default=0.0),
        "mean_attribution": sum(p.attribution for p in pairs) / n,
        "min_attribution": min((p.attribution for p in pairs), default=1.0),
    }


def _gate_pass(overall: dict) -> bool:
    return (
        overall["max_leakage"] <= _GATE_MAX_LEAKAGE
        and overall["mean_recall"] >= _GATE_MIN_RECALL
        and overall["min_attribution"] >= _GATE_MIN_ATTRIBUTION
    )


def _print_pairs(title: str, pairs: list[PairResult], overall: dict) -> None:
    print(f"\n{title}")
    header = f"  {'observer->subject':<22}{'scope':<13}{'prec':>6}{'rec':>6}{'leak':>7}{'attr':>6}  returned"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in pairs:
        label = f"{p.observer}->self={p.subject}" if p.is_self else f"{p.observer}->{p.subject}"
        print(
            f"  {label:<22}{str(p.scoping_mode):<13}"
            f"{p.precision:>6.2f}{p.recall:>6.2f}{p.leakage:>7.2f}{p.attribution:>6.2f}"
            f"  {sorted(p.returned_keys)}"
        )
        for note in p.notes:
            print(f"      ! {note}")
    print(
        f"  -> mean prec {overall['mean_precision']:.3f} | rec {overall['mean_recall']:.3f} | "
        f"leak {overall['mean_leakage']:.3f} (max {overall['max_leakage']:.3f}) | "
        f"attr {overall['mean_attribution']:.3f} (min {overall['min_attribution']:.3f})"
    )


def _print_report(result: dict) -> None:
    print("=" * 84)
    print("RepresentationService end-to-end eval (real ingest -> accretion -> assembly)")
    print("=" * 84)

    print("\nRegistry inventory per name (PERSON node | CONCEPT node), self/mention members:")
    for name, per_type in result["registry_summary"].items():
        parts = []
        for etype in ("person", "concept"):
            info = per_type.get(etype)
            if info is None:
                parts.append(f"{etype}=<none>")
            else:
                parts.append(f"{etype}=(self={info['self']},mention={info['mention']})")
        print(f"  {name:<6} " + "  ".join(parts))

    _print_pairs(
        "DEFAULT mode  get_representation(O, S)  [subject resolves as PERSON — the consumer path]",
        result["default_pairs"],
        result["default_overall"],
    )
    _print_pairs(
        "TYPED mode    subject_type=PERSON (where both self + mention members now live, post-unification)",
        result["typed_pairs"],
        result["typed_overall"],
    )

    verdict = "PASS" if result["gate_pass"] else "FAIL"
    print(
        f"\nGATE (evaluated on DEFAULT mode): {verdict}\n"
        f"  require max_leakage<={_GATE_MAX_LEAKAGE}, mean_recall>={_GATE_MIN_RECALL}, "
        f"min_attribution>={_GATE_MIN_ATTRIBUTION}"
    )
    print("=" * 84)


def main() -> int:
    result = asyncio.run(run())
    _print_report(result)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
