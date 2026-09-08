"""Integration regression gate for ``get_representation`` over the REAL pipeline.

Thin wrapper over ``benchmarks/run_representation_eval.py``: it runs the same
end-to-end eval (real ingest -> entity-registry accretion -> resolution ->
representation assembly on a scripted multi-observer corpus) and asserts the gate
the script prints — leakage == 0, recall near-perfect, attribution total — in the
TYPED resolution mode (where the subject is resolved to the entity type its
mention members actually live under).

This complements ``tests/unit/test_representation_service.py`` (which proves the
scoped-intersection contract against fakes): here the same leakage-safe property
is proven to survive the real accretion + resolution path, not just the assembly
in isolation.

Marked ``integration`` automatically by ``tests/integration/conftest.py``; the
default CI selector (``-m "not integration"``) deselects it, so it does not slow
the unit lane while still being a runnable regression guard.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the benchmark script by path (benchmarks/ is not an importable package).
# Register in sys.modules BEFORE exec so @dataclass (which resolves the defining
# module via cls.__module__ in sys.modules) can process its field() defaults.
_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "run_representation_eval.py"
_spec = importlib.util.spec_from_file_location("run_representation_eval", _BENCH)
assert _spec and _spec.loader
rep_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_representation_eval"] = rep_eval
_spec.loader.exec_module(rep_eval)


@pytest.fixture(scope="module")
async def eval_result() -> dict:
    """Run the end-to-end eval once and share the result across assertions."""
    return await rep_eval.run()


async def test_registry_accretion_unifies_person_and_concept(eval_result: dict):
    """Name-first accretion unifies a person who both speaks AND is mentioned onto
    a SINGLE PERSON node. Bob's self-authored turns (role="self") and the turns
    that mention him (role="mention") co-locate on the PERSON node; there is NO
    CONCEPT fork. This is the fix for the type-split bug that broke the default
    ``get_representation`` path (mentions used to land on a separate CONCEPT node).
    """
    summary = eval_result["registry_summary"]
    # Bob speaks (3 self) and is mentioned by Alice + Carol (5 mention) — ALL on
    # the ONE PERSON node now. No CONCEPT node exists for Bob.
    assert summary["Bob"]["person"]["self"] == 3
    assert summary["Bob"]["person"]["mention"] == 5
    assert summary["Bob"]["concept"] is None, "Bob must not be forked into a CONCEPT node"


async def test_default_mode_is_leakage_safe_and_complete_end_to_end(eval_result: dict):
    """THE headline claim (post-fix): the DEFAULT consumer path —
    ``get_representation(O, S)`` with no type args (subject_type=PERSON) — returns
    the OBSERVER's perspective on the SUBJECT and EXCLUDES other observers'
    statements (leakage == 0), with perfect recall + attribution. This is the path
    the entity-type-split bug broke (recall used to collapse to 0.25 because the
    subject's mention members were filed under a separate CONCEPT node); name-first
    accretion unifies them onto the PERSON node so the default path now works.
    """
    overall = eval_result["default_overall"]
    assert overall["max_leakage"] == 0.0, "leakage guard failed through the real pipeline"
    assert overall["mean_recall"] >= rep_eval._GATE_MIN_RECALL
    assert overall["min_attribution"] >= rep_eval._GATE_MIN_ATTRIBUTION
    assert eval_result["gate_pass"] is True

    # Spell out the marquee pair: Alice's view of Bob (DEFAULT mode, PERSON subject)
    # includes exactly Alice's three statements about Bob and NONE of Carol's two.
    alice_bob = next(
        p for p in eval_result["default_pairs"] if p.observer == "Alice" and p.subject == "Bob"
    )
    assert set(alice_bob.returned_keys) == {"a_bob_1", "a_bob_2", "a_bob_3"}
    assert "c_bob_1" not in alice_bob.returned_keys
    assert "c_bob_2" not in alice_bob.returned_keys


async def test_typed_mode_matches_default_after_unification(eval_result: dict):
    """Post-fix, resolving the subject explicitly as PERSON (where BOTH self and
    mention members now live) yields the SAME leakage-safe, complete result as the
    default path — the typed API still works and is consistent with the default.
    """
    typed_overall = eval_result["typed_overall"]
    assert typed_overall["max_leakage"] == 0.0
    assert typed_overall["mean_recall"] >= rep_eval._GATE_MIN_RECALL
    assert typed_overall["min_attribution"] >= rep_eval._GATE_MIN_ATTRIBUTION
