"""Tests for the correctness gate.

Two layers:
  * pure ``evaluate_gate`` logic (fast, no framework), and
  * an end-to-end subprocess run of ``python -m memorylayer_server.eval gate``
    that boots the real recall pipeline over the bundled fixture corpus. The
    subprocess runs in a fresh interpreter so it never contends with the
    session-scoped framework other tests initialize.

The end-to-end test IS the CI retrieval-regression gate: if a ranking change
drops fixture retrieval quality below the thresholds, this test fails.
"""

import json
import subprocess
import sys

from memorylayer_server.eval.gate import evaluate_gate
from memorylayer_server.eval.models import EvalReport, Thresholds


def _report(recall, first_rel, top1, top1_n=10, errored=0):
    return EvalReport(
        k=5,
        queries=[],
        queries_total=10,
        queries_run=10 - errored,
        queries_errored=errored,
        mean_recall=recall,
        first_relevant_hit_rate=first_rel,
        expected_top1_hit_rate=top1,
        expected_top1_denominator=top1_n,
    )


def test_gate_passes_when_all_floors_met():
    result = evaluate_gate(_report(1.0, 1.0, 1.0), Thresholds())
    assert result.verdict == "pass"
    assert result.breaches == []


def test_gate_fails_on_low_recall():
    result = evaluate_gate(_report(0.5, 1.0, 1.0), Thresholds())
    assert result.verdict == "fail"
    assert any(b.metric == "mean_recall" for b in result.breaches)


def test_gate_fails_closed_on_errored_queries():
    result = evaluate_gate(_report(1.0, 1.0, 1.0, errored=1), Thresholds())
    assert result.verdict == "fail"
    assert any(b.metric == "queries_errored" for b in result.breaches)


def test_gate_skips_expected_top1_when_no_denominator():
    # No qrel declared expected_top1 -> that check must not fire.
    result = evaluate_gate(_report(1.0, 1.0, 0.0, top1_n=0), Thresholds())
    assert result.verdict == "pass"


def _run_gate(*extra_args):
    return subprocess.run(
        [sys.executable, "-m", "memorylayer_server.eval", "gate", "--json", *extra_args],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_gate_passes_on_fixture_corpus():
    """End-to-end CI gate: bundled corpus must retrieve cleanly through recall()."""
    proc = _run_gate()
    assert proc.returncode == 0, f"gate failed unexpectedly:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr[-2000:]}"
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "pass"
    report = payload["report"]
    assert report["queries_errored"] == 0
    assert report["mean_recall"] >= 0.80
    assert report["first_relevant_hit_rate"] >= 0.80
    assert report["expected_top1_hit_rate"] >= 0.60


def test_cli_gate_bites_on_wrong_ground_truth(tmp_path):
    """A qrels file with shuffled-wrong answers must fail the gate (proves it bites)."""
    bad_qrels = {
        "schema_version": 1,
        "queries": [
            {"query_id": "b1", "query": "python web framework for building APIs", "relevant": ["git-vcs"], "expected_top1": "git-vcs"},
            {
                "query_id": "b2",
                "query": "relational database SQL queries",
                "relevant": ["transformer-arch"],
                "expected_top1": "transformer-arch",
            },
            {"query_id": "b3", "query": "in memory caching key value store", "relevant": ["rust-systems"], "expected_top1": "rust-systems"},
        ],
    }
    path = tmp_path / "bad_qrels.json"
    path.write_text(json.dumps(bad_qrels), encoding="utf-8")

    proc = _run_gate("--qrels", str(path))
    assert proc.returncode == 1, f"gate should have failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr[-2000:]}"
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "fail"
