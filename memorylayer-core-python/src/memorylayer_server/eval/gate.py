"""Correctness gate: compare an EvalReport against thresholds and produce a verdict.

Pure comparison logic (no I/O, no framework) so it is trivially unit-testable.
The gate passes only when every checked metric meets its floor; otherwise it
returns the list of breaches. ``expected_top1`` is only checked when at least one
qrel declared an ``expected_top1``.
"""

from __future__ import annotations

from .models import Breach, EvalReport, GateResult, Thresholds


def evaluate_gate(report: EvalReport, thresholds: Thresholds) -> GateResult:
    breaches: list[Breach] = []

    if report.mean_recall < thresholds.recall_at_k:
        breaches.append(Breach("mean_recall", report.mean_recall, thresholds.recall_at_k))

    if report.first_relevant_hit_rate < thresholds.first_relevant_hit:
        breaches.append(Breach("first_relevant_hit_rate", report.first_relevant_hit_rate, thresholds.first_relevant_hit))

    if report.expected_top1_denominator > 0 and report.expected_top1_hit_rate < thresholds.expected_top1:
        breaches.append(Breach("expected_top1_hit_rate", report.expected_top1_hit_rate, thresholds.expected_top1))

    # Any query that errored is an automatic failure (fail-closed).
    if report.queries_errored > 0:
        breaches.append(Breach("queries_errored", float(report.queries_errored), 0.0))

    verdict = "pass" if not breaches else "fail"
    return GateResult(verdict=verdict, breaches=breaches, report=report, thresholds=thresholds)
