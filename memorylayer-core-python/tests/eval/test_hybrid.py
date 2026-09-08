"""Demonstrates the value of hybrid (keyword+vector) fusion in recall().

Strategy: run the eval harness with the *random* ``mock`` embedding provider so
the vector arm carries essentially no signal. With fusion off, recall is poor;
with fusion on, the BM25 keyword arm recovers the relevant documents — a large,
deterministic lift that isolates the keyword contribution of Phase 1 #1.

Run as subprocesses (fresh interpreter each) so they never contend with the
session-scoped framework other tests initialize.
"""

import json
import subprocess
import sys


def _run(*extra_args) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "memorylayer_server.eval", "run", "--json", *extra_args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"run failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr[-2000:]}"
    return json.loads(proc.stdout)


def test_hybrid_fusion_lifts_recall_when_vector_signal_is_weak():
    no_hybrid = _run("--embedding-provider", "mock", "--no-hybrid")
    hybrid = _run("--embedding-provider", "mock", "--hybrid")

    # Vector-only over random embeddings is near-useless; fusion should recover
    # most relevant documents via BM25.
    assert no_hybrid["mean_recall"] < 0.4
    assert hybrid["mean_recall"] >= 0.7
    assert hybrid["mean_recall"] >= no_hybrid["mean_recall"] + 0.3


def test_hybrid_on_does_not_regress_strong_vector_retrieval():
    # With the lexical hash embedding the vector arm is already strong; enabling
    # fusion must not drop recall quality.
    hybrid = _run("--embedding-provider", "hash", "--hybrid")
    assert hybrid["mean_recall"] >= 0.8
    assert hybrid["expected_top1_hit_rate"] >= 0.6
