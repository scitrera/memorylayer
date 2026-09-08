"""CLI entrypoint for the retrieval-eval harness.

    python -m memorylayer_server.eval gate [options]
    python -m memorylayer_server.eval run  [options]

``gate`` exits 0 on pass and 1 on fail (CI gate contract); ``run`` always exits 0
and just prints the report. ``--json`` emits machine-readable output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .gate import evaluate_gate
from .harness import DEFAULT_CORPUS, DEFAULT_QRELS, evaluate
from .models import Thresholds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memorylayer-eval", description="MemoryLayer retrieval evaluation harness")
    parser.add_argument("command", choices=("gate", "run"), help="gate = score + threshold check; run = score only")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Path to corpus JSONL (default: bundled fixture)")
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS), help="Path to qrels JSON (default: bundled fixture)")
    parser.add_argument("--k", type=int, default=5, help="Cutoff k for precision/recall (default: 5)")
    parser.add_argument("--data-dir", default=None, help="Storage dir (default: a fresh temp dir)")
    parser.add_argument("--embedding-provider", default="hash", help="Embedding provider (default: hash)")
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force hybrid (keyword+vector) fusion on/off (default: server default)",
    )
    parser.add_argument("--reranker", default="none", help="Reranker provider: none | rrf | llm | hyde (default: none)")
    parser.add_argument("--dimensions", type=int, default=None, help="Embedding dimensions (e.g. 1960 for embed_server Qwen3-VL)")
    parser.add_argument(
        "--embed-server-url", default=None, help="Embed server URL (default http://localhost:61051 when provider=embed_server)"
    )
    parser.add_argument("--storage-backend", default="sqlite", help="Storage backend: sqlite | postgresql | in_memory (default: sqlite)")
    parser.add_argument("--postgres-url", default=None, help="PostgreSQL URL when --storage-backend=postgresql")
    parser.add_argument("--clean-ingest", action="store_true", help="Batch-embed + insert directly, bypassing write-path enrichment")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    parser.add_argument("--min-recall", type=float, default=None, help="Override mean recall@k floor")
    parser.add_argument("--min-first-relevant", type=float, default=None, help="Override first-relevant-hit-rate floor")
    parser.add_argument("--min-expected-top1", type=float, default=None, help="Override expected-top1-hit-rate floor")
    return parser


def _resolve_thresholds(args: argparse.Namespace) -> Thresholds:
    t = Thresholds(k=args.k)
    if args.min_recall is not None:
        t.recall_at_k = args.min_recall
    if args.min_first_relevant is not None:
        t.first_relevant_hit = args.min_first_relevant
    if args.min_expected_top1 is not None:
        t.expected_top1 = args.min_expected_top1
    return t


def _print_human(report_dict: dict, gate_dict: dict | None) -> None:
    print(
        f"Retrieval eval (k={report_dict['k']}): "
        f"{report_dict['queries_run']}/{report_dict['queries_total']} queries scored, "
        f"{report_dict['queries_errored']} errored"
    )
    print(f"  mean recall@k        : {report_dict['mean_recall']:.3f}")
    print(f"  mean precision@k     : {report_dict['mean_precision']:.3f}")
    print(f"  mean MRR             : {report_dict['mean_mrr']:.3f}")
    print(f"  mean nDCG@k          : {report_dict['mean_ndcg']:.3f}")
    print(f"  first-relevant hit   : {report_dict['first_relevant_hit_rate']:.3f}")
    print(f"  expected-top1 hit    : {report_dict['expected_top1_hit_rate']:.3f} (n={report_dict['expected_top1_denominator']})")
    print(f"  mean latency (ms)    : {report_dict['mean_latency_ms']:.2f}")
    if gate_dict is not None:
        verdict = gate_dict["verdict"].upper()
        print(f"\nGATE: {verdict}")
        for b in gate_dict["breaches"]:
            print(f"  ✗ {b['metric']}: observed {b['observed']} < threshold {b['threshold']}")


async def _amain(args: argparse.Namespace) -> int:
    report = await evaluate(
        corpus_path=args.corpus,
        qrels_path=args.qrels,
        k=args.k,
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        hybrid=args.hybrid,
        storage_backend=args.storage_backend,
        reranker=args.reranker,
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
        postgres_url=args.postgres_url,
        clean_ingest=args.clean_ingest,
    )

    if args.command == "run":
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _print_human(report.to_dict(), None)
        return 0

    gate = evaluate_gate(report, _resolve_thresholds(args))
    if args.json:
        print(json.dumps(gate.to_dict(), indent=2))
    else:
        _print_human(report.to_dict(), gate.to_dict())
    return 0 if gate.verdict == "pass" else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
