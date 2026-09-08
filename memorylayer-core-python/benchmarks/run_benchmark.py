#!/usr/bin/env python3
"""Observable retrieval benchmark runner.

Ingests a corpus ONCE (with live progress), then sweeps retrieval configs
(vector-only / hybrid / hybrid+rrf) against the same store by toggling the
MemoryService in place — so the expensive embedding happens once, not per
config. Reports overall and per-question-type metrics.

Drives the same MemoryLayer recall pipeline used in production. Works against
sqlite-vec (default) or the enterprise Postgres backend (--storage-backend
postgresql --postgres-url ...). Network/embed-server access required; run
outside the sandbox.

Example:
    python benchmarks/run_benchmark.py \
        --corpus benchmarks/.cache/longmemeval/corpus.jsonl \
        --qrels  benchmarks/.cache/longmemeval/qrels.json \
        --dimensions 1960
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from scitrera_app_framework import get_extension

from memorylayer_server.dependencies import shutdown_services
from memorylayer_server.eval import metrics
from memorylayer_server.eval.harness import _EVAL_WORKSPACE_ID, bootstrap, ingest_corpus
from memorylayer_server.eval.models import load_corpus
from memorylayer_server.models.memory import RecallInput, RecallMode
from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
from memorylayer_server.services.reranker import EXT_RERANKER_SERVICE


def _load_qrels_raw(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("queries", data) if isinstance(data, dict) else data


async def _retrieve(ms, query: str, k: int, associations: bool = False) -> list[str]:
    result = await ms.recall(
        _EVAL_WORKSPACE_ID,
        RecallInput(
            query=query,
            mode=RecallMode.RAG,
            limit=max(k, 20),
            min_relevance=0.0,
            include_associations=associations,
            include_global=False,
            include_global_user=False,
        ),
    )
    return [(m.metadata or {}).get("eval_key") for m in result.memories if (m.metadata or {}).get("eval_key")]


async def _sweep(ms, queries: list[dict], k: int, label: str, associations: bool = False) -> dict:
    overall = defaultdict(list)
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    start = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        relevant = set(q["relevant"])
        grades = q.get("grades") or {k_: 1.0 for k_ in q["relevant"]}
        retrieved = await _retrieve(ms, q["query"], k, associations)
        rec = metrics.recall_at_k(retrieved, relevant, k)
        mrr = metrics.mrr(retrieved, relevant)
        ndcg = metrics.ndcg_at_k(retrieved, grades, k)
        fr = metrics.first_relevant_hit(retrieved, relevant)
        qtype = q.get("_question_type", "all")
        for metric, val in (("recall", rec), ("mrr", mrr), ("ndcg", ndcg), ("fr", fr)):
            overall[metric].append(val)
            by_type[qtype][metric].append(val)
        if i % 25 == 0:
            print(f"  [{label}] {i}/{len(queries)} queries", flush=True)
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "n": len(queries),
        "elapsed_s": round(elapsed, 1),
        "overall": {m: round(metrics.mean(v), 4) for m, v in overall.items()},
        "by_type": {t: {m: round(metrics.mean(v), 4) for m, v in d.items()} for t, d in sorted(by_type.items())},
    }


def _print_report(results: list[dict], k: int) -> None:
    print(f"\n=== Benchmark results (k={k}) ===")
    header = f"{'config':<16}{'recall':>9}{'mrr':>9}{'ndcg':>9}{'first-rel':>11}{'sec':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        o = r["overall"]
        print(f"{r['label']:<16}{o['recall']:>9.3f}{o['mrr']:>9.3f}{o['ndcg']:>9.3f}{o['fr']:>11.3f}{r['elapsed_s']:>8.1f}")
    # Per-type recall for the richest config (last).
    if results:
        last = results[-1]
        print(f"\nPer-question-type recall@{k} ({last['label']}):")
        for t, d in last["by_type"].items():
            print(f"  {t:<28} recall={d['recall']:.3f}  mrr={d['mrr']:.3f}  first-rel={d['fr']:.3f}")


async def _amain(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = _load_qrels_raw(args.qrels)
    print(f"Corpus: {len(docs)} docs | Queries: {len(queries)} | k={args.k} | backend={args.storage_backend}", flush=True)

    # Bootstrap once with rrf available; sweeps toggle it on/off in place.
    print("Bootstrapping service stack + embedding corpus (this is the slow part)...", flush=True)
    v = await bootstrap(
        embedding_provider=args.embedding_provider,
        storage_backend=args.storage_backend,
        reranker="rrf",
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
        postgres_url=args.postgres_url,
        memory_service=args.memory_service,
        data_dir=args.data_dir,
    )
    try:
        ms = get_extension(EXT_MEMORY_SERVICE, v)
        if args.entity_anchor:
            # Entity-anchored RRF fusion channel (default-off feature flag).
            ms.entity_anchor_enabled = True
            print("Entity-anchored channel ON: RRF-fusing speaker/entity-restricted candidates", flush=True)
        if args.entity_registry:
            # Registry-backed entity-expansion channel (default-off feature flag).
            # (a) Enable ACCRETION at ingest so the registry is populated (the
            #     OSS default registry service is auto-wired by bootstrap); set
            #     BEFORE ingest below so members accrete as the corpus is stored.
            # (b) Enable the registry RECALL channel so query-entity -> canonical
            #     entity -> member-memories expansion is RRF-fused at recall.
            ms.entity_registry_enabled = True
            ms.entity_registry_recall_enabled = True
            reg_svc = ms.entity_registry_service
            reg_provider = type(reg_svc).__name__ if reg_svc is not None else "none"
            print(
                "Entity-registry channel ON: accretion at ingest + RRF-fusing "
                f"canonical-entity member memories at recall "
                f"(provider={reg_provider})",
                flush=True,
            )
        if args.associations:
            # Lower the auto-association threshold so the clean corpus actually
            # forms similarity edges for recall's graph expansion to traverse.
            ms.auto_association_threshold = args.assoc_threshold
            # A1: query-aware expansion — re-score graph-discovered neighbors by
            # query similarity instead of graph topology (default off in prod).
            ms.assoc_query_aware = args.assoc_query_aware
            ms.assoc_query_floor = args.assoc_query_floor
            # A2: consensus boost — graph as ranking-only signal, no injection.
            ms.assoc_consensus_boost_weight = args.assoc_consensus_weight if args.assoc_consensus else 0.0
            if args.assoc_consensus:
                mode = f"consensus-boost (weight={args.assoc_consensus_weight}, no injection)"
            elif args.assoc_query_aware:
                mode = f"query-aware (floor={args.assoc_query_floor})"
            else:
                mode = "legacy topological"
            print(
                f"Associations ON: building graph at threshold={args.assoc_threshold}, "
                f"recall include_associations=True, scoring={mode}",
                flush=True,
            )

        if args.skip_ingest:
            print("Skipping ingest (--skip-ingest): reusing existing store", flush=True)
        else:
            t0 = time.perf_counter()
            await ingest_corpus(v, docs, clean=True, progress=True, build_associations=args.associations, build_entity_registry=args.entity_registry)
            print(f"Ingest complete in {time.perf_counter() - t0:.1f}s", flush=True)

        rrf = get_extension(EXT_RERANKER_SERVICE, v)

        # Disable the recall result cache: it keys on (workspace, query,
        # RecallInput) but NOT on the service-level hybrid/reranker settings we
        # toggle between sweeps, so leaving it on would serve config 1's cached
        # results to configs 2/3 (identical metrics, 0s runtime).
        ms.cache = None

        all_configs = {
            "vector": ("vector-only", False, None),
            "hybrid": ("hybrid", True, None),
            "rrf": ("hybrid+rrf", True, rrf),
        }
        selected = args.configs.split(",") if args.configs else list(all_configs)
        configs = [all_configs[c.strip()] for c in selected]
        results = []
        for label, hybrid_on, reranker in configs:
            ms.hybrid_search_enabled = hybrid_on
            ms.reranker_service = reranker
            print(f"\n>> sweeping config: {label} (hybrid={hybrid_on}, rerank={'rrf' if reranker else 'none'})", flush=True)
            results.append(await _sweep(ms, queries, args.k, label, associations=args.associations))

        _print_report(results, args.k)
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nWrote JSON results to {args.json_out}")
    finally:
        await shutdown_services(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Observable single-ingest retrieval benchmark sweep")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dimensions", type=int, default=1960)
    ap.add_argument(
        "--embedding-provider",
        default="embed_server",
        help="Embedding provider: 'embed_server' (default, needs a live server) or 'hash' "
        "(deterministic lexical, zero-infra; use for offline sqlite smoke/direction runs — "
        "NOT a substitute for the real embed-server number).",
    )
    ap.add_argument("--embed-server-url", default=None)
    ap.add_argument("--storage-backend", default="sqlite")
    ap.add_argument("--postgres-url", default=None)
    ap.add_argument("--memory-service", default=None, help="Force memory service impl (e.g. 'default' for parity vs enterprise 'saas')")
    ap.add_argument("--data-dir", default=None, help="Persistent storage dir (reuse with --skip-ingest)")
    ap.add_argument("--skip-ingest", action="store_true", help="Reuse an already-ingested --data-dir store (skip embedding)")
    ap.add_argument("--configs", default=None, help="Comma list of configs to sweep: vector,hybrid,rrf (default: all)")
    ap.add_argument("--associations", action="store_true", help="Build similarity-edge graph at ingest + recall with include_associations=True")
    ap.add_argument("--assoc-threshold", type=float, default=0.8, help="Auto-association similarity threshold when --associations (default 0.8)")
    ap.add_argument("--assoc-query-aware", action="store_true", help="A1: re-score graph-discovered neighbors by query similarity (drops off-query noise)")
    ap.add_argument("--assoc-query-floor", type=float, default=0.0, help="Min query-similarity for a query-aware expanded neighbor to be kept (default 0.0)")
    ap.add_argument("--assoc-consensus", action="store_true", help="A2: use graph as ranking-only consensus boost (no candidate injection)")
    ap.add_argument("--assoc-consensus-weight", type=float, default=0.3, help="A2 consensus boost magnitude (default 0.3)")
    ap.add_argument("--entity-anchor", action="store_true", help="Enable the entity-anchored RRF fusion channel (requires entity metadata at ingest)")
    ap.add_argument("--entity-registry", action="store_true", help="Enable the registry-backed entity-expansion channel: accretion at ingest + RRF-fusing canonical-entity member memories at recall. NOTE: OSS uses exact+alias resolution (provider=DefaultEntityRegistryService); a meaningful evaluation requires MEMORYLAYER_ENTITY_REGISTRY_PROVIDER=postgresql (enterprise fuzzy/LLM-merged canonical entities) — an OSS-only run is not indicative of the full lift.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    # Resolve --json-out against the launch cwd NOW: bootstrap() may change the
    # process working directory, after which a relative path would be written to
    # (and silently lost in) an ephemeral dir even though the write "succeeds".
    if args.json_out:
        args.json_out = str(Path(args.json_out).resolve())
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
