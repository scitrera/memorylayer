#!/usr/bin/env python3
"""Multivector (ColPali / MaxSim late-interaction) retrieval baseline — Postgres only.

Ingests each corpus doc as a ColPali-style multi-vector (token vectors, dim 128)
via the embed server's /v1/embeddings/multi (documents encoded with
input_type="document"), stored in the enterprise ``multivector`` column. Queries
are encoded with input_type="query" and retrieved via
``storage.search_memories_multivector`` (native pgvector ``max_sim``). This is a
distinct retrieval mode from the single-vector recall() pipeline, run as a base
to compare against single-vector vector-only / hybrid.

Requires the enterprise venv + a Postgres backend + the embed server. Example:
    proprietary/memorylayer-enterprise/.venv/bin/python benchmarks/run_benchmark_multivector.py \
        --corpus benchmarks/.cache/longmemeval/corpus.jsonl \
        --qrels  benchmarks/.cache/longmemeval/qrels.json \
        --postgres-url postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer
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
from memorylayer_server.eval.harness import _EVAL_KEY, _EVAL_WORKSPACE_ID, bootstrap
from memorylayer_server.eval.models import load_corpus
from memorylayer_server.models.memory import MemoryType, RememberInput
from memorylayer_server.services.embedding import EXT_EMBEDDING_PROVIDER
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND


def _load_qrels_raw(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("queries", data) if isinstance(data, dict) else data


# The ColPali/ModernVBERT multi-vector encoder caps around ~1600 tokens; very
# long inputs 500/timeout. The single-vector endpoint silently truncates to its
# model max, so for a fair comparison we truncate multi-vector inputs to a
# matching char budget (≈ the tested-working ceiling) rather than failing.
_MV_MAX_CHARS = 20000


async def _ingest_multivector(v, docs, batch_size: int = 8) -> None:
    storage = get_extension(EXT_STORAGE_BACKEND, v)
    provider = get_extension(EXT_EMBEDDING_PROVIDER, v)
    total = len(docs)
    truncated = 0
    skipped = 0

    async def _store(doc, vectors) -> None:
        memory = await storage.create_memory(
            _EVAL_WORKSPACE_ID,
            RememberInput(
                content=doc.content,
                type=MemoryType(doc.type),
                importance=doc.importance,
                tags=doc.tags,
                metadata={_EVAL_KEY: doc.key},
                context_id="_default",
            ),
        )
        # Store the full multivector (for exact MaxSim rerank) AND flatten its
        # token-vectors into the HNSW index (for stage-1 candidate pruning).
        await storage.update_memory(_EVAL_WORKSPACE_ID, memory.id, multivector=vectors)
        await storage.index_memory_multivector(_EVAL_WORKSPACE_ID, memory.id, vectors)

    for i in range(0, total, batch_size):
        chunk = docs[i : i + batch_size]
        texts = []
        for d in chunk:
            if len(d.content) > _MV_MAX_CHARS:
                truncated += 1
            texts.append(d.content[:_MV_MAX_CHARS])
        try:
            mvs = await provider.embed_batch_multivector(texts)
            for doc, mv in zip(chunk, mvs, strict=True):
                await _store(doc, mv.vectors)
        except Exception as exc:  # noqa: BLE001 - batch failed; fall back per-doc
            print(f"  [ingest-mv] batch @{i} failed ({type(exc).__name__}); retrying per-doc", flush=True)
            for doc, text_ in zip(chunk, texts, strict=True):
                try:
                    mv = await provider.embed_text_multivector(text_)
                    await _store(doc, mv.vectors)
                except Exception as exc2:  # noqa: BLE001 - drop the offending doc
                    skipped += 1
                    print(f"    [ingest-mv] skip {doc.key} (len={len(doc.content)}): {type(exc2).__name__}", flush=True)
        print(f"  [ingest-mv] {min(i + batch_size, total)}/{total}", flush=True)

    print(f"  [ingest-mv] done: truncated={truncated} skipped={skipped}", flush=True)


async def _retrieve_mv(v, qmv, k: int, mode: str, tokens_per_query: int) -> list[str]:
    """Retrieve via brute-force MaxSim or the indexed (ColBERT) two-stage path.

    Modes: 'brute' (exact halfvec MaxSim seq-scan); 'indexed' (halfvec HNSW
    stage-1 ANN + exact halfvec rerank); 'indexed-bit' (binary-quantized Hamming
    stage-1 ANN + exact halfvec rerank). Token + multivector storage is halfvec.
    """
    storage = get_extension(EXT_STORAGE_BACKEND, v)
    _precision = {
        "indexed": "half",
        "indexed-bit": "bit",
    }
    if mode in _precision:
        results = await storage.search_memories_multivector_indexed(
            _EVAL_WORKSPACE_ID, qmv.vectors, limit=max(k, 20), min_relevance=0.0,
            tokens_per_query=tokens_per_query,
            ann_precision=_precision[mode],
        )
    else:
        results = await storage.search_memories_multivector(
            _EVAL_WORKSPACE_ID, qmv.vectors, limit=max(k, 20), min_relevance=0.0,
        )
    return [(m.metadata or {}).get(_EVAL_KEY) for m, _score in results if (m.metadata or {}).get(_EVAL_KEY)]


async def _sweep(v, queries, k, mode, tokens_per_query) -> dict:
    provider = get_extension(EXT_EMBEDDING_PROVIDER, v)
    overall = defaultdict(list)
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    label = f"multivector-{mode}" + (f"(pool={tokens_per_query})" if mode == "indexed" else "")
    start = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        relevant = set(q["relevant"])
        grades = q.get("grades") or {kk: 1.0 for kk in q["relevant"]}
        qmv = await provider.embed_text_multivector(q["query"])
        retrieved = await _retrieve_mv(v, qmv, k, mode, tokens_per_query)
        vals = {
            "recall": metrics.recall_at_k(retrieved, relevant, k),
            "mrr": metrics.mrr(retrieved, relevant),
            "ndcg": metrics.ndcg_at_k(retrieved, grades, k),
            "fr": float(metrics.first_relevant_hit(retrieved, relevant)),
        }
        qtype = q.get("_question_type", "all")
        for m, val in vals.items():
            overall[m].append(val)
            by_type[qtype][m].append(val)
        if i % 25 == 0:
            print(f"  [{label}] {i}/{len(queries)} queries", flush=True)
    elapsed = time.perf_counter() - start
    o = {m: round(metrics.mean(vv), 4) for m, vv in overall.items()}
    return {
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "overall": o,
        "by_type": {t: {m: round(metrics.mean(vv), 4) for m, vv in d.items()} for t, d in sorted(by_type.items())},
    }


async def _amain(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = _load_qrels_raw(args.qrels)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"Corpus: {len(docs)} docs | Queries: {len(queries)} | k={args.k} | modes={modes} | multivector(ColPali) on Postgres", flush=True)

    v = await bootstrap(
        embedding_provider="embed_server",
        storage_backend="postgresql",
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
        postgres_url=args.postgres_url,
        memory_service="default",
        data_dir=args.data_dir,
    )
    try:
        if args.skip_ingest:
            print("Skipping ingest (--skip-ingest): reusing existing multivector store", flush=True)
        else:
            t0 = time.perf_counter()
            await _ingest_multivector(v, docs)
            print(f"Multivector ingest complete in {time.perf_counter() - t0:.1f}s", flush=True)

        results = []
        for mode in modes:
            print(f"\n>> retrieval mode: {mode}", flush=True)
            results.append(await _sweep(v, queries, args.k, mode, args.tokens_per_query))

        print(f"\n=== Multivector results (k={args.k}) ===")
        header = f"{'mode':<28}{'recall':>9}{'mrr':>9}{'ndcg':>9}{'first-rel':>11}{'sec':>8}"
        print(header)
        print("-" * len(header))
        for r in results:
            o = r["overall"]
            print(f"{r['label']:<28}{o['recall']:>9.3f}{o['mrr']:>9.3f}{o['ndcg']:>9.3f}{o['fr']:>11.3f}{r['elapsed_s']:>8.1f}")
        if results:
            last = results[-1]
            print(f"\nPer-question-type recall@{args.k} ({last['label']}):")
            for t, d in last["by_type"].items():
                print(f"  {t:<28} recall={d['recall']:.3f}  mrr={d['mrr']:.3f}  first-rel={d['fr']:.3f}")

        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nWrote JSON results to {args.json_out}")
    finally:
        await shutdown_services(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multivector ColPali/MaxSim retrieval baseline (Postgres)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dimensions", type=int, default=1960)
    ap.add_argument("--embed-server-url", default=None)
    ap.add_argument("--postgres-url", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--modes", default="brute,indexed", help="Comma list of retrieval modes: brute,indexed")
    ap.add_argument("--tokens-per-query", type=int, default=20, help="Indexed stage-1 ANN candidate pool per query token")
    ap.add_argument("--skip-ingest", action="store_true", help="Reuse an already-ingested multivector store")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    # Resolve --json-out against the launch cwd before bootstrap() may chdir
    # (otherwise the "successful" write lands in an ephemeral dir and is lost).
    if args.json_out:
        args.json_out = str(Path(args.json_out).resolve())
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
