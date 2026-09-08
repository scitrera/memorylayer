#!/usr/bin/env python3
"""Hands-on LanceDB probe — multivector (ColPali/MaxSim) retrieval as a 2nd index.

Seeds a LanceDB table from the SAME multivector data already stored in our
Postgres eval DB (so the doc vectors are identical to the pgvector halfvec run),
encodes the same 90 LongMemEval queries via the embed server, and runs a few
LanceDB scoring permutations — scored with the SAME metrics as our PG benchmark
so the quality/perf gap vs pgvector+halfvec (indexed 0.830) is apples-to-apples.

Throwaway experiment (own venv): does NOT touch the product. Run with the
.slop/lancedb-venv interpreter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path

import lancedb
import psycopg
import pyarrow as pa

PG_URL = "postgresql://memorylayer:memorylayer_dev@localhost:55432/memorylayer"
EMBED_URL = "http://localhost:61051/v1/embeddings/multi"
DIM = 128

# Reuse the exact eval metrics for parity with the PG benchmark.
_spec = importlib.util.spec_from_file_location(
    "eval_metrics", str(Path(__file__).resolve().parents[1] / "src/memorylayer_server/eval/metrics.py")
)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

_VEC_RE = re.compile(r"\[([^\]]*)\]")


def _parse_mv(text: str) -> list[list[float]]:
    return [[float(x) for x in g.split(",")] for g in _VEC_RE.findall(text)]


def _load_docs() -> list[dict]:
    docs = []
    with psycopg.connect(PG_URL) as c:
        c.execute("SET statement_timeout = 0")
        for key, mvtext in c.execute(
            "SELECT metadata->>'eval_key', multivector::text FROM memories WHERE multivector IS NOT NULL"
        ):
            if key:
                docs.append({"key": key, "mv": _parse_mv(mvtext)})
    return docs


def _embed_query(text: str) -> list[list[float]]:
    body = json.dumps({"input": [text], "input_type": "query"}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return d["data"][0]["vectors"]


def _score(queries, retrieve, k):
    from collections import defaultdict
    overall = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    start = time.perf_counter()
    for q in queries:
        relevant = set(q["relevant"])
        grades = q.get("grades") or {kk: 1.0 for kk in q["relevant"]}
        retrieved = retrieve(q["query"], k)
        vals = {
            "recall": metrics.recall_at_k(retrieved, relevant, k),
            "mrr": metrics.mrr(retrieved, relevant),
            "ndcg": metrics.ndcg_at_k(retrieved, grades, k),
            "fr": float(metrics.first_relevant_hit(retrieved, relevant)),
        }
        qt = q.get("_question_type", "all")
        for m, v in vals.items():
            overall[m].append(v)
            by_type[qt][m].append(v)
    elapsed = time.perf_counter() - start
    return {m: round(metrics.mean(v), 4) for m, v in overall.items()}, elapsed, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--skip-flat", action="store_true", help="Skip the slow exhaustive flat baseline")
    args = ap.parse_args()

    qr = json.loads(Path(args.qrels).read_text())
    queries = qr.get("queries", qr) if isinstance(qr, dict) else qr

    print("Loading multivector docs from Postgres...", flush=True)
    t0 = time.perf_counter()
    docs = _load_docs()
    print(f"  {len(docs)} docs loaded in {time.perf_counter()-t0:.1f}s "
          f"(avg {sum(len(d['mv']) for d in docs)//max(len(docs),1)} tokens/doc)", flush=True)

    print("Pre-encoding queries via embed server...", flush=True)
    t0 = time.perf_counter()
    qcache = {q["query"]: _embed_query(q["query"]) for q in queries}
    print(f"  {len(qcache)} queries encoded in {time.perf_counter()-t0:.1f}s", flush=True)

    import tempfile
    db = lancedb.connect(tempfile.mkdtemp())
    schema = pa.schema([("key", pa.string()), ("mv", pa.list_(pa.list_(pa.float32(), DIM)))])
    t0 = time.perf_counter()
    table = db.create_table("docs", data=docs, schema=schema)
    print(f"LanceDB table built ({table.count_rows()} rows) in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []

    def make_retrieve(nprobes=None, refine=None):
        def retrieve(query, k):
            qb = table.search(qcache[query], vector_column_name="mv").limit(max(k, args.limit))
            if nprobes is not None:
                qb = qb.nprobes(nprobes)
            if refine is not None:
                qb = qb.refine_factor(refine)
            return [r["key"] for r in qb.to_list()]
        return retrieve

    # P1: flat (exhaustive) multivector MaxSim
    if not args.skip_flat:
        print("\n>> P1: flat (exhaustive) multivector MaxSim", flush=True)
        o, el, bt = _score(queries, make_retrieve(), args.k)
        results.append(("flat", o, el))
        print(f"   recall={o['recall']:.3f} mrr={o['mrr']:.3f} ndcg={o['ndcg']:.3f} first-rel={o['fr']:.3f} ({el:.1f}s)")

    # P2+: IVF-PQ index, default + refine/nprobes permutations (refine = exact rerank)
    print("\n>> building IVF-PQ index", flush=True)
    ti = time.perf_counter()
    table.create_index(metric="cosine", vector_column_name="mv")
    print(f"   index built in {time.perf_counter()-ti:.1f}s", flush=True)
    perms = [
        ("ivf_pq", None, None),
        ("ivf_pq+np20", 20, None),
        ("ivf_pq+rf10", None, 10),
        ("ivf_pq+np20+rf10", 20, 10),
        ("ivf_pq+np40+rf20", 40, 20),
    ]
    for label, nprobes, refine in perms:
        try:
            o, el, bt = _score(queries, make_retrieve(nprobes, refine), args.k)
            results.append((label, o, el))
            print(f">> {label}: recall={o['recall']:.3f} mrr={o['mrr']:.3f} ndcg={o['ndcg']:.3f} "
                  f"first-rel={o['fr']:.3f} ({el:.1f}s)", flush=True)
        except Exception as e:
            print(f">> {label} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    print("\n=== LanceDB multivector probe (k={}) ===".format(args.k))
    print(f"{'mode':<12}{'recall':>9}{'mrr':>9}{'ndcg':>9}{'first-rel':>11}{'sec(90q)':>10}")
    for label, o, el in results:
        print(f"{label:<12}{o['recall']:>9.3f}{o['mrr']:>9.3f}{o['ndcg']:>9.3f}{o['fr']:>11.3f}{el:>10.1f}")
    print("\nPG reference (same data/queries): brute 0.830 / indexed-halfvec 0.830 (51.4s) / indexed-bit 0.816")
    print("Per-type (last mode):")
    for t, d in sorted(bt.items()):
        print(f"  {t:<28} recall={metrics.mean(d['recall']):.3f}")


if __name__ == "__main__":
    main()
