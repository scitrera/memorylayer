#!/usr/bin/env python3
"""Build a smaller, qrels-valid corpus subset for fast retrieval baselines.

Keeps ALL docs that are relevant to any query (so qrels stay valid), then pads
with a deterministic sample of distractors up to --size. All queries are kept.

Example:
    python benchmarks/build_subset.py \
        --corpus benchmarks/.cache/longmemeval/corpus.jsonl \
        --qrels  benchmarks/.cache/longmemeval/qrels.json \
        --size 500 --out-prefix benchmarks/.cache/longmemeval/subset
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    qr = json.loads(Path(args.qrels).read_text(encoding="utf-8"))
    queries = qr.get("queries", qr) if isinstance(qr, dict) else qr

    relevant: set[str] = set()
    for q in queries:
        relevant.update(q["relevant"])

    # Iterate file lines (split on \n only) — NOT str.splitlines(), which also
    # splits on Unicode line separators ( , \x85, ...) embedded in content.
    corpus = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    by_key = {d.get("key"): d for d in corpus}

    rel_present = [k for k in relevant if k in by_key]
    distractors = [k for k in by_key if k not in relevant]

    rng = random.Random(args.seed)
    rng.shuffle(distractors)
    pad = max(0, args.size - len(rel_present))
    chosen = set(rel_present) | set(distractors[:pad])

    subset_docs = [by_key[k] for k in by_key if k in chosen]

    out_corpus = Path(f"{args.out_prefix}_corpus.jsonl")
    out_qrels = Path(f"{args.out_prefix}_qrels.json")
    out_corpus.write_text("\n".join(json.dumps(d) for d in subset_docs) + "\n", encoding="utf-8")
    out_qrels.write_text(json.dumps(qr, indent=2), encoding="utf-8")

    print(f"queries={len(queries)} relevant={len(relevant)} (present={len(rel_present)})")
    print(f"subset corpus={len(subset_docs)} (relevant + {pad} distractors)")
    print(f"wrote {out_corpus}")
    print(f"wrote {out_qrels}")


if __name__ == "__main__":
    main()
