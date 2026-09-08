#!/usr/bin/env python3
"""Convert GammaCorpus-Fact-QA (HF: rubenroy/GammaCorpus-Fact-QA-450k) into the
MemoryLayer eval harness format (corpus.jsonl + qrels.json).

Each Q/A row becomes one memory whose content is the fact card "{question}
{answer}" (a realistic stored fact); the query is the question and the single
relevant item is that card. This is a broad semantic-retrieval benchmark (it
does not exercise temporal/entity/graph features — LongMemEval covers those).

Pulls rows from the HF datasets-server /rows endpoint (no full 450k download),
dedups by normalized question, and writes a gitignored cache. Network only;
run outside the sandbox.

Usage:
    python benchmarks/gammacorpus_convert.py --rows 2000 --queries 400
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "rubenroy/GammaCorpus-Fact-QA-450k"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_OUT = Path(__file__).parent / ".cache" / "gammacorpus"
LOCAL_RAW = Path(__file__).parent / ".cache" / "raw" / "gammacorpus" / "gammacorpus-fact-qa-450k.jsonl"
_WS = re.compile(r"\s+")


def _read_local(path: Path, n: int) -> list[dict]:
    """Read the first ``n`` rows from the locally cached JSONL (offline)."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def _fetch_rows(n: int) -> list[dict]:
    """Page through the datasets-server /rows endpoint (100 rows/page)."""
    out: list[dict] = []
    offset = 0
    while len(out) < n:
        length = min(100, n - len(out))
        qs = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": length})
        with urllib.request.urlopen(f"{ROWS_URL}?{qs}", timeout=60) as resp:  # noqa: S310 - fixed HF host
            data = json.loads(resp.read())
        page = data.get("rows", [])
        if not page:
            break
        out.extend(r["row"] for r in page)
        offset += len(page)
    return out


def convert(rows: int, queries: int, out_dir: Path) -> None:
    # Prefer the locally cached raw file (download_sources.py); fall back to HF API.
    raw = _read_local(LOCAL_RAW, rows) if LOCAL_RAW.exists() else _fetch_rows(rows)

    # Dedup by normalized question; keep first occurrence.
    seen: set[str] = set()
    cards: list[dict] = []
    for r in raw:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        norm = _WS.sub(" ", q.lower())
        if norm in seen:
            continue
        seen.add(norm)
        cards.append({"key": f"fact-{len(cards):05d}", "question": q, "answer": a})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps({"key": c["key"], "content": f"{c['question']} {c['answer']}"}, ensure_ascii=False) + "\n")

    # Queries: a deterministic spread across the corpus (every step-th card).
    n_q = min(queries, len(cards))
    step = max(1, len(cards) // n_q)
    query_cards = cards[::step][:n_q]
    qrels = {
        "schema_version": 1,
        "_description": f"GammaCorpus-Fact-QA retrieval qrels ({len(cards)} cards, {len(query_cards)} queries)",
        "queries": [{"query_id": c["key"], "query": c["question"], "relevant": [c["key"]], "expected_top1": c["key"]} for c in query_cards],
    }
    with open(out_dir / "qrels.json", "w", encoding="utf-8") as fh:
        json.dump(qrels, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(cards)} corpus docs and {len(query_cards)} queries to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert GammaCorpus Fact-QA to eval corpus+qrels")
    ap.add_argument("--rows", type=int, default=2000, help="Source rows to pull from HF (default 2000)")
    ap.add_argument("--queries", type=int, default=400, help="Number of queries to emit (default 400)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output cache dir")
    args = ap.parse_args()
    convert(args.rows, args.queries, args.out)


if __name__ == "__main__":
    main()
