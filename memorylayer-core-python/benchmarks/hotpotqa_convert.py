#!/usr/bin/env python3
"""Convert HotpotQA (HF hotpotqa/hotpot_qa, distractor/validation) into the
MemoryLayer eval harness format (corpus.jsonl + qrels.json).

HotpotQA is a multi-hop QA benchmark: each question's ``context`` has 10
Wikipedia paragraphs (2 gold + 8 distractors), and ``supporting_facts`` names
the gold paragraph titles. This makes a multi-hop *retrieval* eval:

  * each context paragraph -> one memory, keyed ``{qid}::{title}`` (so the
    corpus is the union of sampled questions' paragraphs — real distractors);
  * each question -> a query whose relevant set is its supporting-fact titles
    (multi-hop: usually 2 gold; scored by recall@k / MRR / first-rel / nDCG);
  * stratified round-robin across type (bridge / comparison).

Raw parquet (huggingface.co is reachable):
    curl -sL -o benchmarks/.cache/raw/hotpotqa/validation.parquet \\
      https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation/0.parquet

Usage (needs pyarrow):
    python benchmarks/hotpotqa_convert.py --questions 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path(__file__).parent / ".cache" / "raw" / "hotpotqa" / "validation.parquet"
DEFAULT_OUT = Path(__file__).parent / ".cache" / "hotpotqa"


def convert(source: Path, questions: int, out_dir: Path) -> None:
    rows = pq.read_table(source).to_pylist()

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r.get("type", "unknown")].append(r)
    types = sorted(by_type)

    corpus: dict[str, str] = {}
    queries: list[dict] = []
    depth = 0
    while len(queries) < questions and any(depth < len(by_type[t]) for t in types):
        for t in types:
            if depth >= len(by_type[t]) or len(queries) >= questions:
                continue
            r = by_type[t][depth]
            qid = r["id"]
            ctx = r["context"]
            titles = ctx["title"]
            sentences = ctx["sentences"]
            title_to_key: dict[str, str] = {}
            for title, sents in zip(titles, sentences):
                para = " ".join(s.strip() for s in sents if s and s.strip())
                if not para:
                    continue
                key = f"{qid}::{title}"
                corpus[key] = f"{title}. {para}"
                title_to_key[title] = key
            gold_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))  # unique, order-preserving
            relevant = [title_to_key[t2] for t2 in gold_titles if t2 in title_to_key]
            if not relevant:
                continue
            queries.append(
                {
                    "query_id": qid,
                    "query": r["question"],
                    "relevant": relevant,
                    "grades": {k: 1.0 for k in relevant},
                    "_question_type": r.get("type", "unknown"),
                }
            )
        depth += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for key, content in corpus.items():
            fh.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")
    with open(out_dir / "qrels.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": 1,
                "_description": f"HotpotQA distractor retrieval ({len(corpus)} paragraphs, {len(queries)} queries)",
                "queries": queries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    from collections import Counter
    mix = Counter(q["_question_type"] for q in queries)
    avg_rel = sum(len(q["relevant"]) for q in queries) / max(len(queries), 1)
    print(f"Wrote {len(corpus)} paragraph-memories and {len(queries)} queries to {out_dir}")
    print(f"  type mix: {dict(mix)} | avg relevant/query: {avg_rel:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert HotpotQA distractor to eval corpus+qrels")
    ap.add_argument("--questions", type=int, default=200, help="Number of questions to sample (default 200)")
    ap.add_argument("--source", type=Path, default=RAW, help="Path to raw validation.parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output cache dir")
    args = ap.parse_args()
    convert(args.source, args.questions, args.out)


if __name__ == "__main__":
    main()
