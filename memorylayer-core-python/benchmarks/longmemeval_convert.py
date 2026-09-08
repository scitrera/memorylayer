#!/usr/bin/env python3
"""Convert LongMemEval (HF: xiaowu0162/longmemeval, the ``_s`` set) into the
MemoryLayer eval harness format (corpus.jsonl + qrels.json).

LongMemEval is a memory-specific benchmark: each question comes with a haystack
of chat sessions, only a few of which (``answer_session_ids``) contain the
evidence. This converter makes a discriminating *retrieval* eval:

  * each haystack session -> one memory (rendered turns), keyed per question so
    sessions never collide across questions;
  * the corpus is the union of sampled questions' haystacks (real distractors);
  * each question -> a query whose relevant set is its evidence sessions
    (multi-relevant; measured by recall@k / MRR / first-relevant-hit / nDCG).

A per-question session cap bounds embedding cost while always keeping every
evidence session. Reads the locally cached raw file (download_sources.py).

Usage:
    python benchmarks/longmemeval_convert.py --questions 75 --max-sessions 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW = Path(__file__).parent / ".cache" / "raw" / "longmemeval" / "longmemeval_s"
DEFAULT_OUT = Path(__file__).parent / ".cache" / "longmemeval"


def _render_session(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        role = t.get("role", "?")
        content = (t.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def convert(source: Path, questions: int, max_sessions: int, out_dir: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))

    # Stratified round-robin across question types so the sample isn't dominated
    # by whichever type the dataset happens to list first.
    from collections import defaultdict

    by_type: dict[str, list] = defaultdict(list)
    for it in data:
        by_type[it.get("question_type", "unknown")].append(it)
    types = sorted(by_type)
    items: list[dict] = []
    depth = 0
    while len(items) < questions and any(depth < len(by_type[t]) for t in types):
        for t in types:
            if depth < len(by_type[t]) and len(items) < questions:
                items.append(by_type[t][depth])
        depth += 1

    corpus: dict[str, str] = {}
    qrels_queries: list[dict] = []

    for item in items:
        qid = item["question_id"]
        answer_ids = set(item.get("answer_session_ids") or [])
        sids = item.get("haystack_session_ids") or []
        sessions = item.get("haystack_sessions") or []

        # Always keep evidence sessions; cap distractors to bound embedding cost.
        order = sorted(range(len(sids)), key=lambda i: sids[i] not in answer_ids)
        kept = order[:max_sessions]
        # Ensure all evidence sessions are kept even if beyond the cap.
        kept = sorted(set(kept) | {i for i in range(len(sids)) if sids[i] in answer_ids})

        relevant: list[str] = []
        for i in kept:
            content = _render_session(sessions[i])
            if not content.strip():
                continue  # skip empty sessions (no usable turns)
            key = f"{qid}::{sids[i]}"
            corpus[key] = content
            if sids[i] in answer_ids:
                relevant.append(key)

        if relevant:  # only usable as a query if its evidence survived
            qrels_queries.append(
                {
                    "query_id": qid,
                    "query": item["question"],
                    "relevant": relevant,
                    "grades": {k: 1.0 for k in relevant},
                    "_question_type": item.get("question_type"),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for key, content in corpus.items():
            fh.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")
    with open(out_dir / "qrels.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": 1,
                "_description": f"LongMemEval-s retrieval ({len(corpus)} sessions, {len(qrels_queries)} queries)",
                "queries": qrels_queries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(corpus)} session-memories and {len(qrels_queries)} queries to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LongMemEval-s to eval corpus+qrels")
    ap.add_argument("--questions", type=int, default=75, help="Number of questions to sample (default 75)")
    ap.add_argument("--max-sessions", type=int, default=60, help="Per-question session cap (evidence always kept)")
    ap.add_argument("--source", type=Path, default=RAW, help="Path to raw longmemeval_s file")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output cache dir")
    args = ap.parse_args()
    convert(args.source, args.questions, args.max_sessions, args.out)


if __name__ == "__main__":
    main()
