#!/usr/bin/env python3
"""Convert LoCoMo (snap-research/locomo, locomo10.json) into the MemoryLayer
eval harness format (corpus.jsonl + qrels.json).

LoCoMo is a long-term *conversational* memory benchmark: 10 multi-session
dialogues (~5.9K turns total) with QA whose ``evidence`` points at the specific
dialogue turns (``dia_id``, e.g. "D1:3") that support the answer. This makes a
discriminating *retrieval* eval at turn granularity:

  * every dialogue turn across ALL conversations -> one memory, keyed
    ``{conv_id}::{dia_id}`` (so the corpus is the full cross-conversation
    distractor set);
  * each QA -> a query whose relevant set is its evidence turns (multi-relevant;
    scored by recall@k / MRR / first-relevant-hit / nDCG);
  * categories: 1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop,
    5=adversarial. Adversarial (answer-abstention, not retrieval) is excluded by
    default; pass --include-adversarial to keep it. Queries are sampled
    round-robin across categories so the mix isn't dominated by single_hop.

Raw file (download once; raw.githubusercontent.com):
    curl -s -o benchmarks/.cache/raw/locomo/locomo10.json \\
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Usage:
    python benchmarks/locomo_convert.py --questions 200
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).parent / ".cache" / "raw" / "locomo" / "locomo10.json"
DEFAULT_OUT = Path(__file__).parent / ".cache" / "locomo"

_CATEGORIES = {"1": "multi_hop", "2": "temporal", "3": "open_domain", "4": "single_hop", "5": "adversarial"}
_SESSION_RE = re.compile(r"^session_(\d+)$")


def _parse_evidence(ev) -> list[str]:
    """Normalize the evidence field to a list of dia_id strings."""
    if ev is None:
        return []
    if isinstance(ev, list):
        vals = ev
    elif isinstance(ev, str):
        try:
            parsed = ast.literal_eval(ev)
        except (ValueError, SyntaxError):
            return []
        vals = parsed if isinstance(parsed, list) else [parsed]
    else:
        vals = [ev]
    return [str(v) for v in vals]


def convert(source: Path, questions: int, out_dir: Path, include_adversarial: bool) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))

    corpus: dict[str, str] = {}
    # Per-conversation dia_id -> corpus key, to resolve evidence within a conv.
    candidates: dict[str, list[dict]] = defaultdict(list)  # category_name -> queries

    for idx, conv in enumerate(data):
        conv_id = str(conv.get("sample_id") or f"conv{idx}")
        c = conv["conversation"]
        dia_to_key: dict[str, str] = {}
        for k, turns in c.items():
            m = _SESSION_RE.match(k)
            if not m or not isinstance(turns, list):
                continue
            date = (c.get(f"session_{m.group(1)}_date_time") or "").strip()
            for t in turns:
                if not isinstance(t, dict) or not t.get("dia_id"):
                    continue
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                key = f"{conv_id}::{t['dia_id']}"
                speaker = t.get("speaker", "?")
                corpus[key] = f"[{date}] {speaker}: {text}" if date else f"{speaker}: {text}"
                dia_to_key[t["dia_id"]] = key

        for q in conv.get("qa", []):
            cat = str(q.get("category", ""))
            if cat == "5" and not include_adversarial:
                continue
            relevant = [dia_to_key[d] for d in _parse_evidence(q.get("evidence")) if d in dia_to_key]
            if not relevant:
                continue  # unresolved/empty evidence -> not a usable retrieval query
            candidates[_CATEGORIES.get(cat, cat)].append(
                {
                    "query_id": f"{conv_id}::q{len(candidates[_CATEGORIES.get(cat, cat)])}",
                    "query": q["question"],
                    "relevant": relevant,
                    "grades": {k: 1.0 for k in relevant},
                    "_question_type": _CATEGORIES.get(cat, cat),
                }
            )

    # Stratified round-robin across categories so single_hop doesn't dominate.
    cats = sorted(candidates)
    queries: list[dict] = []
    depth = 0
    while len(queries) < questions and any(depth < len(candidates[c]) for c in cats):
        for c in cats:
            if depth < len(candidates[c]) and len(queries) < questions:
                queries.append(candidates[c][depth])
        depth += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for key, content in corpus.items():
            fh.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")
    with open(out_dir / "qrels.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "schema_version": 1,
                "_description": f"LoCoMo retrieval ({len(corpus)} turn-memories, {len(queries)} queries)",
                "queries": queries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    from collections import Counter
    mix = Counter(q["_question_type"] for q in queries)
    print(f"Wrote {len(corpus)} turn-memories and {len(queries)} queries to {out_dir}")
    print(f"  category mix: {dict(mix)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LoCoMo to eval corpus+qrels")
    ap.add_argument("--questions", type=int, default=200, help="Number of questions to sample (default 200)")
    ap.add_argument("--source", type=Path, default=RAW, help="Path to raw locomo10.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output cache dir")
    ap.add_argument("--include-adversarial", action="store_true", help="Include category-5 adversarial questions")
    args = ap.parse_args()
    convert(args.source, args.questions, args.out, args.include_adversarial)


if __name__ == "__main__":
    main()
