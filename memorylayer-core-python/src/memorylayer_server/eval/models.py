"""Data models for the retrieval-eval harness.

Plain dataclasses (no pydantic) kept deliberately light so the harness and the
metric layer have no heavy dependencies. Qrels and corpus files are loaded from
JSON / JSONL on disk; reports serialize back to plain dicts for ``--json`` output
and for any future baseline persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CorpusDoc:
    """A single fixture-corpus document.

    ``key`` is a stable identifier (e.g. ``"fastapi-web"``) that qrels reference
    and that the harness stashes in memory metadata as ``eval_key`` so retrieved
    memories can be mapped back to their corpus key.
    """

    key: str
    content: str
    type: str = "semantic"
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> CorpusDoc:
        return cls(
            key=d["key"],
            content=d["content"],
            type=d.get("type", "semantic"),
            importance=float(d.get("importance", 0.5)),
            tags=list(d.get("tags", [])),
        )


@dataclass
class Qrel:
    """A query and its ground-truth relevant corpus keys."""

    query: str
    relevant: list[str]
    query_id: str | None = None
    expected_top1: str | None = None
    grades: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Qrel:
        return cls(
            query=d["query"],
            relevant=list(d.get("relevant", [])),
            query_id=d.get("query_id"),
            expected_top1=d.get("expected_top1"),
            grades={k: float(v) for k, v in d["grades"].items()} if d.get("grades") else None,
        )

    def grade_map(self) -> dict[str, float]:
        """Graded relevance map; binary (all relevant -> grade 1) when no grades given."""
        if self.grades:
            return dict(self.grades)
        return {key: 1.0 for key in self.relevant}


@dataclass
class QueryResult:
    """Per-query scoring outcome."""

    query_id: str
    query: str
    retrieved: list[str]
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    first_relevant_hit: int = 0
    expected_top1_hit: int | None = None
    latency_ms: float = 0.0
    errored: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "retrieved": self.retrieved,
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "first_relevant_hit": self.first_relevant_hit,
            "expected_top1_hit": self.expected_top1_hit,
            "latency_ms": round(self.latency_ms, 2),
            "errored": self.errored,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregate result of an eval run."""

    k: int
    queries: list[QueryResult]
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    first_relevant_hit_rate: float = 0.0
    expected_top1_hit_rate: float = 0.0
    expected_top1_denominator: int = 0
    queries_total: int = 0
    queries_run: int = 0
    queries_errored: int = 0
    mean_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "k": self.k,
            "queries_total": self.queries_total,
            "queries_run": self.queries_run,
            "queries_errored": self.queries_errored,
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "first_relevant_hit_rate": round(self.first_relevant_hit_rate, 4),
            "expected_top1_hit_rate": round(self.expected_top1_hit_rate, 4),
            "expected_top1_denominator": self.expected_top1_denominator,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "queries": [q.to_dict() for q in self.queries],
        }


@dataclass
class Thresholds:
    """Correctness-gate floors. A run passes only if every metric meets its floor.

    Defaults are tuned for the bundled fixture corpus (which the hashing
    embedding retrieves cleanly) with margin below observed values, so an actual
    ranking regression trips the gate while normal variation does not.
    """

    recall_at_k: float = 0.80
    first_relevant_hit: float = 0.80
    expected_top1: float = 0.60
    k: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> Thresholds:
        return cls(
            recall_at_k=float(d.get("recall_at_k", 0.80)),
            first_relevant_hit=float(d.get("first_relevant_hit", 0.80)),
            expected_top1=float(d.get("expected_top1", 0.60)),
            k=int(d.get("k", 5)),
        )


@dataclass
class Breach:
    """A single failed threshold."""

    metric: str
    observed: float
    threshold: float

    def to_dict(self) -> dict:
        return {"metric": self.metric, "observed": round(self.observed, 4), "threshold": self.threshold}


@dataclass
class GateResult:
    """Verdict of comparing an EvalReport against Thresholds."""

    verdict: str  # "pass" | "fail"
    breaches: list[Breach]
    report: EvalReport
    thresholds: Thresholds

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "verdict": self.verdict,
            "breaches": [b.to_dict() for b in self.breaches],
            "thresholds": {
                "recall_at_k": self.thresholds.recall_at_k,
                "first_relevant_hit": self.thresholds.first_relevant_hit,
                "expected_top1": self.thresholds.expected_top1,
                "k": self.thresholds.k,
            },
            "report": self.report.to_dict(),
        }


def load_corpus(path: str | Path) -> list[CorpusDoc]:
    """Load a JSONL corpus file (one document object per line)."""
    docs: list[CorpusDoc] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(CorpusDoc.from_dict(json.loads(line)))
    return docs


def load_qrels(path: str | Path) -> list[Qrel]:
    """Load a qrels JSON file (object with a ``queries`` list)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    out: list[Qrel] = []
    for i, q in enumerate(queries):
        qrel = Qrel.from_dict(q)
        if qrel.query_id is None:
            qrel.query_id = f"q{i + 1}"
        out.append(qrel)
    return out
