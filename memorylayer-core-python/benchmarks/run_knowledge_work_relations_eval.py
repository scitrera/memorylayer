#!/usr/bin/env python3
"""Evaluate metadata-first relation acquisition on general knowledge work.

The checked-in v1 fixture spans delivery, operations, policy, research,
architecture, management, organizational work, and analytics.  It evaluates two
separate contracts:

* acquisition precision/recall against hand-authored relation triples; and
* evidence-memory retrieval with the relation arm disabled versus enabled.

No sealed facts are passed through ``RememberInput.relations``.  Every evaluated
edge must be acquired from ``metadata["knowledge_work"]`` by the production
remember pipeline, making this a non-oracle successor to the gbrain ceiling arm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scitrera_app_framework import get_extension

from memorylayer_server.dependencies import shutdown_services
from memorylayer_server.eval import metrics
from memorylayer_server.eval.harness import _EVAL_KEY, _EVAL_WORKSPACE_ID, bootstrap
from memorylayer_server.models.generation import EnrichmentPolicy
from memorylayer_server.models.memory import MemoryType, RecallInput, RecallMode, RememberInput
from memorylayer_server.services.embedding import EXT_EMBEDDING_PROVIDER
from memorylayer_server.services.entity_registry import EXT_ENTITY_REGISTRY_SERVICE
from memorylayer_server.services.entity_relation.metadata import extract_metadata_relations
from memorylayer_server.services.ingest import normalize_connector_metadata
from memorylayer_server.services.llm import EXT_LLM_SERVICE
from memorylayer_server.services.memory import EXT_MEMORY_SERVICE

DEFAULT_DATASET = Path(__file__).with_name("datasets") / "knowledge_work_relations_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty, "python": platform.python_version()}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _triple(candidate) -> tuple[str, str, str]:
    return (
        candidate.source.name or candidate.source.entity_id or "",
        candidate.target.name or candidate.target.entity_id or "",
        candidate.relationship,
    )


def _record_metadata(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Run connector-shaped fixtures through the production boundary adapter."""

    normalized = normalize_connector_metadata(
        record.get("metadata"),
        connector_type=record.get("connector_type"),
        subject_name=record.get("subject_name"),
        subject_type=record.get("subject_type"),
        record_id=record.get("record_id"),
    )
    return normalized.metadata, list(normalized.warnings)


def _acquisition_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted: set[tuple[str, str, str, str]] = set()
    expected: set[tuple[str, str, str, str]] = set()
    errors: list[str] = []
    normalization_warnings: list[str] = []
    by_domain: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        record_id = str(record["id"])
        domain = str(record.get("domain") or "unknown")
        metadata, warnings = _record_metadata(record)
        normalization_warnings.extend(f"{record_id}: {warning}" for warning in warnings)
        extraction = extract_metadata_relations(metadata)
        errors.extend(f"{record_id}: {error}" for error in extraction.errors)
        record_predicted = {(record_id, *_triple(candidate)) for candidate in extraction.candidates}
        record_expected = {(record_id, *tuple(item)) for item in record.get("gold_relations") or []}
        predicted.update(record_predicted)
        expected.update(record_expected)
        by_domain[domain]["tp"] += len(record_predicted & record_expected)
        by_domain[domain]["fp"] += len(record_predicted - record_expected)
        by_domain[domain]["fn"] += len(record_expected - record_predicted)
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    def scores(counts: Counter) -> dict[str, Any]:
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 1.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 1.0
        return {
            "expected": counts["tp"] + counts["fn"],
            "predicted": counts["tp"] + counts["fp"],
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }

    overall = scores(Counter(tp=tp, fp=fp, fn=fn))
    overall["errors"] = errors
    overall["normalization_warnings"] = normalization_warnings
    overall["by_domain"] = {domain: scores(counts) for domain, counts in sorted(by_domain.items())}
    overall["missing"] = [list(item) for item in sorted(expected - predicted)]
    overall["unexpected"] = [list(item) for item in sorted(predicted - expected)]
    return overall


async def _score_queries(memory_service, queries: list[dict[str, Any]], *, relational: bool, k: int, limit: int) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []
    fired = 0
    path_count = 0
    for query in queries:
        started = time.perf_counter()
        result = await memory_service.recall(
            _EVAL_WORKSPACE_ID,
            RecallInput(
                query=str(query["query"]),
                mode=RecallMode.RAG,
                limit=limit,
                min_relevance=0.0,
                include_associations=False,
                traverse_depth=0,
                include_global=False,
                include_global_user=False,
                include_relations=relational,
                include_confidence=True,
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved = [
            memory.metadata.get(_EVAL_KEY)
            for memory in result.memories
            if memory.metadata and memory.metadata.get(_EVAL_KEY)
        ]
        relevant = set(query.get("relevant") or [])
        family = str(query.get("family") or "unknown")
        recall = metrics.recall_at_k(retrieved, relevant, k)
        mrr = metrics.mrr(retrieved, relevant)
        ndcg = metrics.ndcg_at_k(retrieved, {key: 1.0 for key in relevant}, k)
        values["recall"].append(recall)
        values["mrr"].append(mrr)
        values["ndcg"].append(ndcg)
        by_family[family]["recall"].append(recall)
        by_family[family]["mrr"].append(mrr)
        paths = len(result.relation_paths or [])
        fired += int(paths > 0)
        path_count += paths
        latencies.append(elapsed_ms)
        per_query.append(
            {
                "id": query["id"],
                "family": family,
                "query": query["query"],
                "recall_at_k": round(recall, 6),
                "mrr": round(mrr, 6),
                "relation_paths": paths,
                "latency_ms": round(elapsed_ms, 3),
                "retrieved": retrieved[:k],
            }
        )
    return {
        "name": "metadata_relations" if relational else "vector_baseline",
        "relational": relational,
        "recall_at_k": round(_mean(values["recall"]), 6),
        "mrr": round(_mean(values["mrr"]), 6),
        "ndcg_at_k": round(_mean(values["ndcg"]), 6),
        "relation_queries_fired": fired,
        "relation_paths": path_count,
        "latency_ms": {
            "mean": round(_mean(latencies), 3),
            "p50": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "by_family": {
            family: {
                "queries": len(metrics_by_name["recall"]),
                "recall_at_k": round(_mean(metrics_by_name["recall"]), 6),
                "mrr": round(_mean(metrics_by_name["mrr"]), 6),
            }
            for family, metrics_by_name in sorted(by_family.items())
        },
        "per_query": per_query,
    }


def _comparison(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    paired = zip(baseline["per_query"], treatment["per_query"], strict=True)
    outcomes = Counter()
    changed = 0
    for left, right in paired:
        delta = right["mrr"] - left["mrr"]
        outcomes["improved" if delta > 0 else "regressed" if delta < 0 else "tied"] += 1
        changed += int(left["retrieved"] != right["retrieved"])
    return {
        "recall_at_k_delta": round(treatment["recall_at_k"] - baseline["recall_at_k"], 6),
        "mrr_delta": round(treatment["mrr"] - baseline["mrr"], 6),
        "ndcg_at_k_delta": round(treatment["ndcg_at_k"] - baseline["ndcg_at_k"], 6),
        "latency_p95_delta_ms": round(treatment["latency_ms"]["p95"] - baseline["latency_ms"]["p95"], 3),
        "pairwise_mrr": dict(outcomes),
        "rankings_changed": changed,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = list(dataset["records"])
    queries = list(dataset["queries"])
    acquisition = _acquisition_score(records)
    v = await bootstrap(
        embedding_provider=args.embedding_provider,
        reranker=args.reranker,
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
    )
    try:
        memory_service = get_extension(EXT_MEMORY_SERVICE, v)
        registry = get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)
        llm_service = get_extension(EXT_LLM_SERVICE, v)
        provider = get_extension(EXT_EMBEDDING_PROVIDER, v)
        memory_service.cache = None
        memory_service.relational_recall_enabled = True
        llm_service.policy = EnrichmentPolicy.DETERMINISTIC
        writes = Counter()
        for record in records:
            record_metadata, _ = _record_metadata(record)
            memory = await memory_service.remember(
                _EVAL_WORKSPACE_ID,
                RememberInput(
                    content=str(record["content"]),
                    type=MemoryType.SEMANTIC,
                    context_id="_default",
                    metadata={
                        **record_metadata,
                        _EVAL_KEY: record["id"],
                        "source": "knowledge-work-relations-v1",
                        "domain": record.get("domain"),
                    },
                ),
                inline=True,
            )
            report = memory.relation_write_result
            if report is not None:
                writes.update(
                    resolved=report.resolved,
                    unresolved=report.unresolved,
                    rejected=report.rejected,
                    duplicate=report.duplicate,
                )

        # Remove one-time model/transport setup from latency measurements.
        if queries:
            await provider.embed(str(queries[0]["query"]))

        baseline = await _score_queries(memory_service, queries, relational=False, k=args.k, limit=args.limit)
        treatment = await _score_queries(memory_service, queries, relational=True, k=args.k, limit=args.limit)
        controls = [query for query in queries if query.get("family") == "generic_control"]
        relation_queries = [query for query in queries if query.get("family") != "generic_control"]
        baseline_relations = await _score_queries(memory_service, relation_queries, relational=False, k=args.k, limit=args.limit)
        treatment_relations = await _score_queries(memory_service, relation_queries, relational=True, k=args.k, limit=args.limit)
        control_baseline = await _score_queries(memory_service, controls, relational=False, k=args.k, limit=args.limit)
        control_treatment = await _score_queries(memory_service, controls, relational=True, k=args.k, limit=args.limit)
        entities = await registry.list_entities(_EVAL_WORKSPACE_ID, limit=1000)
        return {
            "suite": dataset["name"],
            "version": dataset["version"],
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "revision": _revision(),
            "configuration": {
                "embedding_provider": args.embedding_provider,
                "embedding_model": getattr(provider, "model", None),
                "dimensions": args.dimensions,
                "embed_server_url": args.embed_server_url,
                "reranker": args.reranker,
                "k": args.k,
                "limit": args.limit,
                "enrichment_policy": "deterministic",
            },
            "coverage": {
                "records": len(records),
                "queries": len(queries),
                "relation_queries": len(relation_queries),
                "generic_controls": len(controls),
                "domains": dict(Counter(str(record.get("domain")) for record in records)),
                "ontology_basis": dataset.get("ontology_basis") or [],
            },
            "acquisition": acquisition,
            "persistence": {**dict(writes), "entities": len(entities)},
            "all_queries": {
                "baseline": baseline,
                "treatment": treatment,
                "comparison": _comparison(baseline, treatment),
            },
            "relation_queries": {
                "baseline": baseline_relations,
                "treatment": treatment_relations,
                "comparison": _comparison(baseline_relations, treatment_relations),
            },
            "generic_controls": {
                "baseline": control_baseline,
                "treatment": control_treatment,
                "comparison": _comparison(control_baseline, control_treatment),
            },
        }
    finally:
        await shutdown_services(v)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--embedding-provider", default="hash")
    parser.add_argument("--embed-server-url", default="http://127.0.0.1:8002")
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--reranker", default="none")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json-out")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = Path(args.json_out).resolve() if args.json_out else None
    report = asyncio.run(_run(args))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
