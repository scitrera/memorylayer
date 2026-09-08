#!/usr/bin/env python3
"""Evaluate deterministic-memory rollout candidates on LoCoMo and BrainBench.

The runner has two deliberately separate jobs:

* ``locomo`` checks that deterministic policy, confidence reporting, relation
  routing, and a practical hard token budget do not regress ordinary retrieval.
* ``relations`` imports gbrain-evals' public ``world-v1`` corpus and relational
  query templates. It reports content-only pattern acquisition separately from
  an ``oracle_structured`` ceiling where the corpus' sealed ``_facts`` are treated
  as trusted structured-ingest metadata. The latter measures relation storage and
  retrieval, not extraction quality.

Every arm uses the same in-process MemoryLayer service, corpus, embeddings, and
query order. Results include dataset hashes, revision/configuration, generation
counts, confidence distribution, relation-path firing, and latency percentiles.

Examples:
    python benchmarks/run_deterministic_memory_eval.py --suite locomo
    python benchmarks/run_deterministic_memory_eval.py --suite relations \
        --world-dir /path/to/gbrain-evals/eval/data/world-v1
    python benchmarks/run_deterministic_memory_eval.py --suite all \
        --json-out benchmarks/.cache/deterministic-memory-eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import platform
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scitrera_app_framework import get_extension

from memorylayer_server.dependencies import shutdown_services
from memorylayer_server.eval import metrics
from memorylayer_server.eval.harness import _EVAL_KEY, _EVAL_WORKSPACE_ID, bootstrap, ingest_corpus
from memorylayer_server.eval.models import load_corpus
from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.models.entity_relation import EntityRelationInput
from memorylayer_server.models.generation import EnrichmentPolicy
from memorylayer_server.models.memory import MemoryType, RecallInput, RecallMode, RememberInput
from memorylayer_server.services.embedding import EXT_EMBEDDING_PROVIDER
from memorylayer_server.services.entity_registry import EXT_ENTITY_REGISTRY_SERVICE
from memorylayer_server.services.llm import EXT_LLM_SERVICE
from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
from memorylayer_server.services.storage import EXT_STORAGE_BACKEND


@dataclass(frozen=True)
class Variant:
    name: str
    policy: EnrichmentPolicy
    relational: bool = False
    budget_tokens: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
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
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _load_qrels(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("queries", value) if isinstance(value, dict) else value


async def _score_variant(
    memory_service,
    llm_service,
    queries: list[dict[str, Any]],
    variant: Variant,
    *,
    k: int,
    limit: int,
) -> dict[str, Any]:
    memory_service.cache = None
    memory_service.relational_recall_enabled = variant.relational
    llm_service.policy = variant.policy

    metric_values: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    confidence = Counter()
    generation = Counter(
        calls=0,
        denied_calls=0,
        budget_denied_calls=0,
        input_tokens=0,
        output_tokens=0,
    )
    budget = Counter(used_tokens=0, truncated_items=0, omitted_items=0, violations=0)
    relation_paths = 0
    fired_queries = 0
    rankings: dict[str, list[str]] = {}
    per_query: list[dict[str, Any]] = []

    for index, query in enumerate(queries):
        qtype = str(query.get("_question_type") or query.get("family") or "all")
        raw_query_id = str(query.get("query_id") or query.get("id") or f"q{index + 1}")
        # Some imported suites scope IDs only within a query family. Including
        # the family keeps rankings and comparisons lossless across mixed sets.
        query_id = f"{raw_query_id}::{qtype}"
        relevant = set(query.get("relevant") or [])
        grades = query.get("grades") or {key: 1.0 for key in relevant}
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
                include_relations=variant.relational,
                include_confidence=True,
                budget_tokens=variant.budget_tokens,
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved = [
            memory.metadata.get(_EVAL_KEY)
            for memory in result.memories
            if memory.metadata and memory.metadata.get(_EVAL_KEY)
        ]
        rankings[query_id] = retrieved
        recall = metrics.recall_at_k(retrieved, relevant, k)
        mrr = metrics.mrr(retrieved, relevant)
        ndcg = metrics.ndcg_at_k(retrieved, grades, k)
        first = float(metrics.first_relevant_hit(retrieved, relevant))
        for name, value in (("recall", recall), ("mrr", mrr), ("ndcg", ndcg), ("first_relevant", first)):
            metric_values[name].append(value)
            by_type[qtype][name].append(value)
        latencies.append(elapsed_ms)
        confidence[str(result.retrieval_confidence or "unset")] += 1
        summary = result.generation_summary
        if summary is not None:
            generation["calls"] += summary.calls
            generation["denied_calls"] += summary.denied_calls
            generation["budget_denied_calls"] += summary.budget_denied_calls
            generation["input_tokens"] += summary.input_tokens
            generation["output_tokens"] += summary.output_tokens
        budget_summary = result.budget_summary
        if budget_summary is not None:
            budget["used_tokens"] += budget_summary.used
            budget["truncated_items"] += budget_summary.truncated_items
            budget["omitted_items"] += budget_summary.omitted_items
            if variant.budget_tokens is not None and budget_summary.used > variant.budget_tokens:
                budget["violations"] += 1
        paths = len(result.relation_paths or [])
        relation_paths += paths
        fired_queries += int(paths > 0)
        per_query.append(
            {
                "query_id": query_id,
                "type": qtype,
                "recall_at_k": round(recall, 6),
                "mrr": round(mrr, 6),
                "latency_ms": round(elapsed_ms, 3),
                "confidence": str(result.retrieval_confidence or "unset"),
                "relation_paths": paths,
                "retrieved": retrieved[:k],
            }
        )

    return {
        "name": variant.name,
        "policy": variant.policy.value,
        "relational": variant.relational,
        "budget_tokens": variant.budget_tokens,
        "queries": len(queries),
        "overall": {name: round(_mean(values), 6) for name, values in metric_values.items()},
        "by_type": {
            qtype: {name: round(_mean(values), 6) for name, values in typed.items()}
            for qtype, typed in sorted(by_type.items())
        },
        "latency_ms": {
            "mean": round(_mean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "confidence": dict(sorted(confidence.items())),
        "generation": dict(generation),
        "budget": dict(budget),
        "relation_queries_fired": fired_queries,
        "relation_paths": relation_paths,
        "rankings": rankings,
        "per_query": per_query,
    }


async def _warm_query_embeddings(memory_service, queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Populate the shared query cache before timing retrieval arms.

    This keeps feature-arm latency comparisons from charging only the first arm
    for remote embedding transport. The warmup duration is still reported so
    deployment-level latency can be reconstructed rather than hidden.
    """
    started = time.perf_counter()
    for query in queries:
        await memory_service.embedding.embed(str(query["query"]))
    return {
        "queries": len(queries),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _comparison(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    common = sorted(set(baseline["rankings"]) & set(treatment["rankings"]))
    changed = sum(baseline["rankings"][key] != treatment["rankings"][key] for key in common)
    return {
        "baseline": baseline["name"],
        "treatment": treatment["name"],
        "recall_delta": round(treatment["overall"]["recall"] - baseline["overall"]["recall"], 6),
        "mrr_delta": round(treatment["overall"]["mrr"] - baseline["overall"]["mrr"], 6),
        "p95_latency_delta_ms": round(treatment["latency_ms"]["p95"] - baseline["latency_ms"]["p95"], 3),
        "rankings_changed": changed,
        "queries_compared": len(common),
    }


async def _run_locomo(args: argparse.Namespace) -> dict[str, Any]:
    corpus_path = Path(args.locomo_corpus).resolve()
    qrels_path = Path(args.locomo_qrels).resolve()
    docs = load_corpus(corpus_path)
    queries = _load_qrels(qrels_path)
    v = await bootstrap(
        embedding_provider=args.embedding_provider,
        reranker=args.reranker,
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
    )
    try:
        await ingest_corpus(v, docs, clean=True, progress=args.progress)
        memory_service = get_extension(EXT_MEMORY_SERVICE, v)
        llm_service = get_extension(EXT_LLM_SERVICE, v)
        warmup = await _warm_query_embeddings(memory_service, queries)
        variants = (
            Variant("compatibility", EnrichmentPolicy.GENERATIVE),
            Variant(
                "deterministic_budgeted",
                EnrichmentPolicy.DETERMINISTIC,
                relational=True,
                budget_tokens=args.budget_tokens,
            ),
        )
        arms = [
            await _score_variant(memory_service, llm_service, queries, variant, k=args.k, limit=max(args.k, args.limit))
            for variant in variants
        ]
        return {
            "suite": "locomo",
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256(corpus_path),
            "qrels": str(qrels_path),
            "qrels_sha256": _sha256(qrels_path),
            "documents": len(docs),
            "queries": len(queries),
            "query_embedding_warmup": warmup,
            "arms": arms,
            "comparison": _comparison(arms[0], arms[1]),
        }
    finally:
        await shutdown_services(v)


def _load_world(world_dir: Path) -> list[dict[str, Any]]:
    pages = []
    for path in sorted(world_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        page = json.loads(path.read_text(encoding="utf-8"))
        page["_path"] = str(path)
        pages.append(page)
    return pages


def _world_content(page: dict[str, Any]) -> str:
    return "\n\n".join(
        value.strip()
        for value in (str(page.get("title") or ""), str(page.get("compiled_truth") or ""), str(page.get("timeline") or ""))
        if value.strip()
    )


def _entity_type(page: dict[str, Any]) -> EntityType:
    page_type = str(page.get("type") or page.get("_facts", {}).get("type") or "concept")
    return {
        "person": EntityType.PERSON,
        "company": EntityType.ORG,
        "meeting": EntityType.EVENT,
        "project": EntityType.PROJECT,
        "place": EntityType.PLACE,
    }.get(page_type, EntityType.CONCEPT)


def _world_queries(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = {str(page["slug"]) for page in pages}
    queries: list[dict[str, Any]] = []

    def add(family: str, text: str, target: list[str], evidence: list[str]) -> None:
        relevant = list(dict.fromkeys(slug for slug in target if slug in existing))
        if relevant:
            queries.append(
                {
                    "query_id": f"{family}-{len(queries) + 1:04d}",
                    "query": text,
                    "relevant": relevant,
                    "grades": {slug: 1.0 for slug in relevant},
                    "_question_type": family,
                    "evidence_relevant": list(dict.fromkeys(slug for slug in evidence if slug in existing)),
                }
            )

    for page in pages:
        facts = page.get("_facts") or {}
        if facts.get("type") == "meeting":
            add("attended", f"Who attended {page['title']}?", list(facts.get("attendees") or []), [page["slug"]])
        elif facts.get("type") == "company":
            add(
                "works_at",
                f"Who works at {page['title']}?",
                list(facts.get("employees") or []) + list(facts.get("founders") or []),
                [page["slug"]],
            )
            add("invested_in", f"Who invested in {page['title']}?", list(facts.get("investors") or []), [page["slug"]])
            add("advises", f"Who advises {page['title']}?", list(facts.get("advisors") or []), [page["slug"]])
    return queries


def _world_control_queries(pages: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    """Build a deterministic generic-query control slice with exact page qrels."""
    if not pages or limit <= 0:
        return []
    step = max(1, len(pages) // limit)
    selected = pages[::step][:limit]
    return [
        {
            "query_id": f"generic-control-{index:04d}",
            "query": f"What information is available about {page['title']}?",
            "relevant": [page["slug"]],
            "grades": {page["slug"]: 1.0},
            "_question_type": "generic_control",
        }
        for index, page in enumerate(selected, start=1)
    ]


def _structured_relations(page: dict[str, Any], entity_ids: dict[str, str]) -> list[EntityRelationInput]:
    facts = page.get("_facts") or {}
    slug = str(page["slug"])
    values: list[EntityRelationInput] = []

    def add(source_slug: str, target_slug: str, relationship: str) -> None:
        if source_slug in entity_ids and target_slug in entity_ids:
            values.append(
                EntityRelationInput(
                    source_entity_id=entity_ids[source_slug],
                    target_entity_id=entity_ids[target_slug],
                    relationship=relationship,
                    confidence=1.0,
                    extraction_method="benchmark:structured_ingest",
                )
            )

    if facts.get("type") == "person":
        role = str(facts.get("role") or "")
        primary = facts.get("primary_affiliation")
        secondary = list(facts.get("secondary_affiliations") or [])
        if primary:
            if role in {"founder", "engineer"}:
                add(slug, str(primary), "employed_by")
            elif role == "advisor":
                add(slug, str(primary), "advises")
        for affiliation in secondary:
            if role == "partner":
                add(slug, str(affiliation), "invested_in")
            elif role == "advisor":
                add(slug, str(affiliation), "advises")
    elif facts.get("type") == "meeting":
        for attendee in facts.get("attendees") or []:
            add(str(attendee), slug, "attended")
    return values


async def _score_evidence(
    memory_service,
    llm_service,
    queries: list[dict[str, Any]],
    variant: Variant,
    *,
    k: int,
    limit: int,
) -> dict[str, Any]:
    evidence_queries = [
        {
            **query,
            "relevant": query.get("evidence_relevant") or query["relevant"],
            "grades": {key: 1.0 for key in (query.get("evidence_relevant") or query["relevant"])},
        }
        for query in queries
    ]
    result = await _score_variant(memory_service, llm_service, evidence_queries, variant, k=k, limit=limit)
    result["name"] = f"{variant.name}_evidence"
    return result


async def _run_relations(args: argparse.Namespace) -> dict[str, Any]:
    world_dir = Path(args.world_dir).resolve()
    pages = _load_world(world_dir)
    queries = _world_queries(pages)
    control_queries = _world_control_queries(pages)
    v = await bootstrap(
        embedding_provider=args.embedding_provider,
        reranker=args.reranker,
        dimensions=args.dimensions,
        embed_server_url=args.embed_server_url,
    )
    try:
        storage = get_extension(EXT_STORAGE_BACKEND, v)
        provider = get_extension(EXT_EMBEDDING_PROVIDER, v)
        registry = get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)
        memory_service = get_extension(EXT_MEMORY_SERVICE, v)
        llm_service = get_extension(EXT_LLM_SERVICE, v)
        llm_service.policy = EnrichmentPolicy.DETERMINISTIC

        contents = [_world_content(page) for page in pages]
        embeddings: list[list[float]] = []
        for offset in range(0, len(contents), 64):
            embeddings.extend(await provider.embed_batch(contents[offset : offset + 64]))

        memories: dict[str, Any] = {}
        for page, content, embedding in zip(pages, contents, embeddings, strict=True):
            memory = await storage.create_memory(
                _EVAL_WORKSPACE_ID,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    context_id="_default",
                    metadata={_EVAL_KEY: page["slug"], "source": "gbrain-evals/world-v1"},
                ),
            )
            memories[page["slug"]] = await storage.update_memory(
                _EVAL_WORKSPACE_ID,
                memory.id,
                embedding=embedding,
            )

        entity_ids: dict[str, str] = {}
        for page in pages:
            entity = await registry.upsert(
                _EVAL_WORKSPACE_ID,
                str(page["title"]),
                _entity_type(page),
                aliases=[str(page["slug"]), str((page.get("_facts") or {}).get("name") or page["title"])],
                confidence=1.0,
                provenance={"source": "gbrain-evals/world-v1"},
                representative_memory_id=memories[page["slug"]].id,
            )
            entity_ids[page["slug"]] = entity.id

        warmup = await _warm_query_embeddings(memory_service, queries + control_queries)

        baseline_variant = Variant("baseline", EnrichmentPolicy.DETERMINISTIC)
        empty_arm_variant = Variant("relation_arm_no_edges", EnrichmentPolicy.DETERMINISTIC, relational=True)
        baseline = await _score_variant(
            memory_service,
            llm_service,
            queries,
            baseline_variant,
            k=args.k,
            limit=max(args.k, args.limit),
        )
        empty_arm = await _score_variant(
            memory_service,
            llm_service,
            queries,
            empty_arm_variant,
            k=args.k,
            limit=max(args.k, args.limit),
        )
        control_baseline = await _score_variant(
            memory_service,
            llm_service,
            control_queries,
            Variant("generic_control_baseline", EnrichmentPolicy.DETERMINISTIC),
            k=args.k,
            limit=max(args.k, args.limit),
        )

        pattern_counts = Counter()
        for memory in memories.values():
            report = await memory_service.entity_relation_service.write_for_memory(memory, [], include_patterns=True)
            pattern_counts.update(
                resolved=report.resolved,
                unresolved=report.unresolved,
                rejected=report.rejected,
                duplicate=report.duplicate,
            )
        pattern_variant = Variant("content_patterns", EnrichmentPolicy.DETERMINISTIC, relational=True)
        pattern = await _score_variant(
            memory_service,
            llm_service,
            queries,
            pattern_variant,
            k=args.k,
            limit=max(args.k, args.limit),
        )

        structured_counts = Counter()
        for page in pages:
            values = _structured_relations(page, entity_ids)
            if not values:
                continue
            report = await memory_service.entity_relation_service.write_for_memory(
                memories[page["slug"]],
                values,
                include_patterns=False,
            )
            structured_counts.update(
                resolved=report.resolved,
                unresolved=report.unresolved,
                rejected=report.rejected,
                duplicate=report.duplicate,
            )
        structured_variant = Variant("oracle_structured", EnrichmentPolicy.DETERMINISTIC, relational=True)
        structured = await _score_variant(
            memory_service,
            llm_service,
            queries,
            structured_variant,
            k=args.k,
            limit=max(args.k, args.limit),
        )
        structured_evidence = await _score_evidence(
            memory_service,
            llm_service,
            queries,
            structured_variant,
            k=args.k,
            limit=max(args.k, args.limit),
        )
        control_structured = await _score_variant(
            memory_service,
            llm_service,
            control_queries,
            Variant("generic_control_relations", EnrichmentPolicy.DETERMINISTIC, relational=True),
            k=args.k,
            limit=max(args.k, args.limit),
        )

        first_meeting = next(page for page in pages if (page.get("_facts") or {}).get("type") == "meeting")
        safety_queries = {
            "generic": f"What information is available about {first_meeting['title']}?",
            "missing_seed": "Who works at Nonexistent Quasar Holdings?",
            "wrong_relation_type": f"Who works at {first_meeting['title']}?",
        }
        safety_controls: dict[str, Any] = {}
        for name, query in safety_queries.items():
            recalled = await memory_service.entity_relation_service.recall(_EVAL_WORKSPACE_ID, query)
            safety_controls[name] = {
                "query": query,
                "memories": len(recalled.memories),
                "paths": len(recalled.paths),
                "unresolved_seed": recalled.unresolved_seed,
            }

        corpus_hash = hashlib.sha256()
        for page in pages:
            corpus_hash.update(Path(page["_path"]).read_bytes())
        return {
            "suite": "gbrain_world_relations",
            "world_dir": str(world_dir),
            "corpus_sha256": corpus_hash.hexdigest(),
            "documents": len(pages),
            "queries": len(queries),
            "generic_control_queries": len(control_queries),
            "query_mix": dict(sorted(Counter(query["_question_type"] for query in queries).items())),
            "query_embedding_warmup": warmup,
            "relation_writes": {
                "content_patterns": dict(pattern_counts),
                "oracle_structured": dict(structured_counts),
            },
            "arms": [
                baseline,
                empty_arm,
                pattern,
                structured,
                structured_evidence,
                control_baseline,
                control_structured,
            ],
            "comparisons": [
                _comparison(baseline, empty_arm),
                _comparison(baseline, pattern),
                _comparison(baseline, structured),
                _comparison(control_baseline, control_structured),
            ],
            "safety_controls": safety_controls,
            "methodology_note": (
                "oracle_structured consumes sealed _facts as trusted explicit relation input; "
                "it measures storage/routing/retrieval ceiling, not extraction quality"
            ),
        }
    finally:
        await shutdown_services(v)


def _compact_arm(arm: dict[str, Any], k: int) -> str:
    overall = arm["overall"]
    latency = arm["latency_ms"]
    return (
        f"{arm['name']:<28} R@{k}={overall['recall']:.3f}  MRR={overall['mrr']:.3f}  "
        f"p95={latency['p95']:.1f}ms  paths={arm['relation_paths']}  gen={arm['generation'].get('calls', 0)}"
    )


async def _amain(args: argparse.Namespace) -> int:
    if not args.progress:
        logging.disable(logging.INFO)
    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at_epoch": time.time(),
        "revision": _revision(),
        "configuration": {
            "suite": args.suite,
            "embedding_provider": args.embedding_provider,
            "embed_server_url": args.embed_server_url,
            "reranker": args.reranker,
            "dimensions": args.dimensions,
            "k": args.k,
            "limit": args.limit,
            "budget_tokens": args.budget_tokens,
        },
        "errors": [],
        "skipped": [],
        "suites": [],
    }
    if args.suite in {"locomo", "all"}:
        output["suites"].append(await _run_locomo(args))
    if args.suite in {"relations", "all"}:
        output["suites"].append(await _run_relations(args))

    for suite in output["suites"]:
        print(f"\n=== {suite['suite']} ({suite['documents']} docs, {suite['queries']} queries) ===")
        for arm in suite["arms"]:
            print(_compact_arm(arm, args.k))
        comparisons = suite.get("comparisons") or [suite.get("comparison")]
        for comparison in (value for value in comparisons if value):
            print(
                f"  {comparison['treatment']} vs {comparison['baseline']}: "
                f"ΔR={comparison['recall_delta']:+.3f}, "
                f"ΔMRR={comparison['mrr_delta']:+.3f}, "
                f"Δp95={comparison['p95_latency_delta_ms']:+.1f}ms, "
                f"changed={comparison['rankings_changed']}/{comparison['queries_compared']}"
            )

    if args.json_out:
        destination = Path(args.json_out).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("locomo", "relations", "all"), default="all")
    parser.add_argument("--locomo-corpus", default="benchmarks/.cache/locomo/corpus.jsonl")
    parser.add_argument("--locomo-qrels", default="benchmarks/.cache/locomo/qrels.json")
    parser.add_argument("--world-dir", help="path to gbrain-evals/eval/data/world-v1 (required for relations/all)")
    parser.add_argument("--embedding-provider", default="hash")
    parser.add_argument("--embed-server-url", default=None)
    parser.add_argument(
        "--reranker",
        default="none",
        help="reranker provider (default: none; avoids re-embedding candidates in embedding-only evaluations)",
    )
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--budget-tokens", type=int, default=2048)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.suite in {"relations", "all"} and not args.world_dir:
        parser.error("--world-dir is required for the relations and all suites")
    if args.json_out:
        args.json_out = str(Path(args.json_out).resolve())
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
