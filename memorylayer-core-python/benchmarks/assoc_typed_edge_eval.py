"""Do typed association edges earn their keep in retrieval?

This eval isolates the retrieval value of *typing* auto-association edges
(``causes``/``supports``/``contradicts``/… vs. a flat ``similar_to``). It answers
the question behind the LLM-cost work: if precise edge types don't move recall,
the cheapest correct policy is deterministic ``similar_to`` + near-duplicate
short-circuit, and the LLM/cross-encoder classifier can stay dark.

Why this is measurable at all: two recall mechanisms are type-SENSITIVE —
  * the backlink (hub-salience) boost EXCLUDES ``similar_to`` edges as an
    anti-signal (``MemoryService.apply_backlink_boost``), so only typed edges
    raise a memory's in-degree salience; and
  * the association-consensus boost / graph-recall arm traverse the edge graph.
So flipping every edge from a real type to ``similar_to`` (or removing edges
entirely) changes what those arms contribute — which is exactly the delta we want.

Configs compared (same corpus, same queries, same embeddings):
  off          — no association edges at all (pure vector/keyword baseline)
  similar_to   — edges built but all labeled ``similar_to`` (LLM classify OFF;
                 this is the deterministic tier from the cost work)
  typed        — edges labeled with a deterministic, network-free TYPED stub so
                 the retrieval arms see type diversity

The ``typed`` config uses a STUB classifier (no LLM, no NLI service) so the eval
runs fully offline on the ``hash`` embedding provider. It measures the retrieval
side's *sensitivity to typing*, not classifier quality: if ``typed`` ≈
``similar_to`` here, better typing cannot help downstream either, so typed edges
don't earn their keep. To instead measure a REAL classifier, bootstrap against
the enterprise stack and set ``MEMORYLAYER_ONTOLOGY_SERVICE`` +
``MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED=1`` (or point the cross-encoder at
an NLI endpoint) rather than using the stub.

Run:
    python -m benchmarks.assoc_typed_edge_eval            # fixture corpus, k=5
    python benchmarks/assoc_typed_edge_eval.py --k 10
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scitrera_app_framework import get_extension

from memorylayer_server.eval import harness, metrics
from memorylayer_server.eval.models import EvalReport, Qrel, QueryResult, load_corpus, load_qrels
from memorylayer_server.models.memory import RecallInput, RecallMode

# Deterministic typed-edge stub. Chosen so the label depends only on the pair's
# content hash — reproducible across runs, no network, and biased toward the
# high-value learning/causal types the classifier work cares about.
_STUB_TYPES = ["supports", "contradicts", "causes", "refines", "supersedes", "related_to"]


def _install_typed_stub(v) -> None:
    """Replace the ontology's batched classifier with a deterministic typed stub.

    Assigns a type per (anchor, candidate) pair from a content hash so the edge
    graph carries realistic type diversity without an LLM/NLI call. Also forces
    the LLM-classify master switch ON so ``auto_associate`` takes the classify
    path (the stub stands in for the model).
    """
    from memorylayer_server.config import MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED
    from memorylayer_server.services.association import EXT_ASSOCIATION_SERVICE

    assoc = get_extension(EXT_ASSOCIATION_SERVICE, v)
    assoc.llm_classify_enabled = True  # already read in __init__; override live
    onto = assoc.ontology_service
    # Give the stub a truthy llm_service so auto_associate's use_llm gate passes.
    if getattr(onto, "llm_service", None) is None:
        onto.llm_service = object()

    async def _stub_batch(content_a, candidates, tenant_id="_default", workspace_id=None):
        out: dict[str, str] = {}
        for cand_id, cand_content in candidates:
            h = hash((content_a, cand_content)) % len(_STUB_TYPES)
            out[cand_id] = _STUB_TYPES[h]
        return out

    onto.classify_relationships_batch = _stub_batch
    # Keep the env flag coherent for any re-read.
    v.set(MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED, True)


def _set_association_recall_flags(v) -> None:
    """Turn on the type-sensitive retrieval arms so edge types can matter."""
    from memorylayer_server.services.memory.base import (
        MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
        MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
        MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
    )

    # Backlink boost is the cleanest type-sensitive lever (excludes similar_to).
    v.set(MEMORYLAYER_BACKLINK_BOOST_WEIGHT, 0.3)
    v.set(MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT, 0.2)
    v.set(MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS, True)


async def _retrieve_keys_with_assoc(v, query: str, k: int) -> list[str]:
    """Recall WITH association expansion enabled (harness default disables it)."""
    from memorylayer_server.services.memory import EXT_MEMORY_SERVICE

    memory_service = get_extension(EXT_MEMORY_SERVICE, v)
    result = await memory_service.recall(
        harness._EVAL_WORKSPACE_ID,
        RecallInput(
            query=query,
            mode=RecallMode.RAG,
            limit=max(k, 20),
            min_relevance=0.0,
            include_associations=True,
            traverse_depth=1,
        ),
    )
    keys: list[str] = []
    for memory in result.memories:
        key = (memory.metadata or {}).get(harness._EVAL_KEY)
        if key is not None:
            keys.append(key)
    return keys


async def _run_eval_assoc(v, qrels: list[Qrel], k: int) -> EvalReport:
    results: list[QueryResult] = []
    for qrel in qrels:
        relevant = set(qrel.relevant)
        grades = qrel.grade_map()
        retrieved = await _retrieve_keys_with_assoc(v, qrel.query, k)
        qr = QueryResult(
            query_id=qrel.query_id or qrel.query,
            query=qrel.query,
            retrieved=retrieved,
            precision_at_k=metrics.precision_at_k(retrieved, relevant, k),
            recall_at_k=metrics.recall_at_k(retrieved, relevant, k),
            mrr=metrics.mrr(retrieved, relevant),
            ndcg_at_k=metrics.ndcg_at_k(retrieved, grades, k),
            first_relevant_hit=metrics.first_relevant_hit(retrieved, relevant),
            latency_ms=0.0,
        )
        results.append(qr)
    return harness._aggregate(results, k)


async def _run_config(name: str, corpus, qrels, k: int) -> EvalReport:
    """Bootstrap a fresh stack, ingest under one association config, score."""
    from memorylayer_server.config import MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED
    from memorylayer_server.dependencies import shutdown_services

    v = await harness.bootstrap(embedding_provider="hash")
    try:
        _set_association_recall_flags(v)

        if name == "off":
            await harness.ingest_corpus(v, corpus, clean=True, build_associations=False)
        elif name == "similar_to":
            v.set(MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED, False)
            await harness.ingest_corpus(v, corpus, clean=True, build_associations=True)
        elif name == "typed":
            _install_typed_stub(v)
            await harness.ingest_corpus(v, corpus, clean=True, build_associations=True)
        else:
            raise ValueError(f"unknown config: {name}")

        return await _run_eval_assoc(v, qrels, k)
    finally:
        await shutdown_services(v)


def _fmt(r: EvalReport) -> str:
    return (
        f"recall@k={r.mean_recall:.3f}  ndcg={r.mean_ndcg:.3f}  "
        f"mrr={r.mean_mrr:.3f}  p@k={r.mean_precision:.3f}  "
        f"top1={r.expected_top1_hit_rate:.3f}"
    )


async def main_async(args) -> int:
    corpus = load_corpus(args.corpus)
    qrels = load_qrels(args.qrels)
    print(f"corpus={len(corpus)} docs  queries={len(qrels)}  k={args.k}\n")

    reports: dict[str, EvalReport] = {}
    for name in ("off", "similar_to", "typed"):
        reports[name] = await _run_config(name, corpus, qrels, args.k)
        print(f"{name:12s} {_fmt(reports[name])}")

    base = reports["similar_to"].mean_recall
    typed_delta = reports["typed"].mean_recall - base
    off_delta = reports["off"].mean_recall - base
    print("\n--- verdict (recall@k vs similar_to baseline) ---")
    print(f"  off   Δ = {off_delta:+.3f}  (does building edges at all help?)")
    print(f"  typed Δ = {typed_delta:+.3f}  (does TYPING edges beyond similar_to help?)")
    if abs(typed_delta) < 0.005:
        print("  => typed edges do NOT change recall here: keep classifier dark;")
        print("     deterministic similar_to + duplicate short-circuit is sufficient.")
    elif typed_delta > 0:
        print("  => typed edges IMPROVE recall: the classifier earns its keep;")
        print("     prefer the cheap cross-encoder over the LLM at this quality.")
    else:
        print("  => typing edges HURTS recall here (noise): keep similar_to.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(harness.DEFAULT_CORPUS))
    p.add_argument("--qrels", default=str(harness.DEFAULT_QRELS))
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    assert Path(args.corpus).exists(), f"corpus not found: {args.corpus}"
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
