"""Retrieval-eval harness: boot the service stack, ingest a corpus, score recall.

The harness drives the *real* ``MemoryService.recall()`` pipeline in-process so
that any ranking change (hybrid fusion, pooling, boosts, …) is reflected in the
measured metrics. It defaults to the deterministic ``hash`` embedding provider
so runs are stable and lexically meaningful without any network access.

Typical use is via the ``gate`` CLI (``python -m memorylayer_server.eval``); the
functions here are also importable for ad-hoc experiments.
"""

from __future__ import annotations

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from scitrera_app_framework import Variables, get_extension

from ..models.memory import MemoryType, RecallInput, RecallMode, RememberInput
from ..services.memory.entities import extract_entities
from . import metrics
from .models import CorpusDoc, EvalReport, Qrel, QueryResult

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_CORPUS = DATASETS_DIR / "fixture_corpus.jsonl"
DEFAULT_QRELS = DATASETS_DIR / "fixture_qrels.json"

_EVAL_WORKSPACE_ID = "eval"
_EVAL_TENANT_ID = "eval_tenant"
_EVAL_KEY = "eval_key"


def _eval_metadata(doc: CorpusDoc, extraction_service=None) -> dict:
    """Build the per-doc ingest metadata: stable eval key + entities.

    Entity metadata (``speaker`` + ``entities`` + ``entity_types``) feeds the
    entity-anchored retrieval channel and registry accretion; it is computed
    text-only at ingest so the channel needs no re-embed. When an
    ``extraction_service`` is supplied the entities (and their types) come from
    the selected provider (e.g. GLiNER2 typed NER) so the typed eval exercises
    the real hot path; otherwise it falls back to the regex util (untyped).
    Eval-scoped only — the core write path is untouched.
    """
    if extraction_service is not None:
        ents = extraction_service.extract_entities(doc.content)
    else:
        ents = extract_entities(doc.content)
    md = {
        _EVAL_KEY: doc.key,
        "speaker": ents.get("speaker"),
        "entities": ents.get("entities", []),
    }
    entity_types = ents.get("entity_types")
    if entity_types:
        md["entity_types"] = entity_types
    return md


async def bootstrap(
    data_dir: str | None = None,
    embedding_provider: str = "hash",
    hybrid: bool | None = None,
    *,
    storage_backend: str = "sqlite",
    reranker: str = "none",
    dimensions: int | None = None,
    embed_server_url: str | None = None,
    postgres_url: str | None = None,
    memory_service: str | None = None,
) -> Variables:
    """Initialize an isolated MemoryLayer service stack for evaluation.

    Mirrors the test bootstrap in ``tests/conftest.py`` (mock-free, no LLM, no
    background tasks) but defaults to the lexical ``hash`` embedding provider.

    Optional knobs select the realistic-benchmark configuration: a real
    embedding provider (``embed_server`` + ``embed_server_url`` + ``dimensions``),
    a real reranker (``rrf``), and the enterprise Postgres backend
    (``storage_backend="postgresql"`` + ``postgres_url``). ``hybrid`` overrides
    the server default for keyword+vector fusion when set.
    """
    from ..config import (
        MEMORYLAYER_DATA_DIR,
        MEMORYLAYER_EMBEDDING_PROVIDER,
        MEMORYLAYER_RERANKER_PROVIDER,
        MEMORYLAYER_STORAGE_BACKEND,
    )
    from ..dependencies import initialize_services, preconfigure
    from ..services.llm.base import MEMORYLAYER_LLM_REGISTRY
    from ..services.memory.base import MEMORYLAYER_HYBRID_SEARCH_ENABLED
    from ..services.tasks.asyncio_impl import MEMORYLAYER_TASKS_ENABLED

    if data_dir is None:
        data_dir = tempfile.mkdtemp(prefix="memorylayer_eval_")

    v = Variables()
    v.set(MEMORYLAYER_EMBEDDING_PROVIDER, embedding_provider)
    v.set(MEMORYLAYER_STORAGE_BACKEND, storage_backend)
    v.set(MEMORYLAYER_LLM_REGISTRY, "default")
    v.set(MEMORYLAYER_RERANKER_PROVIDER, reranker)
    v.set(MEMORYLAYER_TASKS_ENABLED, "false")
    v.set(MEMORYLAYER_DATA_DIR, data_dir)
    if hybrid is not None:
        v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, hybrid)
    if dimensions is not None:
        v.set("MEMORYLAYER_EMBEDDING_DIMENSIONS", str(dimensions))
    if embedding_provider == "embed_server":
        v.set("MEMORYLAYER_EMBED_TRANSPORT", "http")
        v.set("MEMORYLAYER_EMBED_SERVER_URL", embed_server_url or "http://localhost:61051")
    if postgres_url is not None:
        v.set("MEMORYLAYER_POSTGRESQL_URL", postgres_url)
    if memory_service is not None:
        v.set("MEMORYLAYER_MEMORY_SERVICE", memory_service)

    # The enterprise PostgreSQL backend (and other memorylayer_saas plugins) are
    # registered as a side effect of importing memorylayer_saas.dependencies,
    # which appends an enterprise hook to the OSS preconfigure hook list. The OSS
    # entry-point discovery does not pick them up, so import explicitly here.
    if storage_backend == "postgresql":
        # Pin the optional/external-dependency services to lightweight equivalents
        # so the in-process enterprise stack doesn't require prometheus/redis/aether.
        # Explicit v.set() overrides the enterprise hook's set_default_value choices.
        v.set("MEMORYLAYER_METRICS_SERVICE", "noop")
        v.set("MEMORYLAYER_AUDIT_SERVICE", "noop")
        v.set("MEMORYLAYER_RATE_LIMIT_SERVICE", "noop")
        v.set("MEMORYLAYER_CACHE_SERVICE", "lru")
        v.set("MEMORYLAYER_EMBEDDING_SERVICE", "default")  # single-vector path
        v.set("MEMORYLAYER_AUTHENTICATION_SERVICE", "default")
        v.set("MEMORYLAYER_AUTHORIZATION_SERVICE", "default")
        v.set("MEMORYLAYER_TASK_PROVIDER", "asyncio")  # avoid aether task service (no gateway here)
        # The enterprise SQLAlchemy model sizes its pgvector column from
        # os.environ['MEMORYLAYER_EMBEDDING_DIMENSIONS'] at import time (default
        # 1536). Variables.set() doesn't reach os.environ, so set it here BEFORE
        # importing memorylayer_saas (which imports the model) — otherwise the
        # column is 1536 and 1960-dim inserts fail.
        if dimensions is not None:
            import os

            os.environ["MEMORYLAYER_EMBEDDING_DIMENSIONS"] = str(dimensions)
        try:
            import memorylayer_saas.dependencies  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "storage_backend='postgresql' requires the memorylayer_saas (enterprise) package — run this with the enterprise venv."
            ) from e

    v, _ = preconfigure(v=v, test_mode=True)
    v = await initialize_services(v)

    await _ensure_workspace(v, seed_context=(storage_backend == "postgresql"))
    return v


async def _ensure_workspace(v: Variables, seed_context: bool = False) -> None:
    from ..models.workspace import Context, Workspace
    from ..services.storage import EXT_STORAGE_BACKEND

    storage = get_extension(EXT_STORAGE_BACKEND, v)
    if not await storage.get_workspace(_EVAL_WORKSPACE_ID):
        now = datetime.now(UTC)
        await storage.create_workspace(
            Workspace(
                id=_EVAL_WORKSPACE_ID,
                tenant_id=_EVAL_TENANT_ID,
                name="Retrieval Eval Workspace",
                created_at=now,
                updated_at=now,
            )
        )

    # Seed the _default context only for Postgres: the enterprise PG backend
    # enforces the memories.context_id -> contexts.id FK and does not auto-create
    # it on connect, whereas sqlite tolerates the default context_id without a
    # seeded row (so the sqlite/CI path is left untouched). Skip if present.
    if seed_context and not any(c.id == "_default" for c in await storage.list_contexts(_EVAL_WORKSPACE_ID)):
        await storage.create_context(
            _EVAL_WORKSPACE_ID,
            Context(id="_default", workspace_id=_EVAL_WORKSPACE_ID, name="_default"),
        )


async def ingest_corpus(v: Variables, docs: list[CorpusDoc], clean: bool = False, batch_size: int = 128, progress: bool = False, build_associations: bool = False, build_entity_registry: bool = False, build_cue_anchors: bool = False) -> None:
    """Store every corpus document, stashing its stable ``key`` in metadata.

    ``clean=True`` bypasses the write-path enrichment (auto-association, tiering,
    contradiction) and batch-embeds the corpus, inserting directly via the
    storage backend. This keeps a benchmark corpus side-effect-free (no
    fabricated associations that would skew backlink/graph signals) and fast.

    ``build_entity_registry=True`` opts the clean path back into entity-registry
    accretion (otherwise skipped, since the clean path bypasses ``remember()``'s
    ``_inline_auto_enrich`` where accretion normally fires). Eval-scoped: an eval
    is a controlled batch, so driving accretion inline is correct and deterministic.
    """
    # Drop empty-content docs up front: Memory validation rejects empty content,
    # and skipping here avoids wasting embedding work before the insert fails.
    docs = [doc for doc in docs if doc.content and doc.content.strip()]

    from ..services.extraction import EXT_EXTRACTION_SERVICE

    # Source entity metadata from the selected extraction provider so the typed
    # eval (e.g. GLiNER2) exercises the real hot path and bakes entity_types;
    # falls back to the regex util inside _eval_metadata if unavailable.
    try:
        extraction_service = get_extension(EXT_EXTRACTION_SERVICE, v)
    except Exception:
        extraction_service = None

    if not clean:
        from ..services.memory import EXT_MEMORY_SERVICE

        memory_service = get_extension(EXT_MEMORY_SERVICE, v)
        for doc in docs:
            await memory_service.remember(
                _EVAL_WORKSPACE_ID,
                RememberInput(
                    content=doc.content,
                    type=MemoryType(doc.type),
                    importance=doc.importance,
                    tags=doc.tags,
                    metadata=_eval_metadata(doc, extraction_service),
                ),
                inline=True,
            )
        return

    from ..services.embedding import EXT_EMBEDDING_PROVIDER
    from ..services.storage import EXT_STORAGE_BACKEND

    storage = get_extension(EXT_STORAGE_BACKEND, v)
    provider = get_extension(EXT_EMBEDDING_PROVIDER, v)

    # Optional: build the association graph over the clean corpus (reusing each
    # doc's ingest embedding) so recall(include_associations=True) has edges to
    # expand. Uses the same auto-associate logic as the production write path.
    # The same memory-service handle also drives entity-registry accretion when
    # requested (the clean path skips remember()'s _inline_auto_enrich).
    eval_ms = None
    if build_associations or build_entity_registry:
        from ..services.memory import EXT_MEMORY_SERVICE

        eval_ms = get_extension(EXT_MEMORY_SERVICE, v)

    # Optional: generate + store cue anchors per memory (Memora-style abstraction
    # +cue indexing) so recall's cue channel has an index to search. The clean
    # path bypasses remember()'s _inline_auto_enrich where cue gen normally fires,
    # so drive it inline here. Isolated cost: one LLM call (generate_cue_anchors)
    # + one embed batch per memory — much cheaper than full write-path enrichment.
    # No-op unless the extraction provider implements generate_cue_anchors AND the
    # storage backend implements store_cue_anchors (enterprise PG).
    cue_extraction = extraction_service if build_cue_anchors else None

    contents = [doc.content for doc in docs]
    total = len(contents)
    embeddings: list[list[float]] = []
    for i in range(0, total, batch_size):
        embeddings.extend(await provider.embed_batch(contents[i : i + batch_size]))
        if progress:
            print(f"  [ingest] embedded {min(i + batch_size, total)}/{total}", flush=True)

    for n, (doc, embedding) in enumerate(zip(docs, embeddings, strict=True), start=1):
        memory = await storage.create_memory(
            _EVAL_WORKSPACE_ID,
            RememberInput(
                content=doc.content,
                type=MemoryType(doc.type),
                importance=doc.importance,
                tags=doc.tags,
                metadata=_eval_metadata(doc, extraction_service),
                # Explicit: the enterprise PG backend (unlike sqlite) does not
                # default context_id, and Memory requires a non-null context_id.
                context_id="_default",
            ),
        )
        await storage.update_memory(_EVAL_WORKSPACE_ID, memory.id, embedding=embedding)
        if build_associations and eval_ms is not None:
            # Build similarity edges to already-ingested memories (incremental,
            # mirrors production's backward auto-association at write time).
            await eval_ms._inline_auto_enrich(_EVAL_WORKSPACE_ID, memory, embedding)
        elif build_entity_registry and eval_ms is not None:
            # Accretion-only: drive entity-registry accretion inline (the clean
            # path bypasses remember()'s _inline_auto_enrich where it normally
            # fires). Reads the speaker/entities already on metadata from
            # _eval_metadata; no re-embed needed. Never raises (logs + skips).
            await eval_ms._accrete_entities(_EVAL_WORKSPACE_ID, memory)
        if cue_extraction is not None:
            try:
                cues = await cue_extraction.generate_cue_anchors(doc.content)
                if cues:
                    cue_embeddings = await provider.embed_batch([c["cue"] for c in cues])
                    await storage.store_cue_anchors(
                        _EVAL_WORKSPACE_ID,
                        memory.id,
                        [{"cue": c["cue"], "embedding": e, "entity_id": None} for c, e in zip(cues, cue_embeddings, strict=True)],
                    )
            except Exception as e:
                if progress:
                    print(f"  [cue] gen/store failed for {memory.id}: {e}", flush=True)
        if progress and n % 1000 == 0:
            print(f"  [ingest] stored {n}/{total}", flush=True)


async def _retrieve_keys(v: Variables, query: str, k: int) -> list[str]:
    """Run recall and map returned memories back to their corpus keys, in rank order."""
    from ..services.memory import EXT_MEMORY_SERVICE

    memory_service = get_extension(EXT_MEMORY_SERVICE, v)
    result = await memory_service.recall(
        _EVAL_WORKSPACE_ID,
        RecallInput(
            query=query,
            mode=RecallMode.RAG,
            limit=max(k, 20),
            min_relevance=0.0,
            include_associations=False,
            traverse_depth=0,
        ),
    )
    keys: list[str] = []
    for memory in result.memories:
        key = (memory.metadata or {}).get(_EVAL_KEY)
        if key is not None:
            keys.append(key)
    return keys


async def run_eval(v: Variables, qrels: list[Qrel], k: int) -> EvalReport:
    """Score every qrel against the live recall pipeline and aggregate."""
    results: list[QueryResult] = []

    for qrel in qrels:
        relevant = set(qrel.relevant)
        grades = qrel.grade_map()
        start = time.perf_counter()
        try:
            retrieved = await _retrieve_keys(v, qrel.query, k)
        except Exception as exc:  # fail-closed: surface as a per-query error, never drop silently
            results.append(
                QueryResult(
                    query_id=qrel.query_id or qrel.query,
                    query=qrel.query,
                    retrieved=[],
                    errored=True,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
            continue

        latency_ms = (time.perf_counter() - start) * 1000.0
        qr = QueryResult(
            query_id=qrel.query_id or qrel.query,
            query=qrel.query,
            retrieved=retrieved,
            precision_at_k=metrics.precision_at_k(retrieved, relevant, k),
            recall_at_k=metrics.recall_at_k(retrieved, relevant, k),
            mrr=metrics.mrr(retrieved, relevant),
            ndcg_at_k=metrics.ndcg_at_k(retrieved, grades, k),
            first_relevant_hit=metrics.first_relevant_hit(retrieved, relevant),
            latency_ms=latency_ms,
        )
        if qrel.expected_top1 is not None:
            qr.expected_top1_hit = metrics.expected_top1_hit(retrieved, qrel.expected_top1)
        results.append(qr)

    return _aggregate(results, k)


def _aggregate(results: list[QueryResult], k: int) -> EvalReport:
    ok = [r for r in results if not r.errored]
    with_top1 = [r for r in ok if r.expected_top1_hit is not None]
    return EvalReport(
        k=k,
        queries=results,
        queries_total=len(results),
        queries_run=len(ok),
        queries_errored=len(results) - len(ok),
        mean_precision=metrics.mean([r.precision_at_k for r in ok]),
        mean_recall=metrics.mean([r.recall_at_k for r in ok]),
        mean_mrr=metrics.mean([r.mrr for r in ok]),
        mean_ndcg=metrics.mean([r.ndcg_at_k for r in ok]),
        first_relevant_hit_rate=metrics.mean([float(r.first_relevant_hit) for r in ok]),
        expected_top1_hit_rate=metrics.mean([float(r.expected_top1_hit or 0) for r in with_top1]),
        expected_top1_denominator=len(with_top1),
        mean_latency_ms=metrics.mean([r.latency_ms for r in ok]),
    )


async def evaluate(
    corpus_path: str | Path = DEFAULT_CORPUS,
    qrels_path: str | Path = DEFAULT_QRELS,
    k: int = 5,
    data_dir: str | None = None,
    embedding_provider: str = "hash",
    hybrid: bool | None = None,
    *,
    storage_backend: str = "sqlite",
    reranker: str = "none",
    dimensions: int | None = None,
    embed_server_url: str | None = None,
    postgres_url: str | None = None,
    clean_ingest: bool = False,
) -> EvalReport:
    """End-to-end convenience: bootstrap, ingest, score, shut down."""
    from ..dependencies import shutdown_services
    from .models import load_corpus, load_qrels

    docs = load_corpus(corpus_path)
    qrels = load_qrels(qrels_path)

    v = await bootstrap(
        data_dir=data_dir,
        embedding_provider=embedding_provider,
        hybrid=hybrid,
        storage_backend=storage_backend,
        reranker=reranker,
        dimensions=dimensions,
        embed_server_url=embed_server_url,
        postgres_url=postgres_url,
    )
    try:
        await ingest_corpus(v, docs, clean=clean_ingest)
        return await run_eval(v, qrels, k)
    finally:
        await shutdown_services(v)
