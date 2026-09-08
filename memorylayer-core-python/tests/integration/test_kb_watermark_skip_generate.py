"""Integration tests for P2 strategy-C dirty-watermark KB skip-generate (Gate A).

Exercises DefaultKnowledgebaseService against the real SQLite storage backend with a
controllable fake graph service and spy reflect service. Gate A skips the ENTIRE
generate() run (no analyze, no rendering, no LLM) when the workspace's change-watermark
is unchanged since the last successful generation, and returns the cached Knowledgebase.

Fail-safe is the overriding invariant: the gate skips ONLY when it is certain nothing
changed. Adding a memory, adding an association, OR deleting an association each advances
the watermark and forces a real run -- the association-delete case is the staleness guard
(associations have no updated_at, so the gate relies on association_count to detect it).
"""

import uuid

import pytest
import pytest_asyncio

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.graph_analysis import (
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)
from memorylayer_server.models.memory import MemoryType, ReflectResult, RememberInput
from memorylayer_server.services.knowledgebase.base import KBGenerateOptions
from memorylayer_server.services.knowledgebase.default import DefaultKnowledgebaseService


class SpyReflectService:
    """Summary spy: records each summarization call so tests can assert zero-LLM reuse.

    Community summaries go through the direct LLM handle (``synthesize``); god-node /
    entity articles still use ``reflect``. Both increment ``calls`` so the existing
    "one (re)summary call per community" assertions hold. The instance is wired as
    BOTH ``reflect_service`` and ``llm_service`` in ``_service``.
    """

    def __init__(self):
        self.calls = 0

    async def reflect(self, workspace_id, input):  # noqa: A002 - mirror real signature
        self.calls += 1
        return ReflectResult(reflection=f"reflection #{self.calls}")

    async def synthesize(self, prompt, max_tokens=None, profile=None, **_generation_metadata):
        # The KB community summary path calls this directly; return the structured
        # "title\n\nsummary" the parser expects. Distinct body per call so reused
        # (cache-hit) articles are provably NOT re-summarized.
        self.calls += 1
        return f"Community Topic\n\nSummary #{self.calls}"


class FakeGraphService:
    """Graph service returning a caller-supplied GraphAnalysis, counting analyze() calls."""

    def __init__(self):
        self.analysis: GraphAnalysis | None = None
        self.analyze_calls = 0

    async def analyze(self, workspace_id, context_id=None, include_rpg=False) -> GraphAnalysis:
        self.analyze_calls += 1
        return self.analysis


def _analysis(workspace_id: str, communities: list[Community]) -> GraphAnalysis:
    return GraphAnalysis(
        snapshot=GraphSnapshot(workspace_id=workspace_id, node_count=10, edge_count=5),
        communities=communities,
        central_nodes=[],
        bridges=[],
        stats=GraphStats(node_count=10, edge_count=5, community_count=len(communities)),
    )


@pytest_asyncio.fixture
async def kb_workspace(storage_backend) -> str:
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace

    ws_id = f"kbwm_{uuid.uuid4().hex[:8]}"
    await storage_backend.create_workspace(
        Workspace(
            id=ws_id,
            tenant_id="default_tenant",
            name="KB Watermark Test Workspace",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    return ws_id


async def _make_memories(storage_backend, ws_id: str, count: int) -> list[str]:
    ids = []
    for i in range(count):
        mem = await storage_backend.create_memory(
            ws_id,
            RememberInput(content=f"memory content {i}", type=MemoryType.SEMANTIC),
        )
        ids.append(mem.id)
    return ids


def _service(storage_backend, graph_service, reflect_service) -> DefaultKnowledgebaseService:
    return DefaultKnowledgebaseService(
        storage=storage_backend,
        graph_service=graph_service,
        reflect_service=reflect_service,
        inference_service=None,
        # Same spy drives community summaries (llm.synthesize) and god-node
        # articles (reflect) so a single `.calls` counter tracks both.
        llm_service=reflect_service,
        v=None,
    )


@pytest.mark.asyncio
async def test_unchanged_workspace_skips_entire_generate(storage_backend, kb_workspace):
    """Two generate() runs on an unchanged workspace: the 2nd makes ZERO analyze AND
    ZERO LLM calls and returns the same KB (the whole-run skip)."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 3)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=3, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])

    # Run 1 — full generation: one analyze, one reflect for the single community.
    kb1 = await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1
    assert spy.calls == 1

    # Verify the start-of-run watermark was recorded in the index article.
    index = await storage_backend.get_kb_article(kb_workspace, "index")
    assert "change_watermark" in index["metadata"]

    # Run 2 — workspace untouched -> watermark match -> whole run skipped.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids, size=3, central_node_ids=mem_ids[:1])],
    )
    kb2 = await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1, "unchanged workspace must NOT re-run analyze"
    assert spy.calls == 1, "unchanged workspace must NOT make any LLM call"

    # The cached KB is returned (same stats/article_count as run 1).
    assert kb2.workspace_id == kb1.workspace_id
    assert kb2.article_count == kb1.article_count


@pytest.mark.asyncio
async def test_adding_memory_advances_watermark_and_runs(storage_backend, kb_workspace):
    """Adding a memory changes memory_count -> watermark advances -> next run executes."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    # Add a memory -> memory_count + max(updated_at) change.
    new_ids = await _make_memories(storage_backend, kb_workspace, 1)
    all_ids = mem_ids + new_ids
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=all_ids, size=3, central_node_ids=all_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 2, "adding a memory must force a real run"


@pytest.mark.asyncio
async def test_adding_association_advances_watermark_and_runs(storage_backend, kb_workspace):
    """Adding an association changes association_count -> watermark advances -> run."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    await storage_backend.create_association(
        kb_workspace,
        AssociateInput(source_id=mem_ids[0], target_id=mem_ids[1], relationship="related_to", strength=0.7),
    )
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 2, "adding an association must force a real run"


@pytest.mark.asyncio
async def test_deleting_association_advances_watermark_and_runs(storage_backend, kb_workspace):
    """STALENESS GUARD: deleting an association advances NO timestamp (associations have
    only created_at), so the gate MUST rely on association_count dropping to detect it.
    This proves an edge removal is never silently skipped."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    assoc = await storage_backend.create_association(
        kb_workspace,
        AssociateInput(source_id=mem_ids[0], target_id=mem_ids[1], relationship="related_to", strength=0.7),
    )
    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])

    # Run 1 with the association present.
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    # Run 2 with NO change -> skipped (baseline that the gate does skip when truly idle).
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1, "truly-idle run must be skipped"

    # Delete the association -> association_count drops 1 -> 0 -> watermark advances.
    deleted = await storage_backend.delete_association(kb_workspace, assoc.id)
    assert deleted
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 2, "deleting an association must force a real run (staleness guard)"


@pytest.mark.asyncio
async def test_regenerate_true_always_runs(storage_backend, kb_workspace):
    """regenerate=True bypasses the watermark gate entirely, even when idle."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions(regenerate=True))
    assert graph.analyze_calls == 2, "regenerate=True must always run analyze"


@pytest.mark.asyncio
async def test_no_prior_kb_runs(storage_backend, kb_workspace):
    """First-ever generate (no prior watermark recorded) always runs -- fail-safe."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])

    # No prior index article -> _get_stored_watermark returns None -> cannot skip.
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1


@pytest.mark.asyncio
async def test_watermark_unavailable_runs(storage_backend, kb_workspace, monkeypatch):
    """If the watermark cannot be computed (returns None), the run proceeds -- fail-safe."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    # Force the watermark to be unavailable on the next run.
    async def _no_watermark(workspace_id):
        return None

    monkeypatch.setattr(storage_backend, "get_workspace_change_watermark", _no_watermark)
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 2, "watermark-unavailable must fall back to a real run"


@pytest.mark.asyncio
async def test_partial_store_failure_does_not_record_watermark(storage_backend, kb_workspace, monkeypatch):
    """MAJOR regression guard: if a community/entity article store fails mid-run, the
    watermark must NOT be recorded in the index. A subsequent idle run (workspace
    unchanged) must still call analyze() and regenerate -- never skip and permanently
    serve the incomplete KB.

    Mechanism: gc_safe flips False on any store failure; the index metadata only embeds
    change_watermark when gc_safe=True. The flaky store raises on 'community-0' (a
    community article), so gc_safe=False before the index is written -> the index is
    written WITHOUT change_watermark -> _get_stored_watermark returns None next run ->
    Gate A falls through to a real generate (fail-safe).
    """
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])

    # Run 1 with a community store failure.
    real_store = storage_backend.store_kb_article

    async def flaky_community_store(workspace_id, article_id, **kwargs):
        if article_id == "community-0":
            raise RuntimeError("simulated community store failure")
        return await real_store(workspace_id, article_id, **kwargs)

    monkeypatch.setattr(storage_backend, "store_kb_article", flaky_community_store)
    # generate() fails loudly on the community store failure; the index (sans watermark)
    # is written and GC skipped BEFORE that raise, so the guards below still hold.
    with pytest.raises(RuntimeError, match="failed to persist"):
        await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 1

    # Confirm the index was written (the run reached that step) but WITHOUT the watermark.
    index = await storage_backend.get_kb_article(kb_workspace, "index")
    assert index is not None, "index article must still be written on partial failure"
    assert "change_watermark" not in index["metadata"], (
        "watermark must NOT be recorded when a community/entity store failed "
        "(gc_safe=False guards against permanently skipping an incomplete KB)"
    )

    # Restore the real store so run 2 succeeds.
    monkeypatch.setattr(storage_backend, "store_kb_article", real_store)

    # Run 2 -- workspace is unchanged, but because run 1 recorded no watermark, Gate A
    # must NOT skip: _get_stored_watermark returns None -> fail-safe full regenerate.
    graph.analysis = _analysis(
        kb_workspace, [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])]
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert graph.analyze_calls == 2, (
        "partial-failure run must NOT record a watermark; the next idle run must "
        "still regenerate (fail-safe) rather than skip on an incomplete KB"
    )
