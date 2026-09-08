"""Integration tests for incremental KB rendering (Lever 3) in generate().

Exercises DefaultKnowledgebaseService against the real SQLite storage backend with a
controllable fake graph service and a spy reflect service so we can assert the dominant
win: ZERO LLM (reflect) calls on an unchanged second run, plus stale-article GC behavior.

Default content_version_mode is "content" (sha256 of member content) so all tests here
work correctly at sub-second resolution without workarounds.
"""

import uuid

import pytest
import pytest_asyncio

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
    """Graph service returning a caller-supplied GraphAnalysis (no NetworkX needed)."""

    def __init__(self):
        self.analysis: GraphAnalysis | None = None

    async def analyze(self, workspace_id, context_id=None, include_rpg=False) -> GraphAnalysis:
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
    """Isolated workspace for KB tests."""
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace

    ws_id = f"kbtest_{uuid.uuid4().hex[:8]}"
    await storage_backend.create_workspace(
        Workspace(
            id=ws_id,
            tenant_id="default_tenant",
            name="KB Test Workspace",
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
    """Build a service instance with defaults (content_version_mode="content")."""
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
async def test_unchanged_second_run_makes_zero_llm_calls(storage_backend, kb_workspace):
    """The dominant win: unchanged inputs across two runs -> zero reflect calls on run 2,
    and the community article is reused verbatim (same content)."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 3)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=3, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])

    # Run 1 — full generation, one reflect call for the single community.
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1
    art1 = await storage_backend.get_kb_article(kb_workspace, "community-0")
    assert art1 is not None
    assert "content_key" in art1["metadata"]

    # Advance the change-watermark with a memory OUTSIDE the community so the P2
    # strategy-C skip-generate gate runs this fire and the renderer's per-article
    # content-hash skip (P1.2) is the layer that prevents the LLM call (not the
    # whole-run watermark skip). The community's members -- and thus its content_key
    # -- are unchanged, so the article is reused verbatim with zero new reflect call.
    await _make_memories(storage_backend, kb_workspace, 1)

    # Run 2 — identical analysis & community members -> content_key hit -> NO new reflect call.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids, size=3, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1, "second run must not re-summarize unchanged community"

    art2 = await storage_backend.get_kb_article(kb_workspace, "community-0")
    assert art2["content_md"] == art1["content_md"]


@pytest.mark.asyncio
async def test_volatile_community_id_carries_forward_stable_article_id(storage_backend, kb_workspace):
    """KEY BUG REGRESSION: when Louvain renumbers a community (same members, id 3->9)
    the matched community must inherit the PRIOR stable article id 'community-3', NOT create
    'community-9'. Requires correct get_graph_analysis unwrap (the blocker bug); without the
    fix prior_communities is always [] so matching never triggers.

    Asserts:
    - spy.calls == 1: no re-summary on unchanged content (the LLM-skip win)
    - 'community-3' survives (was produced run 1 and matched in run 2 under the stable id)
    - 'community-9' is never created
    - no live article is deleted
    """
    mem_ids = await _make_memories(storage_backend, kb_workspace, 4)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Run 1: community id=3 with these members.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=3, memory_ids=mem_ids, size=4, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1
    art1 = await storage_backend.get_kb_article(kb_workspace, "community-3")
    assert art1 is not None, "run 1 must store 'community-3'"
    assert "content_key" in art1["metadata"]

    # Advance the change-watermark so the P2 strategy-C skip-generate gate runs this
    # fire (otherwise an idle workspace is skipped and the matcher path never executes).
    # The new memory is NOT part of the community under test, so its membership (and
    # therefore its content_key) is unchanged -- the content-hash reuse is still proven.
    await _make_memories(storage_backend, kb_workspace, 1)

    # Run 2: identical members but Louvain renumbered the community to id=9.
    # Jaccard = 4/4 = 1.0 >= 0.5 -> matched to prior community-3 -> inherits article id
    # 'community-3'. Content unchanged -> content_key hit -> zero new LLM call.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=9, memory_ids=mem_ids, size=4, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1, "renumbered community with same members must reuse prior article (no LLM)"

    # The stable prior article must still exist.
    art2 = await storage_backend.get_kb_article(kb_workspace, "community-3")
    assert art2 is not None, "stable 'community-3' must survive the id rename"
    assert art2["content_md"] == art1["content_md"], "reused article must be verbatim"

    # The volatile new id must NOT have been created as a new article.
    assert await storage_backend.get_kb_article(kb_workspace, "community-9") is None, (
        "'community-9' must never be created when the community was matched to prior 'community-3'"
    )


@pytest.mark.asyncio
async def test_changed_member_busts_cache_and_calls_llm(storage_backend, kb_workspace):
    """Editing a member's content busts the hash -> a fresh reflect call."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1

    # Edit a member's content so its content_version (sha256 of content) changes.
    await storage_backend.update_memory(kb_workspace, mem_ids[0], content="edited content")

    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 2, "changed member must trigger a re-summary"


@pytest.mark.asyncio
async def test_dissolved_community_article_gc_on_success(storage_backend, kb_workspace):
    """A dissolved community's orphan article is deleted by GC on the success path.

    Uses distinct community ids across both runs (5 and 7) to exercise the GC path
    under id volatility: run1 stores community-5 and community-7; run2 only produces
    community-5 (id=5 same members matched, id=7 community dissolved). GC must delete
    the orphaned community-7 article.
    """
    mem_ids = await _make_memories(storage_backend, kb_workspace, 4)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Run 1: two communities with ids 5 and 7.
    graph.analysis = _analysis(
        kb_workspace,
        [
            Community(id=5, memory_ids=mem_ids[:2], size=2),
            Community(id=7, memory_ids=mem_ids[2:], size=2),
        ],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    listed = await storage_backend.list_kb_articles(kb_workspace, limit=100)
    ids = {a["article_id"] for a in listed}
    assert "community-5" in ids and "community-7" in ids

    # Run 2: only the first community survives (second dissolves); id unchanged.
    # In production a community dissolves because its underlying memories changed, so
    # delete the dissolved community's members. This both (a) faithfully models the
    # dissolution and (b) advances the change-watermark so the P2 strategy-C
    # skip-generate gate runs this fire (a truly-idle workspace would be skipped).
    for dissolved_member in mem_ids[2:]:
        await storage_backend.delete_memory(kb_workspace, dissolved_member)
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=5, memory_ids=mem_ids[:2], size=2)],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())

    listed2 = await storage_backend.list_kb_articles(kb_workspace, limit=100)
    ids2 = {a["article_id"] for a in listed2}
    assert "community-5" in ids2
    assert "community-7" not in ids2, "dissolved community-7 orphan must be GC'd"
    assert "index" in ids2


@pytest.mark.asyncio
async def test_partial_failure_skips_gc(storage_backend, kb_workspace, monkeypatch):
    """If a store_kb_article raises mid-run, GC is skipped -> no live article deleted."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 4)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Seed two communities; run 2 will only produce one, making the other an orphan candidate.
    graph.analysis = _analysis(
        kb_workspace,
        [
            Community(id=0, memory_ids=mem_ids[:2], size=2),
            Community(id=1, memory_ids=mem_ids[2:], size=2),
        ],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert (await storage_backend.get_kb_article(kb_workspace, "community-1")) is not None

    # Run 2: only community-0 survives, BUT make the index store fail mid-run.
    # Delete the dissolved community's members so the change-watermark advances and the
    # P2 strategy-C skip-generate gate runs this fire (an idle workspace would skip,
    # and the partial-failure / GC-skip path would never execute).
    for dissolved_member in mem_ids[2:]:
        await storage_backend.delete_memory(kb_workspace, dissolved_member)
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids[:2], size=2)],
    )

    real_store = storage_backend.store_kb_article

    async def flaky_store(workspace_id, article_id, **kwargs):
        if article_id == "index":
            raise RuntimeError("simulated store failure")
        return await real_store(workspace_id, article_id, **kwargs)

    monkeypatch.setattr(storage_backend, "store_kb_article", flaky_store)
    # generate() fails loudly on a persist failure (it must never report success while
    # articles were silently dropped); the GC-skip happens BEFORE that raise.
    with pytest.raises(RuntimeError, match="failed to persist"):
        await svc.generate(kb_workspace, options=KBGenerateOptions())

    # GC must have been skipped -> the now-orphaned community-1 still exists (not deleted).
    assert (await storage_backend.get_kb_article(kb_workspace, "community-1")) is not None, (
        "GC must be skipped on a partial-failure run"
    )


@pytest.mark.asyncio
async def test_regenerate_true_forces_full_resummary(storage_backend, kb_workspace):
    """regenerate=True force-bypasses content-hash reuse and re-summaries with fresh content_keys.

    (Deletion is now deferred to the success path -- see the GC/data-loss tests below --
    but the FORCE semantics here are unchanged: every article is freshly re-summarized.)
    """
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 1

    # regenerate=True ignores the prior content_key -> forces a fresh reflect call.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions(regenerate=True))
    assert spy.calls == 2, "regenerate=True must force a full re-summary"

    art = await storage_backend.get_kb_article(kb_workspace, "community-0")
    assert "content_key" in art["metadata"]


@pytest.mark.asyncio
async def test_regenerate_true_analyze_raises_keeps_prior_articles(storage_backend, kb_workspace):
    """DATA-LOSS GUARD (Fix A): if graph_service.analyze() raises on a regenerate=True run,
    the prior articles must STILL be present (NOT eagerly deleted before analysis)."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 2)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Run 1: establish a populated KB.
    community = Community(id=0, memory_ids=mem_ids, size=2, central_node_ids=mem_ids[:1])
    graph.analysis = _analysis(kb_workspace, [community])
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert (await storage_backend.get_kb_article(kb_workspace, "community-0")) is not None
    assert (await storage_backend.get_kb_article(kb_workspace, "index")) is not None

    # Run 2: regenerate=True but analyze() blows up AFTER the (former) eager-delete point.
    async def boom(workspace_id, context_id=None, include_rpg=False):
        raise RuntimeError("simulated analyze failure")

    graph.analyze = boom
    with pytest.raises(RuntimeError, match="simulated analyze failure"):
        await svc.generate(kb_workspace, options=KBGenerateOptions(regenerate=True))

    # The prior KB must survive -- a stale article is safer than a deleted live one.
    assert (await storage_backend.get_kb_article(kb_workspace, "community-0")) is not None, (
        "analyze failure on regenerate=True must NOT have deleted prior articles"
    )
    assert (await storage_backend.get_kb_article(kb_workspace, "index")) is not None


@pytest.mark.asyncio
async def test_regenerate_true_gcs_orphans_on_success(storage_backend, kb_workspace):
    """regenerate=True happy path: all articles freshly re-summarized AND stale prior
    articles (a dissolved community) GC'd on the success path -- no eager pre-delete."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 4)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Run 1: two communities.
    graph.analysis = _analysis(
        kb_workspace,
        [
            Community(id=5, memory_ids=mem_ids[:2], size=2, central_node_ids=mem_ids[:1]),
            Community(id=7, memory_ids=mem_ids[2:], size=2, central_node_ids=mem_ids[2:3]),
        ],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert spy.calls == 2
    ids1 = {a["article_id"] for a in await storage_backend.list_kb_articles(kb_workspace, limit=100)}
    assert "community-5" in ids1 and "community-7" in ids1

    # Run 2: regenerate=True, only community-5 survives. Force re-summary of the survivor
    # (spy +1) AND GC of the now-orphaned community-7 on the success path.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=5, memory_ids=mem_ids[:2], size=2, central_node_ids=mem_ids[:1])],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions(regenerate=True))
    assert spy.calls == 3, "regenerate=True must force a fresh re-summary of the survivor"

    ids2 = {a["article_id"] for a in await storage_backend.list_kb_articles(kb_workspace, limit=100)}
    assert "community-5" in ids2
    assert "community-7" not in ids2, "orphaned community-7 must be GC'd on the regenerate success path"
    assert "index" in ids2


@pytest.mark.asyncio
async def test_regenerate_true_store_failure_skips_gc(storage_backend, kb_workspace, monkeypatch):
    """regenerate=True with a store failure mid-run -> gc_safe False -> no deletion."""
    mem_ids = await _make_memories(storage_backend, kb_workspace, 4)
    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    # Run 1: two communities.
    graph.analysis = _analysis(
        kb_workspace,
        [
            Community(id=0, memory_ids=mem_ids[:2], size=2, central_node_ids=mem_ids[:1]),
            Community(id=1, memory_ids=mem_ids[2:], size=2, central_node_ids=mem_ids[2:3]),
        ],
    )
    await svc.generate(kb_workspace, options=KBGenerateOptions())
    assert (await storage_backend.get_kb_article(kb_workspace, "community-1")) is not None

    # Run 2: regenerate=True, only community-0 survives, but the index store fails mid-run.
    graph.analysis = _analysis(
        kb_workspace,
        [Community(id=0, memory_ids=mem_ids[:2], size=2, central_node_ids=mem_ids[:1])],
    )
    real_store = storage_backend.store_kb_article

    async def flaky_store(workspace_id, article_id, **kwargs):
        if article_id == "index":
            raise RuntimeError("simulated store failure")
        return await real_store(workspace_id, article_id, **kwargs)

    monkeypatch.setattr(storage_backend, "store_kb_article", flaky_store)
    # generate() fails loudly on a persist failure; the GC-skip happens BEFORE that raise.
    with pytest.raises(RuntimeError, match="failed to persist"):
        await svc.generate(kb_workspace, options=KBGenerateOptions(regenerate=True))

    # gc_safe=False -> the now-orphaned community-1 must NOT have been deleted.
    assert (await storage_backend.get_kb_article(kb_workspace, "community-1")) is not None, (
        "regenerate=True must skip GC on a partial-failure run (no live article deleted)"
    )


@pytest.mark.asyncio
async def test_entity_articles_with_colliding_slugs_get_distinct_ids(storage_backend, kb_workspace):
    """SLUG-COLLISION GUARD (Fix B): two distinct entities whose content slugifies
    identically must get DISTINCT article ids; neither clobbers the other, both discoverable."""
    from memorylayer_server.models.graph_analysis import CentralNode

    # Two memories whose first-60-char content slugifies to the same value ("foo-bar").
    m1 = await storage_backend.create_memory(
        kb_workspace, RememberInput(content="Foo Bar", type=MemoryType.SEMANTIC)
    )
    m2 = await storage_backend.create_memory(
        kb_workspace, RememberInput(content="Foo-Bar", type=MemoryType.SEMANTIC)
    )
    assert m1.id != m2.id

    graph = FakeGraphService()
    spy = SpyReflectService()
    svc = _service(storage_backend, graph, spy)

    analysis = _analysis(kb_workspace, [])
    analysis.central_nodes = [
        CentralNode(memory_id=m1.id, community_id=0, degree=1),
        CentralNode(memory_id=m2.id, community_id=0, degree=1),
    ]
    graph.analysis = analysis

    await svc.generate(kb_workspace, options=KBGenerateOptions())

    listed = await storage_backend.list_kb_articles(kb_workspace, limit=100)
    entity_articles = [a for a in listed if a.get("article_type") == "entity"]
    entity_ids = {a["article_id"] for a in entity_articles}
    assert len(entity_ids) == 2, f"colliding slugs must yield 2 distinct ids, got {entity_ids}"

    # Both entities' memory_ids must remain discoverable (no overwrite).
    memory_ids = {a["metadata"]["memory_id"] for a in entity_articles}
    assert memory_ids == {m1.id, m2.id}, "both entity memory_ids must be discoverable"
    # The pure slug stays human-readable in metadata.
    assert all(a["metadata"]["slug"] == "foo-bar" for a in entity_articles)
    # The id is the slug plus a memory_id disambiguator.
    assert all(a["article_id"].startswith("entity-foo-bar-") for a in entity_articles)
