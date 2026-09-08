"""An exported vault must be internally consistent and OKF-conformant.

The end-to-end statement of the link fix: generate a KB with communities AND
god-node entities, export the vault, and assert every ``[[wikilink]]`` in it
resolves to a file the zip actually contains. Before the fix this failed outright --
entity articles are filed under an id carrying a disambiguation suffix that the
links never included, so every entity cross-link pointed at nothing.

Also pins the OKF front matter contract: a god-node article used to be the only
page in the vault carrying no front matter at all.
"""

import io
import uuid
import zipfile
from datetime import UTC, datetime

import pytest
import pytest_asyncio

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.graph_analysis import (
    CentralNode,
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)
from memorylayer_server.models.memory import MemoryType, ReflectResult, RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.knowledgebase.base import KBGenerateOptions
from memorylayer_server.services.knowledgebase.default import DefaultKnowledgebaseService
from memorylayer_server.services.knowledgebase.linkcheck import extract_wikilinks


class _Spy:
    """Stands in for both the reflect service (entity insights) and the LLM."""

    def __init__(self):
        self.calls = 0

    async def reflect(self, workspace_id, input):  # noqa: A002 - mirror real signature
        self.calls += 1
        return ReflectResult(reflection=f"Notable insight #{self.calls}.")

    async def synthesize(self, prompt, max_tokens=None, profile=None):
        self.calls += 1
        return f"Community Topic {self.calls}\n\nSummary of the cluster [m1]."


class _Graph:
    def __init__(self, analysis=None):
        self.analysis = analysis

    async def analyze(self, workspace_id, context_id=None, include_rpg=False):
        return self.analysis


@pytest_asyncio.fixture
async def vault_workspace(storage_backend) -> str:
    ws_id = f"kbvault_{uuid.uuid4().hex[:8]}"
    await storage_backend.create_workspace(
        Workspace(
            id=ws_id,
            tenant_id="default_tenant",
            name="Vault Test Workspace",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    return ws_id


async def _generate_vault(storage_backend, ws_id: str):
    """Build a KB with two communities and two god nodes; return (service, kb, entries)."""
    mem_ids = []
    for i in range(6):
        mem = await storage_backend.create_memory(
            ws_id,
            RememberInput(
                # Distinct, prose-like content so the entity slug is a real slug rather
                # than a uuid fragment -- the shape the link bug actually appeared in.
                content=f"The team decided to adopt approach number {i} for the rollout",
                type=MemoryType.SEMANTIC,
            ),
        )
        mem_ids.append(mem.id)

    # Associations give the god-node articles a Connections section to link from.
    # mem_ids[3] is itself a god node, so at least one connection points at an article
    # that exists -- the case that must render as a resolvable link.
    for target in mem_ids[1:4]:
        await storage_backend.create_association(
            ws_id,
            AssociateInput(
                source_id=mem_ids[0],
                target_id=target,
                relationship="relates_to",
                strength=0.8,
            ),
        )

    communities = [
        Community(id=0, memory_ids=mem_ids[:3], size=3, central_node_ids=mem_ids[:1]),
        Community(id=1, memory_ids=mem_ids[3:], size=3, central_node_ids=mem_ids[3:4]),
    ]
    analysis = GraphAnalysis(
        snapshot=GraphSnapshot(workspace_id=ws_id, node_count=6, edge_count=3),
        communities=communities,
        # Two god nodes -> two entity articles, and the index links to both.
        central_nodes=[
            CentralNode(memory_id=mem_ids[0], degree=3, community_id=0),
            CentralNode(memory_id=mem_ids[3], degree=1, community_id=1),
        ],
        bridges=[],
        stats=GraphStats(node_count=6, edge_count=3, community_count=2),
    )

    spy = _Spy()
    svc = DefaultKnowledgebaseService(
        storage=storage_backend,
        graph_service=_Graph(analysis),
        reflect_service=spy,
        inference_service=None,
        llm_service=spy,
        v=None,
    )
    kb = await svc.generate(ws_id, options=KBGenerateOptions())

    blob = await svc.export_vault(ws_id)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        entries = {name: zf.read(name).decode("utf-8") for name in zf.namelist()}
    return svc, kb, entries


@pytest.mark.asyncio
async def test_every_wikilink_in_the_exported_vault_resolves_to_a_file(
    storage_backend, vault_workspace
):
    """The regression, stated end to end."""
    _svc, _kb, entries = await _generate_vault(storage_backend, vault_workspace)

    known = {name.removesuffix(".md") for name in entries}
    broken = [
        (path, target)
        for path, md in entries.items()
        for target, _line in extract_wikilinks(md)
        if target not in known
    ]

    assert not broken, f"dangling links in exported vault: {broken}"


@pytest.mark.asyncio
async def test_the_vault_actually_contains_entity_links_to_check(
    storage_backend, vault_workspace
):
    """Guards the test above from passing vacuously on a vault with no entity links."""
    _svc, _kb, entries = await _generate_vault(storage_backend, vault_workspace)

    entity_files = [n for n in entries if n.startswith("entities/")]
    assert entity_files, "fixture produced no entity articles"

    all_targets = [t for md in entries.values() for t, _l in extract_wikilinks(md)]
    assert any(t.startswith("entities/") for t in all_targets), "no entity links rendered"


@pytest.mark.asyncio
async def test_entity_link_targets_carry_the_id_suffix_the_filenames_use(
    storage_backend, vault_workspace
):
    """The precise failure: a link built without the article id's suffix."""
    _svc, _kb, entries = await _generate_vault(storage_backend, vault_workspace)

    entity_targets = {
        t
        for md in entries.values()
        for t, _l in extract_wikilinks(md)
        if t.startswith("entities/")
    }
    entity_files = {n.removesuffix(".md") for n in entries if n.startswith("entities/")}

    assert entity_targets, "no entity links rendered"
    assert entity_targets <= entity_files


@pytest.mark.asyncio
async def test_every_article_carries_okf_front_matter(storage_backend, vault_workspace):
    """`type` is OKF's one required field; god-node articles used to have none."""
    _svc, _kb, entries = await _generate_vault(storage_backend, vault_workspace)

    for path, md in entries.items():
        assert md.startswith("---\n"), f"{path} has no front matter"
        if path == "index.md":
            # Per OKF §11 the bundle root declares the format version instead.
            assert 'okf_version: "0.1"' in md
            continue
        head = md.split("---", 2)[1]
        assert "type: " in head, f"{path} front matter has no OKF type"
        assert "title: " in head, f"{path} front matter has no title"
        assert "timestamp: " in head, f"{path} front matter has no timestamp"


@pytest.mark.asyncio
async def test_entity_front_matter_declares_the_entity_type(storage_backend, vault_workspace):
    _svc, _kb, entries = await _generate_vault(storage_backend, vault_workspace)

    entity_pages = [md for name, md in entries.items() if name.startswith("entities/")]
    assert entity_pages
    for md in entity_pages:
        head = md.split("---", 2)[1]
        assert "type: entity" in head
        assert "connection_count: " in head


@pytest.mark.asyncio
async def test_generate_reports_coverage_and_link_integrity(storage_backend, vault_workspace):
    _svc, kb, _entries = await _generate_vault(storage_backend, vault_workspace)

    assert kb.coverage is not None
    # Both communities are selected and every memory belongs to one, so the whole
    # graph is covered and nothing was dropped by a cut.
    assert kb.coverage["coverage_ratio"] == 1.0
    assert kb.coverage["coverage_uncovered"] == 0
    assert kb.coverage["coverage_dropped_small_communities"] == 0
    assert kb.coverage["coverage_dropped_capped_communities"] == 0


@pytest.mark.asyncio
async def test_coverage_reports_the_gap_when_a_cap_drops_communities(
    storage_backend, vault_workspace
):
    """A capped run must not look like a complete one."""
    mem_ids = []
    for i in range(6):
        mem = await storage_backend.create_memory(
            vault_workspace,
            RememberInput(content=f"memory number {i}", type=MemoryType.SEMANTIC),
        )
        mem_ids.append(mem.id)

    communities = [
        Community(id=0, memory_ids=mem_ids[:3], size=3, central_node_ids=mem_ids[:1]),
        Community(id=1, memory_ids=mem_ids[3:], size=3, central_node_ids=mem_ids[3:4]),
    ]
    analysis = GraphAnalysis(
        snapshot=GraphSnapshot(workspace_id=vault_workspace, node_count=6, edge_count=0),
        communities=communities,
        central_nodes=[],
        bridges=[],
        stats=GraphStats(node_count=6, edge_count=0, community_count=2),
    )
    spy = _Spy()
    svc = DefaultKnowledgebaseService(
        storage=storage_backend,
        graph_service=_Graph(analysis),
        reflect_service=spy,
        inference_service=None,
        llm_service=spy,
        v=None,
    )

    # Only the first community gets an article; the second is dropped by the cap.
    kb = await svc.generate(vault_workspace, options=KBGenerateOptions(max_communities=1))

    assert kb.coverage["coverage_dropped_capped_communities"] == 1
    assert kb.coverage["coverage_covered"] == 3
    assert kb.coverage["coverage_uncovered"] == 3
    assert kb.coverage["coverage_ratio"] == 0.5
