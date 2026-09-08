"""Wikilinks must point at articles that exist.

The regression these lock down: an entity article is FILED under an id carrying a
disambiguation suffix (``entity-{slug}-{id8}``), but cross-links used to be built
from a bare slug of whatever display string was on hand -- a truncated uuid in OSS,
a canonical name in enterprise. The two could never match, so every entity-to-entity
link in the knowledgebase was dangling. Both sites (the index's Central Nodes list
and an entity's Connections list) had the defect.

The fix is construction, not repair: the renderer resolves a target through the
:class:`LinkIndex` of articles the run will actually produce, and emits plain text
when there is nothing to point at. :func:`validate_articles` is the safety net for
what construction cannot see -- chiefly an article reused verbatim from a prior run
whose target has since been GC'd.
"""

from memorylayer_server.models.graph_analysis import Bridge, CentralNode, Community, GraphStats
from memorylayer_server.services.knowledgebase.linkcheck import (
    LinkIndex,
    article_vault_path,
    extract_wikilinks,
    validate_articles,
)
from memorylayer_server.services.knowledgebase.renderer import ObsidianRenderer


def _renderer() -> ObsidianRenderer:
    return ObsidianRenderer()


# --------------------------------------------------------------------------- #
# Path derivation + extraction
# --------------------------------------------------------------------------- #


def test_vault_path_strips_the_entity_prefix_but_keeps_the_id_suffix():
    # The suffix is exactly what the old link construction dropped.
    assert article_vault_path("entity", "entity-acme-corp-8f14e45f") == "entities/acme-corp-8f14e45f"
    assert article_vault_path("community", "community-3") == "communities/community-3"
    assert article_vault_path("index", "index") == "index"


def test_extract_wikilinks_reads_plain_and_piped_targets_with_line_numbers():
    md = "# Title\n\nSee [[entities/a-1234abcd|Acme]] and [[communities/community-2]].\n"
    assert extract_wikilinks(md) == [
        ("entities/a-1234abcd", 3),
        ("communities/community-2", 3),
    ]


def test_link_index_returns_none_for_unregistered_targets():
    index = LinkIndex()
    index.add_entity("m1", "entity-alpha-11111111")
    assert index.entity_path("m1") == "entities/alpha-11111111"
    assert index.entity_path("m-unknown") is None
    assert index.community_path(7) is None


# --------------------------------------------------------------------------- #
# The regression: a link must equal its target's vault path
# --------------------------------------------------------------------------- #


def test_entity_connection_link_matches_the_target_articles_vault_path():
    index = LinkIndex()
    index.add_entity("mem-target", "entity-beta-llc-22222222")

    md = _renderer().render_entity(
        entity_id="mem-source",
        title="Acme Corp",
        entity_card=None,
        connections=[{"target_id": "mem-target", "target_title": "Beta LLC", "strength": 0.9}],
        community=None,
        source_memories=[],
        link_index=index,
    )

    # Exactly the path the target article is exported to -- suffix included.
    assert "[[entities/beta-llc-22222222|Beta LLC]]" in md
    # The old, always-broken form must not appear.
    assert "[[entities/beta-llc]]" not in md
    assert "[[entities/beta-llc|Beta LLC]]" not in md


def test_index_central_node_link_matches_the_entity_articles_vault_path():
    index = LinkIndex()
    index.add_entity("mem-1", "entity-user-prefers-dark-mode-33333333")

    md = _renderer().render_index(
        workspace_name="ws",
        stats=None,
        communities=[],
        god_nodes=[CentralNode(memory_id="mem-1", degree=9, community_id=0)],
        node_titles={"mem-1": "User prefers dark mode"},
        link_index=index,
    )

    assert "[[entities/user-prefers-dark-mode-33333333|User prefers dark mode]]" in md


def test_unresolvable_target_renders_as_plain_text_not_a_dangling_link():
    """A referenced thing with no article (capped run, min-size filter) must not link."""
    md = _renderer().render_entity(
        entity_id="mem-source",
        title="Acme Corp",
        entity_card=None,
        connections=[{"target_id": "mem-uncapped", "target_title": "Beta LLC", "strength": 0.5}],
        community=None,
        source_memories=[],
        link_index=LinkIndex(),  # nothing registered
    )

    assert "Beta LLC" in md          # the relationship is still shown ...
    assert "[[" not in md            # ... just not as a link


def test_community_bridge_links_only_to_generated_communities():
    index = LinkIndex()
    index.add_community(1, "community-1")
    # Community 2 exceeded max_communities and has no article.

    community = Community(id=1, size=4, cohesion_score=0.5, memory_ids=["a", "b"])
    bridges = [
        Bridge(
            source_community_id=1,
            target_community_id=2,
            memory_id_source="a",
            memory_id_target="z",
            relationship_type="relates_to",
            strength=0.7,
        )
    ]

    md = _renderer().render_community(
        community=community,
        summary="s",
        members=[],
        bridges=bridges,
        link_index=index,
    )

    assert "Community 2" in md
    assert "[[communities/community-2]]" not in md


# --------------------------------------------------------------------------- #
# The safety net
# --------------------------------------------------------------------------- #


def test_validate_articles_flags_a_link_whose_target_was_reclaimed():
    """The reuse + GC hazard: a verbatim-reused article outliving its target."""
    articles = {
        "index": "# ws\n",
        # Reused from a prior run; entities/gone-99999999 was GC'd this run.
        "entities/kept-11111111": "See [[entities/gone-99999999|Gone]].\n",
    }
    report = validate_articles(articles, set(articles))

    assert not report.ok
    assert report.total == 1
    assert report.broken_count == 1
    broken = report.broken[0]
    assert broken.source_path == "entities/kept-11111111"
    assert broken.target == "entities/gone-99999999"
    assert broken.line == 1


def test_validate_articles_passes_when_every_target_exists():
    articles = {
        "index": "- [[entities/alpha-11111111|Alpha]]\n",
        "entities/alpha-11111111": "Back to [[index]].\n",
    }
    report = validate_articles(articles, set(articles))

    assert report.ok
    assert report.total == 2
    assert report.as_metadata() == {
        "links_total": 2,
        "links_broken": 0,
        "links_broken_sample": [],
    }


def test_rendered_vault_round_trips_clean_through_validation():
    """Render index + entity against one index, then validate: nothing dangles."""
    renderer = _renderer()
    index = LinkIndex()
    index.add_entity("mem-1", "entity-alpha-11111111")
    index.add_entity("mem-2", "entity-beta-22222222")
    index.add_community(0, "community-0")

    entity_md = renderer.render_entity(
        entity_id="mem-1",
        title="Alpha",
        entity_card=None,
        connections=[{"target_id": "mem-2", "target_title": "Beta", "strength": 0.4}],
        community=Community(id=0, size=2, cohesion_score=0.3),
        source_memories=[],
        link_index=index,
    )
    index_md = renderer.render_index(
        workspace_name="ws",
        stats=GraphStats(node_count=2),
        communities=[Community(id=0, size=2, cohesion_score=0.3)],
        god_nodes=[CentralNode(memory_id="mem-1", degree=3, community_id=0)],
        node_titles={"mem-1": "Alpha"},
        link_index=index,
    )

    produced = {
        "index": index_md,
        "entities/alpha-11111111": entity_md,
        "entities/beta-22222222": "# Beta\n",
        "communities/community-0": "# C0\n",
    }
    report = validate_articles(produced, set(produced))

    assert report.total > 0, "the fixture should actually contain links"
    assert report.ok, f"unresolved: {report.broken}"
