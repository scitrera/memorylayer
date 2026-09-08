"""Mermaid diagrams must be safe to embed or not embedded at all.

An invalid diagram is worse than no diagram: Obsidian renders a fenced error block
where the picture should be. These tests pin the two halves of that guarantee --
labels are escaped so hostile community labels cannot break out of the syntax, and
the validation gate rejects anything malformed so the renderer omits the section
instead of shipping a broken fence.
"""

from memorylayer_server.models.graph_analysis import Bridge, Community
from memorylayer_server.services.knowledgebase.mermaid import (
    build_community_flowchart,
    escape_label,
    node_id,
    validate_flowchart,
)
from memorylayer_server.services.knowledgebase.renderer import ObsidianRenderer


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #


def test_escape_label_strips_characters_that_would_break_out_of_the_label():
    # A quote would close the label; brackets/pipes open shape and edge syntax.
    assert '"' not in escape_label('He said "hello"')
    assert escape_label("a[b]c{d}") == "abcd"
    assert "|" not in escape_label("a|b")


def test_escape_label_collapses_newlines_that_would_end_the_statement():
    assert escape_label("line one\nline two") == "line one line two"


def test_escape_label_never_returns_empty():
    """An empty label renders as an anonymous box."""
    assert escape_label("") == "untitled"
    assert escape_label('"""') == "untitled"
    assert escape_label("   ") == "untitled"


def test_escape_label_truncates_to_caption_length():
    out = escape_label("x" * 200)
    assert len(out) <= 48


def test_node_id_is_mermaid_safe_and_never_bare_numeric():
    assert node_id(3, prefix="c") == "c3"
    assert node_id("a-b c", prefix="c") == "ca_b_c"


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #


def test_validate_accepts_a_well_formed_flowchart():
    block = 'flowchart LR\nc0["Alpha"]\nc1["Beta"]\nc0 --> |"relates_to 0.80"| c1'
    assert validate_flowchart(block) == []


def test_validate_rejects_empty_and_headerless_and_nodeless_diagrams():
    assert validate_flowchart("") == ["empty diagram"]
    assert validate_flowchart('c0["Alpha"]')  # no header
    assert validate_flowchart("flowchart LR")  # header only, no nodes


def test_validate_rejects_a_label_that_escaped_its_quotes():
    """The check that makes the escaping load-bearing rather than hopeful."""
    block = 'flowchart LR\nc0["He said "hi""]'
    assert validate_flowchart(block)


# --------------------------------------------------------------------------- #
# Diagram construction
# --------------------------------------------------------------------------- #


def test_no_neighbours_means_no_diagram():
    assert build_community_flowchart(0, "Alpha", []) is None


def test_flowchart_is_fenced_and_valid():
    out = build_community_flowchart(0, "Alpha", [(1, "Beta", "relates_to", 0.8)])

    assert out is not None
    assert out.startswith("```mermaid\n") and out.endswith("\n```")
    inner = out.removeprefix("```mermaid\n").removesuffix("\n```")
    assert validate_flowchart(inner) == []
    assert 'c0["Alpha"]' in inner
    assert 'c1["Beta"]' in inner


def test_a_hostile_label_still_produces_a_valid_diagram():
    """The end of the escaping story: bad input degrades the caption, not the syntax."""
    out = build_community_flowchart(
        0, 'Alpha "quoted" [bracketed]', [(1, "Beta|piped", "rel", 0.5)]
    )

    assert out is not None
    inner = out.removeprefix("```mermaid\n").removesuffix("\n```")
    assert validate_flowchart(inner) == []


def test_neighbour_cap_keeps_the_diagram_readable():
    neighbours = [(i, f"Community {i}", "relates_to", 0.5) for i in range(1, 30)]
    out = build_community_flowchart(0, "Alpha", neighbours, max_neighbours=5)

    inner = out.removeprefix("```mermaid\n").removesuffix("\n```")
    assert validate_flowchart(inner) == []
    # centre + 5 neighbours
    assert inner.count('["') == 6


# --------------------------------------------------------------------------- #
# Renderer integration
# --------------------------------------------------------------------------- #


def test_community_article_embeds_a_valid_diagram_alongside_the_full_bullet_list():
    community = Community(id=0, size=3, cohesion_score=0.5, memory_ids=["a"])
    bridges = [
        Bridge(
            source_community_id=0,
            target_community_id=n,
            memory_id_source="a",
            memory_id_target="z",
            relationship_type="relates_to",
            strength=0.7,
        )
        for n in (1, 2)
    ]

    md = ObsidianRenderer().render_community(
        community=community, summary="s", members=[], bridges=bridges
    )

    assert "```mermaid" in md
    inner = md.split("```mermaid\n")[1].split("\n```")[0]
    assert validate_flowchart(inner) == []
    # The bullets remain the complete record; the diagram is the overview.
    assert md.count("- ") >= 2


def test_community_article_without_bridges_has_no_mermaid_fence():
    community = Community(id=0, size=3, cohesion_score=0.5, memory_ids=["a"])

    md = ObsidianRenderer().render_community(
        community=community, summary="s", members=[], bridges=[]
    )

    assert "```mermaid" not in md


def test_diagram_does_not_introduce_wikilink_syntax():
    """Node syntax uses single brackets; the link validator must not see them as links."""
    from memorylayer_server.services.knowledgebase.linkcheck import extract_wikilinks

    out = build_community_flowchart(0, "Alpha", [(1, "Beta", "relates_to", 0.8)])
    assert extract_wikilinks(out) == []
