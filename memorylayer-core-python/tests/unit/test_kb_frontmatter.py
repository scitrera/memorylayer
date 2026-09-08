"""Unit tests for KB article YAML frontmatter + provenance aggregation."""

import types

from memorylayer_server.services.knowledgebase.default import DefaultKnowledgebaseService as KB
from memorylayer_server.services.knowledgebase.renderer import ObsidianRenderer


def _r():
    return ObsidianRenderer()


def test_frontmatter_empty_is_blank():
    assert _r().render_frontmatter({}) == ""
    assert _r().render_frontmatter({"x": None, "y": []}) == ""


def test_frontmatter_scalars_and_lists():
    fm = _r().render_frontmatter({"type": "community", "confidence": 0.83, "tags": ["ai", "ml"]})
    assert fm.startswith("---\n") and fm.rstrip().endswith("---")
    assert "type: community" in fm
    assert "confidence: 0.83" in fm
    assert "tags: [ai, ml]" in fm


def test_frontmatter_quotes_special_strings():
    fm = _r().render_frontmatter({"aliases": ["A: Corp", "plain"], "title": "- leads"})
    assert 'aliases: ["A: Corp", plain]' in fm  # colon forces quote, plain stays bare
    assert 'title: "- leads"' in fm  # leading '-' forces quote


def test_frontmatter_skips_none_and_empty_list():
    fm = _r().render_frontmatter({"keep": 1, "drop_none": None, "drop_empty": []})
    assert "keep: 1" in fm and "drop_none" not in fm and "drop_empty" not in fm


def test_provenance_fields_aggregation():
    members = [
        {"id": "m1", "importance": 0.6, "tags": ["a", "b"], "source_document_id": "doc1"},
        {"id": "m2", "importance": 0.8, "tags": ["b", "c"], "source_document_id": "doc2"},
        {"id": "m3", "importance": None, "tags": [], "source_document_id": "doc1"},
    ]
    pf = KB._provenance_fields(members)
    assert pf["confidence"] == 0.7  # mean of 0.6, 0.8 (None skipped)
    assert pf["tags"] == ["a", "b", "c"]  # sorted union
    assert pf["source_count"] == 2 and sorted(pf["sources"]) == ["doc1", "doc2"]


def test_provenance_fields_omits_absent_signals():
    members = [{"id": "m1", "importance": None, "tags": [], "source_document_id": None}]
    assert KB._provenance_fields(members) == {}


def test_member_dict_shape():
    mem = types.SimpleNamespace(
        id="x", content="c", type=types.SimpleNamespace(value="semantic"),
        importance=0.5, tags=["t"], source_document_id="d",
    )
    assert KB._member_dict(mem) == {
        "id": "x", "content": "c", "type": "semantic",
        "importance": 0.5, "tags": ["t"], "source_document_id": "d",
    }
