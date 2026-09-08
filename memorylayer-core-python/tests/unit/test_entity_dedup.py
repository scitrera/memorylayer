"""Unit tests for entity dedup helpers (union-find + representative selection)."""

import types

from memorylayer_server.services.entity_registry.dedup import UnionFind, pick_representative


def test_union_find_transitive_clusters():
    uf = UnionFind()
    for x in ["a", "b", "c", "d", "e"]:
        uf.add(x)
    uf.union("a", "b")
    uf.union("b", "c")  # transitively links a-b-c
    uf.union("d", "e")
    groups = sorted(tuple(g) for g in uf.groups())
    assert groups == [("a", "b", "c"), ("d", "e")]


def test_union_find_singletons_kept():
    uf = UnionFind()
    for x in ["x", "y"]:
        uf.add(x)
    groups = sorted(tuple(g) for g in uf.groups())
    assert groups == [("x",), ("y",)]


def _e(eid, aliases, confidence):
    return types.SimpleNamespace(id=eid, aliases=aliases, confidence=confidence)


def test_representative_prefers_most_members():
    ents = [_e("e1", ["x"], 0.9), _e("e2", [], 1.0), _e("e3", [], 1.0)]
    assert pick_representative(ents, {"e1": 3, "e2": 9, "e3": 1}).id == "e2"


def test_representative_tiebreak_by_aliases_then_confidence_then_id():
    ents = [_e("e1", ["x"], 0.9), _e("e2", ["x", "y", "z"], 0.5), _e("e3", [], 1.0)]
    # members tie (5) between e1,e2 -> e2 wins on alias count
    assert pick_representative(ents, {"e1": 5, "e2": 5, "e3": 2}).id == "e2"
    # all equal -> lowest id
    flat = [_e("eb", [], 1.0), _e("ea", [], 1.0)]
    assert pick_representative(flat, {"eb": 1, "ea": 1}).id == "ea"
