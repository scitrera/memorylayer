"""Unit tests for the incremental KB rendering differ (Lever 3).

Covers community-stability matching (Jaccard classify: unchanged/changed/new/dissolved,
deterministic tie-breaking) and content-hash key computation.
"""

from memorylayer_server.models.graph_analysis import Community
from memorylayer_server.services.knowledgebase import differ as kb_differ


def _community(cid: int, members: list[str], label: str | None = None, central: list[str] | None = None) -> Community:
    return Community(
        id=cid,
        memory_ids=members,
        size=len(members),
        central_node_ids=central or [],
        label=label,
    )


class TestMatchCommunities:
    def test_no_prior_all_new(self):
        current = [_community(0, ["a", "b"]), _community(1, ["c", "d"])]
        result = kb_differ.match_communities(current, prior=[], threshold=0.5)
        assert all(m.is_new for m in result.matches)
        assert all(m.prior_id is None for m in result.matches)
        assert result.dissolved_prior_ids == []
        # New communities key on their own current id.
        assert {m.article_id for m in result.matches} == {"community-0", "community-1"}

    def test_unchanged_carries_forward_prior_id(self):
        prior = [_community(7, ["a", "b", "c"], label="Cats")]
        # Same members, but a different (volatile) current Louvain idx.
        current = [_community(0, ["a", "b", "c"])]
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        m = result.matches[0]
        assert not m.is_new
        assert m.prior_id == 7
        assert m.article_id == "community-7"  # stable prior id, not current idx 0
        assert m.prior_label == "Cats"
        assert result.dissolved_prior_ids == []

    def test_changed_still_matches_above_threshold(self):
        # 3/4 overlap -> Jaccard = 3 / (4) ... |A&B|=3, |AuB|=5 -> 0.6 >= 0.5
        prior = [_community(2, ["a", "b", "c", "d"], label="Old")]
        current = [_community(0, ["a", "b", "c", "e"])]
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        m = result.matches[0]
        assert not m.is_new
        assert m.prior_id == 2
        assert m.article_id == "community-2"

    def test_below_threshold_is_new_and_prior_dissolved(self):
        # 1/3 overlap -> Jaccard = 1/5 = 0.2 < 0.5
        prior = [_community(5, ["a", "b", "c"])]
        current = [_community(0, ["a", "x", "y"])]
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        assert result.matches[0].is_new
        assert result.dissolved_prior_ids == [5]

    def test_dissolved_prior_listed(self):
        prior = [_community(0, ["a", "b"]), _community(1, ["c", "d"])]
        current = [_community(0, ["a", "b"])]  # only matches prior 0
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        assert result.dissolved_prior_ids == [1]
        assert result.matches[0].prior_id == 0

    def test_one_to_one_greedy(self):
        # Two current communities; prior 0 overlaps both but should only match the best.
        prior = [_community(0, ["a", "b", "c", "d"]), _community(1, ["e", "f"])]
        current = [
            _community(0, ["a", "b", "c", "d"]),  # exact match prior 0 (J=1.0)
            _community(1, ["a", "e", "f"]),  # overlaps prior0 (1) and prior1 (2)
        ]
        result = kb_differ.match_communities(current, prior, threshold=0.3)
        by_cur = {m.community.id: m for m in result.matches}
        assert by_cur[0].prior_id == 0  # exact match wins
        assert by_cur[1].prior_id == 1  # prior 0 already taken -> falls to prior 1
        assert result.dissolved_prior_ids == []

    def test_deterministic_tie_break_prefers_larger_overlap(self):
        # current matches two priors at equal Jaccard; larger absolute overlap wins.
        # prior A: 2/2 size, prior B sized so Jaccard ties but overlap differs.
        # current = {a,b,c,d}; priorA={a,b,c,d,e,f} overlap=4 union=6 J=0.666
        #            priorB={a,b} overlap=2 union=4 J=0.5
        # Not a tie; construct an actual tie instead:
        # current={a,b,c,d}; priorA={a,b,c,d} J=1.0 overlap4; priorB={a,b,c,d} J=1.0 overlap4
        # tie on J and overlap and size_delta -> lower prior id wins.
        prior = [_community(9, ["a", "b", "c", "d"]), _community(3, ["a", "b", "c", "d"])]
        current = [_community(0, ["a", "b", "c", "d"])]
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        # Lower prior id (3) wins the tie.
        assert result.matches[0].prior_id == 3
        assert result.dissolved_prior_ids == [9]

    def test_empty_current_community_skipped(self):
        prior = [_community(0, ["a"])]
        current = [_community(0, [])]
        result = kb_differ.match_communities(current, prior, threshold=0.5)
        assert result.matches[0].is_new
        assert result.dissolved_prior_ids == [0]


class TestContentKeys:
    def test_community_key_stable_for_same_inputs(self):
        c = _community(0, ["b", "a"], central=["a"])
        versions = {"a": "v1", "b": "v2"}
        k1 = kb_differ.community_content_key(c, versions)
        k2 = kb_differ.community_content_key(c, versions)
        assert k1 == k2

    def test_community_key_order_independent(self):
        c1 = _community(0, ["a", "b"], central=["a", "b"])
        c2 = _community(0, ["b", "a"], central=["b", "a"])
        versions = {"a": "v1", "b": "v2"}
        assert kb_differ.community_content_key(c1, versions) == kb_differ.community_content_key(c2, versions)

    def test_community_key_busts_on_version_change(self):
        c = _community(0, ["a", "b"])
        k1 = kb_differ.community_content_key(c, {"a": "v1", "b": "v2"})
        k2 = kb_differ.community_content_key(c, {"a": "v1", "b": "v3"})
        assert k1 != k2

    def test_community_key_busts_on_centrality_change(self):
        versions = {"a": "v1", "b": "v2"}
        c1 = _community(0, ["a", "b"], central=["a"])
        c2 = _community(0, ["a", "b"], central=["b"])
        assert kb_differ.community_content_key(c1, versions) != kb_differ.community_content_key(c2, versions)

    def test_entity_key_stable_and_connection_order_independent(self):
        conns1 = [
            {"target_id": "t1", "relationship": "rel", "strength": 0.5},
            {"target_id": "t2", "relationship": "rel", "strength": 0.6},
        ]
        conns2 = list(reversed(conns1))
        k1 = kb_differ.entity_content_key("m1", "v1", conns1)
        k2 = kb_differ.entity_content_key("m1", "v1", conns2)
        assert k1 == k2

    def test_entity_key_busts_on_connection_change(self):
        conns = [{"target_id": "t1", "relationship": "rel", "strength": 0.5}]
        k1 = kb_differ.entity_content_key("m1", "v1", conns)
        k2 = kb_differ.entity_content_key("m1", "v1", conns + [{"target_id": "t2", "relationship": "rel", "strength": 0.9}])
        assert k1 != k2

    def test_entity_key_busts_on_version_change(self):
        conns = [{"target_id": "t1", "relationship": "rel", "strength": 0.5}]
        assert kb_differ.entity_content_key("m1", "v1", conns) != kb_differ.entity_content_key("m1", "v2", conns)


class TestParsePriorCommunities:
    def test_none_returns_empty(self):
        assert kb_differ.parse_prior_communities(None) == []

    def test_empty_dict_returns_empty(self):
        assert kb_differ.parse_prior_communities({}) == []

    def test_parses_communities(self):
        analysis_json = {
            "communities": [
                {"id": 0, "memory_ids": ["a", "b"], "size": 2, "central_node_ids": [], "label": "X"},
            ]
        }
        parsed = kb_differ.parse_prior_communities(analysis_json)
        assert len(parsed) == 1
        assert parsed[0].id == 0
        assert parsed[0].label == "X"
