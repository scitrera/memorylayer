"""KB coverage: the share of the association graph that reached an article.

A generated KB is a filtered view -- communities below min_community_size are
dropped, the remainder capped at max_communities, central nodes capped at
max_god_nodes -- and before this metric existed a KB covering a third of a
workspace was indistinguishable from one covering all of it.
"""

from memorylayer_server.services.knowledgebase.coverage import CoverageReport, compute_coverage


def test_ratio_and_uncovered_over_a_partial_graph():
    report = compute_coverage(graph_nodes=10, covered_memory_ids={"a", "b", "c", "d"})

    assert report.covered == 4
    assert report.graph_nodes == 10
    assert report.uncovered == 6
    assert report.ratio == 0.4


def test_empty_graph_counts_as_fully_covered():
    """Nothing to cover is not a coverage failure -- it must not report 0%."""
    report = compute_coverage(graph_nodes=0, covered_memory_ids=set())

    assert report.ratio == 1.0
    assert report.uncovered == 0


def test_covered_beyond_the_node_count_clamps_instead_of_going_negative():
    """God nodes and entity members can sit outside the partition the count came from."""
    report = compute_coverage(graph_nodes=2, covered_memory_ids={"a", "b", "c"})

    assert report.uncovered == 0
    assert report.ratio == 1.0


def test_dropped_counts_attribute_the_loss_to_a_specific_cut():
    report = compute_coverage(
        graph_nodes=100,
        covered_memory_ids={str(i) for i in range(30)},
        dropped_small_communities=12,
        dropped_capped_communities=7,
    )

    assert report.as_metadata() == {
        "coverage_ratio": 0.3,
        "coverage_covered": 30,
        "coverage_graph_nodes": 100,
        "coverage_uncovered": 70,
        "coverage_dropped_small_communities": 12,
        "coverage_dropped_capped_communities": 7,
    }


def test_metadata_keys_are_all_coverage_prefixed():
    """get_knowledgebase reconstructs coverage by prefix-filtering index metadata."""
    keys = CoverageReport().as_metadata().keys()

    assert all(k.startswith("coverage_") for k in keys)
