"""Unit tests for the pure retrieval metric functions."""

from memorylayer_server.eval import metrics


def test_precision_at_k_divides_by_k():
    # 2 relevant in top-5 -> 2/5 regardless of how many are returned
    retrieved = ["a", "b", "c", "d", "e", "f"]
    assert metrics.precision_at_k(retrieved, {"a", "c"}, 5) == 2 / 5


def test_precision_at_k_penalizes_short_results():
    # Only one result, relevant: precision is 1/k, not 1/1
    assert metrics.precision_at_k(["a"], {"a"}, 5) == 1 / 5


def test_recall_at_k_divides_by_relevant_count():
    retrieved = ["a", "b", "c"]
    assert metrics.recall_at_k(retrieved, {"a", "z"}, 3) == 1 / 2


def test_recall_at_k_respects_cutoff():
    retrieved = ["x", "y", "a"]
    # relevant "a" is at rank 3, outside k=2
    assert metrics.recall_at_k(retrieved, {"a"}, 2) == 0.0
    assert metrics.recall_at_k(retrieved, {"a"}, 3) == 1.0


def test_mrr_first_relevant_rank():
    assert metrics.mrr(["x", "a", "y"], {"a"}) == 1 / 2
    assert metrics.mrr(["a", "x"], {"a"}) == 1.0
    assert metrics.mrr(["x", "y"], {"a"}) == 0.0


def test_first_relevant_hit():
    assert metrics.first_relevant_hit(["a", "b"], {"a"}) == 1
    assert metrics.first_relevant_hit(["b", "a"], {"a"}) == 0
    assert metrics.first_relevant_hit([], {"a"}) == 0


def test_expected_top1_hit_is_exact():
    assert metrics.expected_top1_hit(["a", "b"], "a") == 1
    # "a" is relevant-ish but not the exact expected top1
    assert metrics.expected_top1_hit(["b", "a"], "a") == 0


def test_ndcg_perfect_and_imperfect_ordering():
    grades = {"a": 1.0, "b": 1.0}
    # Both relevant in the top-2 -> perfect nDCG
    assert metrics.ndcg_at_k(["a", "b", "c"], grades, 2) == 1.0
    # Relevant items pushed down -> nDCG < 1
    assert metrics.ndcg_at_k(["c", "d", "a"], grades, 3) < 1.0


def test_jaccard_overlap():
    assert metrics.jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert metrics.jaccard([], []) == 1.0
    assert metrics.jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_empty_inputs_are_zero():
    assert metrics.precision_at_k([], {"a"}, 5) == 0.0
    assert metrics.recall_at_k(["a"], set(), 5) == 0.0
    assert metrics.ndcg_at_k(["a"], {}, 5) == 0.0
    assert metrics.mean([]) == 0.0


def test_redundancy_identical_vectors_is_one():
    # Two identical (parallel) vectors -> cosine 1.0
    embs = [[1.0, 0.0], [2.0, 0.0]]
    assert metrics.redundancy(embs) == 1.0


def test_redundancy_orthogonal_vectors_is_zero():
    embs = [[1.0, 0.0], [0.0, 1.0]]
    assert metrics.redundancy(embs) == 0.0


def test_redundancy_needs_two_embeddings():
    assert metrics.redundancy([]) == 0.0
    assert metrics.redundancy([[1.0, 0.0]]) == 0.0


def test_redundancy_mean_over_pairs():
    # Pairs: (a,b)=1.0, (a,c)=0.0, (b,c)=0.0 -> mean 1/3
    embs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert abs(metrics.redundancy(embs) - 1 / 3) < 1e-9


def test_redundancy_clamps_negative_cosine():
    # Opposite vectors -> cosine -1, clamped to 0
    embs = [[1.0, 0.0], [-1.0, 0.0]]
    assert metrics.redundancy(embs) == 0.0


def test_retrieval_cost_single_shot_is_cheap():
    # 1 step, no requery, default max_steps=5 -> 0.2
    assert metrics.retrieval_cost(1) == 0.2


def test_retrieval_cost_requery_surcharge():
    # 2 steps + 1 requery*0.2 = 2.2 / 5
    assert abs(metrics.retrieval_cost(2, requeries=1) - 2.2 / 5) < 1e-9


def test_retrieval_cost_clamps_to_one():
    assert metrics.retrieval_cost(20, requeries=5) == 1.0
