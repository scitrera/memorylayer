"""Unit tests for the pure MMR (Maximal Marginal Relevance) selection helper."""

from memorylayer_server.services.reranker.mmr import mmr_select


def test_empty_pool_returns_empty():
    assert mmr_select([], []) == []


def test_lambda_one_is_pure_relevance_order():
    # lambda=1.0 -> diversity term ignored, so a plain relevance sort.
    relevance = [0.2, 0.9, 0.5]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    order = mmr_select(relevance, embeddings, lambda_param=1.0)
    assert order == [1, 2, 0]


def test_diversity_demotes_near_duplicate():
    # Items 0 and 1 are identical embeddings (near-duplicates); item 2 is
    # distinct but slightly less relevant. With balanced lambda, the distinct
    # item should be selected 2nd instead of the duplicate.
    relevance = [0.90, 0.85, 0.70]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    order = mmr_select(relevance, embeddings, lambda_param=0.5)
    assert order[0] == 0  # most relevant seeds
    assert order[1] == 2  # distinct item beats the duplicate
    assert order[2] == 1


def test_k_truncates_selection():
    relevance = [0.9, 0.8, 0.7, 0.6]
    embeddings = [[1.0, 0.0]] * 4
    order = mmr_select(relevance, embeddings, lambda_param=0.5, k=2)
    assert len(order) == 2
    assert order[0] == 0


def test_none_embeddings_fall_back_to_relevance():
    # No embeddings -> diversity term is 0 for all, so relevance order holds.
    relevance = [0.3, 0.7, 0.5]
    embeddings = [None, None, None]
    order = mmr_select(relevance, embeddings, lambda_param=0.5)
    assert order == [1, 2, 0]


def test_k_larger_than_pool_selects_all():
    relevance = [0.5, 0.9]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    order = mmr_select(relevance, embeddings, k=10)
    assert sorted(order) == [0, 1]
    assert order[0] == 1  # most relevant first
