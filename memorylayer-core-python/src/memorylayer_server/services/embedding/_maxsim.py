"""Multi-vector embedding container and MaxSim scoring helper.

These primitives used to live inside the ColPali in-process provider
(``services/embedding/colpali.py``). They were lifted into this module
when the heavy ML providers were retired in favour of the
``memorylayer-embed-server`` HTTP service, so callers that still need
to score multi-vector embeddings client-side (e.g. on the retrieval
hot path) do not require any GPU/torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiVectorEmbedding:
    """Multi-vector embedding (ColPali-style late interaction).

    Instead of a single vector, stores multiple vectors that represent
    different aspects/patches of the content. Used by MaxSim scoring.
    """

    vectors: list[list[float]]

    @property
    def num_vectors(self) -> int:
        return len(self.vectors)

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


def maxsim_score(
    query_vectors: MultiVectorEmbedding,
    doc_vectors: MultiVectorEmbedding,
) -> float:
    """Calculate MaxSim score between query and document multi-vectors.

    Late interaction scoring used by ColPali/ColBERT. For each query
    vector, find max cosine similarity to any document vector, then
    sum those max similarities.
    """
    q_vecs = np.array(query_vectors.vectors)
    d_vecs = np.array(doc_vectors.vectors)

    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    d_norms = np.linalg.norm(d_vecs, axis=1, keepdims=True)

    q_norms = np.where(q_norms == 0, 1, q_norms)
    d_norms = np.where(d_norms == 0, 1, d_norms)

    q_normalized = q_vecs / q_norms
    d_normalized = d_vecs / d_norms

    similarities = np.dot(q_normalized, d_normalized.T)
    max_sims = np.max(similarities, axis=1)
    return float(np.sum(max_sims))
