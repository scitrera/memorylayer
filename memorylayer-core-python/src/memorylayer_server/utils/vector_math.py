"""Vector math utilities for embedding operations."""
import numpy as np


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute dot product similarity between two unit-normalized vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors using numpy for performance.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if vectors have different lengths
        or either vector is zero.
    """
    if len(vec1) != len(vec2):
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
