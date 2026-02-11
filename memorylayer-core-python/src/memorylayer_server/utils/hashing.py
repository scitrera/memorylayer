"""Content hashing utilities for deduplication."""

from hashlib import sha256


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication.

    This is the single source of truth for content hashing.
    All services should use this function rather than implementing
    their own hashing logic.

    Args:
        content: The text content to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    return sha256(content.encode()).hexdigest()
