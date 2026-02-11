"""ID generation utilities."""

import uuid


def generate_id(prefix: str, length: int = 16) -> str:
    """Generate a unique ID with the given prefix.

    Format: {prefix}_{uuid_hex[:length]}
    Example: mem_a1b2c3d4e5f6g7h8

    Args:
        prefix: The prefix for the ID (e.g., "mem", "assoc", "ctx", "traj")
        length: Number of hex characters from UUID (default: 16)

    Returns:
        Unique ID string
    """
    return f"{prefix}_{uuid.uuid4().hex[:length]}"
