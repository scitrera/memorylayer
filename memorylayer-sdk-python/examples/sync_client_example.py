"""Example usage of synchronous MemoryLayer client."""

from memorylayer import SyncMemoryLayerClient, sync_client, MemoryType, MemorySubtype


def example_with_context_manager():
    """Example using context manager."""
    with SyncMemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Store a memory
        memory = client.remember(
            content="User prefers concise code comments",
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.PREFERENCE,
            importance=0.8,
            tags=["preferences", "coding-style"],
        )
        print(f"Created memory: {memory.id}")

        # Search memories
        results = client.recall(
            query="what are the user's coding preferences?",
            types=[MemoryType.SEMANTIC],
            limit=5,
            min_relevance=0.7,
        )
        print(f"Found {len(results.memories)} memories")

        # Reflect on memories
        reflection = client.reflect(
            query="summarize user's development preferences",
            max_tokens=300,
        )
        print(f"Reflection: {reflection.reflection}")


def example_with_helper():
    """Example using sync_client() helper."""
    with sync_client(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Create a session
        session = client.create_session(ttl_seconds=3600)
        print(f"Created session: {session.id}")

        # Set context
        client.set_context(
            session.id,
            "current_task",
            {"type": "refactoring", "file": "auth.py"},
        )

        # Get context
        context = client.get_context(session.id, ["current_task"])
        print(f"Context: {context}")


def example_manual_lifecycle():
    """Example with manual lifecycle management."""
    client = SyncMemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    )

    try:
        # Manually connect
        client.connect()

        # Use the client
        memory = client.remember(
            content="Important project decision",
            type=MemoryType.EPISODIC,
            importance=0.9,
        )
        print(f"Created memory: {memory.id}")

    finally:
        # Always close
        client.close()


if __name__ == "__main__":
    print("Example 1: Context manager")
    # example_with_context_manager()

    print("\nExample 2: Helper function")
    # example_with_helper()

    print("\nExample 3: Manual lifecycle")
    # example_manual_lifecycle()

    print("\nExamples are commented out. Uncomment to run with a real server.")
