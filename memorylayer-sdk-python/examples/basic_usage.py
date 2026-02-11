"""Basic usage examples for MemoryLayer.ai Python SDK."""

import asyncio

from memorylayer import (
    MemoryLayerClient,
    MemorySubtype,
    MemoryType,
    RecallMode,
    RelationshipType,
    SearchTolerance,
)


async def basic_example():
    """Basic memory operations example."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Store a memory
        memory = await client.remember(
            content="User prefers Python for backend development",
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.PREFERENCE,
            importance=0.8,
            tags=["preferences", "programming"],
        )
        print(f"Created memory: {memory.id}")

        # Search memories
        results = await client.recall(
            query="what programming language does the user prefer?",
            types=[MemoryType.SEMANTIC],
            limit=5,
            min_relevance=0.7,
        )
        print(f"\nFound {len(results.memories)} memories:")
        for mem in results.memories:
            print(f"  - {mem.content}")

        # Synthesize memories
        reflection = await client.reflect(
            query="summarize user's technology preferences", max_tokens=300
        )
        print(f"\nReflection:\n{reflection.reflection}")


async def relationship_example():
    """Memory relationships example."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Store a problem
        problem = await client.remember(
            content="API rate limiting causing 429 errors",
            type=MemoryType.EPISODIC,
            subtype=MemorySubtype.PROBLEM,
            importance=0.7,
        )

        # Store a solution
        solution = await client.remember(
            content="Implemented exponential backoff with retry logic",
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.SOLUTION,
            importance=0.9,
        )

        # Link them
        association = await client.associate(
            source_id=solution.id,
            target_id=problem.id,
            relationship=RelationshipType.SOLVES,
            strength=0.95,
        )
        print(f"Created association: {association.relationship}")

        # Get all associations for a memory
        associations = await client.get_associations(problem.id)
        print(f"\nProblem has {len(associations)} associations:")
        for assoc in associations:
            print(f"  - {assoc.relationship}: {assoc.target_id}")


async def session_example():
    """Working memory session example."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Create session
        session = await client.create_session(ttl_seconds=3600)
        print(f"Created session: {session.id}")

        # Store temporary context
        await client.set_context(
            session.id,
            "current_task",
            {"description": "Debugging auth", "file": "auth.py", "line": 42},
        )

        await client.set_context(
            session.id, "user_intent", "Fix token refresh issue"
        )

        # Retrieve context
        context = await client.get_context(
            session.id, ["current_task", "user_intent"]
        )
        print(f"\nSession context:")
        print(f"  Task: {context.get('current_task')}")
        print(f"  Intent: {context.get('user_intent')}")


async def advanced_search_example():
    """Advanced search with LLM mode."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Fast RAG search
        rag_results = await client.recall(
            query="error handling patterns",
            mode=RecallMode.RAG,
            tolerance=SearchTolerance.MODERATE,
            limit=10,
        )
        print(f"RAG search found {len(rag_results.memories)} memories")

        # Deep semantic search with LLM
        llm_results = await client.recall(
            query="what did they say about error handling?",
            mode=RecallMode.LLM,
            tolerance=SearchTolerance.LOOSE,
            limit=10,
        )
        print(f"LLM search found {len(llm_results.memories)} memories")

        # Hybrid search (RAG first, LLM if insufficient)
        hybrid_results = await client.recall(
            query="debugging best practices",
            mode=RecallMode.HYBRID,
            limit=10,
        )
        print(f"Hybrid search found {len(hybrid_results.memories)} memories")


async def memory_lifecycle_example():
    """Memory lifecycle operations."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Create memory
        memory = await client.remember(
            content="Temporary note about testing",
            type=MemoryType.WORKING,
            importance=0.3,
        )
        print(f"Created: {memory.id}")

        # Update memory
        updated = await client.update_memory(
            memory.id, importance=0.5, tags=["testing", "temporary"]
        )
        print(f"Updated importance: {updated.importance}")

        # Get specific memory
        retrieved = await client.get_memory(memory.id)
        print(f"Retrieved: {retrieved.content}")

        # Soft delete
        await client.forget(memory.id, hard=False)
        print("Memory soft-deleted")


async def briefing_example():
    """Get session briefing."""
    async with MemoryLayerClient(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ) as client:
        # Get briefing for last 24 hours
        briefing = await client.get_briefing(lookback_hours=24)

        print("Workspace Summary:")
        print(f"  Total memories: {briefing.workspace_summary.get('total_memories')}")
        print(f"  Recent memories: {briefing.workspace_summary.get('recent_memories')}")

        print(f"\nRecent Activity ({len(briefing.recent_activity)} events):")
        for activity in briefing.recent_activity:
            print(f"  - {activity.summary} ({activity.memories_created} memories)")

        print(f"\nOpen Threads ({len(briefing.open_threads)}):")
        for thread in briefing.open_threads:
            print(f"  - {thread.topic}: {thread.status}")

        if briefing.contradictions_detected:
            print(f"\nContradictions ({len(briefing.contradictions_detected)}):")
            for contradiction in briefing.contradictions_detected:
                print(f"  - {contradiction.memory_a} vs {contradiction.memory_b}")


if __name__ == "__main__":
    print("=== Basic Example ===")
    asyncio.run(basic_example())

    print("\n=== Relationship Example ===")
    asyncio.run(relationship_example())

    print("\n=== Session Example ===")
    asyncio.run(session_example())

    print("\n=== Advanced Search Example ===")
    asyncio.run(advanced_search_example())

    print("\n=== Memory Lifecycle Example ===")
    asyncio.run(memory_lifecycle_example())

    print("\n=== Briefing Example ===")
    asyncio.run(briefing_example())
