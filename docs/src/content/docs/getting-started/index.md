---
title: Overview
description: Get started with MemoryLayer - memory infrastructure for LLM-powered agents
sidebar:
  order: 1
---

MemoryLayer is an API-first memory infrastructure for LLM-powered agents. It solves the fundamental challenge of stateless LLMs by providing persistent, queryable, multi-modal memory that agents can read from and write to during execution.

## What is MemoryLayer?

MemoryLayer provides cognitive memory capabilities for AI agents, including episodic, semantic, procedural, and working memory with vector-based retrieval and graph-based associations.

Think of it as a **memory backend** for your AI applications — the same way you'd use PostgreSQL for relational data or Redis for caching, you use MemoryLayer for agent memory.

## Key Features

- **Cognitive Memory Architecture** — Episodic, semantic, procedural, and working memory types modeled after human cognition
- **Vector Search** — SQLite with sqlite-vec for efficient similarity search, with support for OpenAI, sentence-transformers, and vLLM embedding providers
- **Knowledge Graph** — Typed relationships across multiple categories (causal, solution, learning, workflow, hierarchical, etc.) connecting memories into a traversable semantic graph
- **Session Management** — Working memory with TTL that can be committed to long-term storage
- **Multi-Platform SDKs** — Python and TypeScript client libraries with full type safety
- **Framework Integrations** — Drop-in support for LangChain, LlamaIndex, and Claude Code (via MCP)
- **REST API** — FastAPI-based HTTP server with OpenAPI documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Frameworks (LangChain, LlamaIndex, Claude Code)      │
├─────────────────────────────────────────────────────────────┤
│  Client SDKs (Python, TypeScript)                           │
├─────────────────────────────────────────────────────────────┤
│  MemoryLayer Server (REST API + MCP)                        │
├─────────────────────────────────────────────────────────────┤
│  Storage (SQLite + sqlite-vec)                              │
└─────────────────────────────────────────────────────────────┘
```

## Packages

| Package | Install | Description |
|---------|---------|-------------|
| **Server** | `pip install memorylayer-server` | FastAPI server with SQLite storage |
| **Python SDK** | `pip install memorylayer-client` | Async Python client library |
| **TypeScript SDK** | `npm install @scitrera/memorylayer-sdk` | TypeScript/JavaScript client |
| **MCP Server** | `npm install @scitrera/memorylayer-mcp-server` | Model Context Protocol integration |
| **LangChain** | `pip install memorylayer-langchain` | LangChain memory backend |
| **LlamaIndex** | `pip install memorylayer-llamaindex` | LlamaIndex chat store |

## Quick Example

```python
from memorylayer import MemoryLayerClient, MemoryType

async with MemoryLayerClient(base_url="http://localhost:61001") as client:
    # Store a memory
    memory = await client.remember(
        content="User prefers Python for backend development",
        type=MemoryType.SEMANTIC,
        importance=0.8,
        tags=["preferences", "programming"]
    )

    # Recall memories
    results = await client.recall(
        query="What programming languages does the user like?",
        limit=5
    )

    # Synthesize insights
    reflection = await client.reflect(
        query="Summarize user's technology preferences"
    )
    print(reflection.reflection)
```

## Next Steps

- [Installation](/getting-started/installation/) — Install the server and your preferred SDK
- [Core Concepts](/getting-started/concepts/) — Understand memory types, workspaces, and associations
- [Python SDK Quick Start](/sdk-python/quickstart/) — Build your first memory-powered application
- [TypeScript SDK Quick Start](/sdk-typescript/quickstart/) — Get started with TypeScript
