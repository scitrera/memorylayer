---
title: Server Overview
description: MemoryLayer server - FastAPI-based memory infrastructure
sidebar:
  order: 1
  label: Overview
---

The MemoryLayer server (`memorylayer-server`) is a FastAPI-based HTTP server that provides the core memory infrastructure. It handles memory storage, vector search, relationship graphs, session management, and embedding generation.

## Features

- **REST API** — Full-featured HTTP API for all memory operations
- **MCP Compatible** — Compatible with the MCP server package (`@scitrera/memorylayer-mcp-server`) for Claude and other LLMs
- **Vector Search** — SQLite with sqlite-vec for efficient similarity search
- **Multiple Embedding Providers** — OpenAI, Google GenAI, sentence-transformers
- **Cognitive Memory Types** — Episodic, semantic, procedural, and working memory
- **Knowledge Graph** — 60+ typed relationship types across 11 categories for memory associations
- **Session Management** — Working memory with TTL and commit to long-term storage

## Installation

```bash
# Basic
pip install memorylayer-server

# With OpenAI embeddings
pip install memorylayer-server[openai]

# With local embeddings
pip install memorylayer-server[local]

# With Google GenAI embeddings
pip install memorylayer-server[google]

# All providers (includes openai, local, google)
pip install memorylayer-server[embeddings]
```

## Quick Start

```bash
# Start the HTTP server
memorylayer serve --port 61001
```

## API Usage

```python
from memorylayer import MemoryLayerClient

client = MemoryLayerClient(base_url="http://localhost:61001")

# Store a memory
memory = await client.remember(
    content="User prefers Python for backend development",
    type="semantic",
    importance=0.8,
    tags=["preferences", "programming"]
)

# Recall memories
results = await client.recall(
    query="What programming languages does the user like?",
    limit=5
)
```

## Architecture

The server is organized into these layers:

```
┌──────────────────────────────────────────────┐
│  API Layer (FastAPI routes)                   │
│  - /v1/memories (recall, reflect, associate)  │
│  - /v1/sessions, /v1/workspaces              │
│  - /v1/context/*, /health                    │
├──────────────────────────────────────────────┤
│  Service Layer                                │
│  - MemoryService, SessionService             │
│  - WorkspaceService, AssociationService      │
│  - ContradictionService, EmbeddingService    │
│  - ContextEnvironmentService, LLMService     │
├──────────────────────────────────────────────┤
│  Storage Layer                                │
│  - SQLite + sqlite-vec                       │
│  - Embedding providers                       │
└──────────────────────────────────────────────┘
```

## Source Code

The server source code is located at [`oss/memorylayer-core-python/`](https://github.com/scitrera/memorylayer/tree/main/oss/memorylayer-core-python) and published to PyPI as `memorylayer-server`.
