---
title: Architecture
description: MemoryLayer system architecture and design
sidebar:
  order: 1
---

## System Overview

MemoryLayer is designed as **infrastructure** — a memory backend that agent frameworks consume, similar to how applications use PostgreSQL for relational data or Redis for caching.

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Frameworks (LangChain, LlamaIndex, Claude Code)      │
├─────────────────────────────────────────────────────────────┤
│  Client SDKs (Python, TypeScript)                           │
├─────────────────────────────────────────────────────────────┤
│  MemoryLayer Server                                         │
│  ┌─────────────┬──────────────┬─────────────────────┐      │
│  │  REST API    │  MCP Server  │  CLI                │      │
│  ├─────────────┴──────────────┴─────────────────────┤      │
│  │  Service Layer                                    │      │
│  │  Memory · Reflect · Session · Workspace            │      │
│  │  Association · Context · Decay · Cache             │      │
│  ├───────────────────────────────────────────────────┤      │
│  │  Storage Layer                                    │      │
│  │  SQLite + sqlite-vec │ Embedding Providers        │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Server Architecture

The server is a Python FastAPI application organized into three layers:

### API Layer

FastAPI routes that handle HTTP requests and MCP tool calls:

- `/v1/memories` — CRUD operations on memories
- `/v1/memories/recall` — Semantic search
- `/v1/memories/reflect` — Memory synthesis
- `/v1/sessions` — Session management
- `/v1/sessions/briefing` — Session briefings
- `/v1/workspaces` — Workspace management
- `/v1/associations` — Relationship management
- `/v1/memories/{id}/traverse` — Graph queries
- `/v1/context/*` — Context environment operations

### Service Layer

Business logic organized into focused services:

| Service | Responsibility |
|---------|---------------|
| `MemoryService` | Store, retrieve, update, delete, recall, and search memories |
| `ReflectService` | LLM-powered synthesis across memories |
| `SessionService` | Working memory and session lifecycle |
| `WorkspaceService` | Tenant isolation and configuration |
| `AssociationService` | Relationship graph management |
| `ContradictionService` | Contradiction detection and resolution |
| `ContextEnvironmentService` | Server-side sandbox for code execution |
| `DeduplicationService` | Memory deduplication |
| `ExtractionService` | Fact/information extraction |
| `DecayService` | Importance decay over time |
| `CacheService` | Query result caching |

### Storage Layer

- **SQLite** — Primary data store for memories, sessions, workspaces
- **sqlite-vec** — Vector extension for similarity search
- **Embedding Providers** — Pluggable providers for generating vector embeddings

## Client SDKs

Both SDKs wrap the REST API with idiomatic language constructs:

### Python SDK (`memorylayer-client`)

- Async/await with `httpx`
- Context manager support
- Pydantic models for type safety
- Exception hierarchy mapping HTTP status codes

### TypeScript SDK (`@scitrera/memorylayer-sdk`)

- Native `fetch` API
- Full TypeScript type definitions
- Promise-based async operations
- Typed error hierarchy

## MCP Server

The MCP server (`@scitrera/memorylayer-mcp-server`) wraps the TypeScript SDK to provide MCP-compatible tools:

```
MCP Client (Claude) ←→ MCP Server ←→ TypeScript SDK ←→ REST API ←→ Server
```

It provides 21 tools across three categories:
- 9 core memory tools (remember, recall, reflect, forget, associate, briefing, statistics, graph query, audit)
- 4 session management tools (session_start, session_end, session_commit, session_status)
- 8 context environment tools

## Three-Layer Memory Hierarchy

Memories are organized in a three-layer hierarchy enabling progressive abstraction:

```
Category Layer (Aggregated Summaries)
    ↑ aggregates
Item Layer (Discrete Memory Units)
    ↑ extracted from
Resource Layer (Raw Source Material)
```

- **Resource Layer** — Raw conversations, documents, events
- **Item Layer** — Individual facts, preferences, decisions (the primary API surface)
- **Category Layer** — Auto-generated summaries that evolve based on content patterns

## Data Flow

### Remember (Write Path)

```
Client → REST API → MemoryService → Generate Embedding → Store in SQLite
```

### Recall (Read Path)

```
Client → REST API → MemoryService → Vector Search (sqlite-vec) → Rank & Filter → Return
```

### Reflect (Synthesis Path)

```
Client → REST API → ReflectService → Recall relevant memories → LLM synthesis → Return
```

## Deployment

### Local Development

Single-process, single-file SQLite database. Zero configuration:

```bash
pip install memorylayer-server
memorylayer serve
```

### Production (Enterprise)

The enterprise edition adds:

- PostgreSQL backend for scalability
- Redis caching layer
- Horizontal scaling
- Advanced analytics and reflection engine

See [memorylayer.ai](https://memorylayer.ai) for enterprise options.
