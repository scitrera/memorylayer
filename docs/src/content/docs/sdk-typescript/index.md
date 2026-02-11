---
title: TypeScript SDK
description: MemoryLayer TypeScript SDK - full-featured client library for Node.js and browsers
sidebar:
  order: 1
  label: Overview
---

The MemoryLayer TypeScript SDK (`@scitrera/memorylayer-sdk`) provides a fully typed client for interacting with the MemoryLayer server from TypeScript and JavaScript applications.

## Installation

```bash
npm install @scitrera/memorylayer-sdk
```

## Features

- **Full TypeScript Support** — Complete type definitions included, no `@types` packages needed
- **Memory Operations** — Remember, recall, reflect, forget, decay
- **Relationship Graph** — Link memories with typed relationships across 11 semantic categories
- **Session Management** — Working memory with TTL and commit
- **Batch Operations** — Bulk create, update, delete
- **Graph Traversal** — Multi-hop relationship queries
- **Error Handling** — Typed exception hierarchy

## Quick Start

```typescript
import { MemoryLayerClient, MemoryType } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace",
});

// Store a memory
const memory = await client.remember("User prefers dark mode", {
  type: MemoryType.SEMANTIC,
  importance: 0.8,
  tags: ["preference", "ui"],
});

// Recall memories
const result = await client.recall("What are the user's UI preferences?", {
  limit: 5,
  minRelevance: 0.7,
});

console.log(result.memories);
```

## Configuration

```typescript
const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",       // Default
  apiKey: process.env.MEMORYLAYER_API_KEY, // Optional for local dev
  workspaceId: "my-workspace",
  sessionId: "optional-session-id",        // Auto-include in requests
  timeout: 30000,                          // Request timeout in ms
});
```

## Next Steps

- [Quick Start Guide](/sdk-typescript/quickstart/) — Step-by-step tutorial
- [API Reference](/sdk-typescript/api-reference/) — Complete method reference
