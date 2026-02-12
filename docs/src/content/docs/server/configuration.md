---
title: Configuration
description: Configure the MemoryLayer server
sidebar:
  order: 2
---

## Environment Variables

The MemoryLayer server can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORYLAYER_SERVER_HOST` | Server bind address | `127.0.0.1` |
| `MEMORYLAYER_SERVER_PORT` | Server port | `61001` |
| `MEMORYLAYER_DATA_DIR` | Server data directory | `~/.config/memorylayer-server` |
| `MEMORYLAYER_SQLITE_STORAGE_PATH` | SQLite database path (relative to data dir) | `memorylayer.db` |
| `MEMORYLAYER_EMBEDDING_PROVIDER` | Embedding provider to use | `local` |
| `MEMORYLAYER_EMBEDDING_OPENAI_API_KEY` | OpenAI API key (for OpenAI embeddings) | — |
| `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` | Google API key (for Google GenAI embeddings) | — |

Use the `--verbose` CLI flag to enable debug logging.

## Embedding Providers

MemoryLayer supports multiple embedding providers for vector search:

### No Embeddings

Without an embedding provider, memory search uses keyword matching only. The default provider is `local` (sentence-transformers), which requires no API key.

### OpenAI

```bash
pip install memorylayer-server[openai]
export MEMORYLAYER_EMBEDDING_OPENAI_API_KEY="sk-..."
export MEMORYLAYER_EMBEDDING_PROVIDER="openai"
memorylayer serve
```

### Google GenAI

```bash
pip install memorylayer-server[google]
export MEMORYLAYER_EMBEDDING_PROVIDER="google"
memorylayer serve
```

Requires `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` to be set.

### Sentence Transformers (Local, Default)

```bash
pip install memorylayer-server[local]
export MEMORYLAYER_EMBEDDING_PROVIDER="local"
memorylayer serve
```

No API key required. Models are downloaded and run locally.

## Context Environment

The Context Environment provides server-side Python sandboxes for memory analysis and computation.

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORYLAYER_CONTEXT_EXECUTOR` | Executor backend (`smolagents` or `restricted`) | `smolagents` |
| `MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS` | Timeout per code execution | `30` |
| `MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS` | Max captured stdout characters | `50000` |
| `MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS` | Max tokens for server-side LLM queries | `4096` |
| `MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES` | Memory limit per sandbox | `268435456` (256 MB) |
| `MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS` | Max iterations for RLM loops | `10` |
| `MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS` | Total timeout for RLM loops | `120` |

### Executor Backends

The Context Environment supports two executor backends:

```bash
# Use smolagents executor (default, recommended)
export MEMORYLAYER_CONTEXT_EXECUTOR="smolagents"

# Use restricted executor (no imports, AST-based whitelist)
export MEMORYLAYER_CONTEXT_EXECUTOR="restricted"
```

The `smolagents` backend provides a more capable execution environment with support for common data science libraries. The `restricted` backend uses an AST-based whitelist for maximum safety but limited functionality.

## Docker

The official Docker image (`scitrera/memorylayer-server`) comes with all optional dependencies pre-installed and defaults to the `local` embedding provider (sentence-transformers), so it works out of the box with no API keys.

### Container Defaults

| Variable | Docker Default | Code Default |
|----------|---------------|--------------|
| `MEMORYLAYER_SERVER_HOST` | `0.0.0.0` | `127.0.0.1` |
| `MEMORYLAYER_EMBEDDING_PROVIDER` | `local` | `local` |
| `MEMORYLAYER_DATA_DIR` | `/data` | — |

### Passing Environment Variables

Use `-e` flags to configure the container:

```bash
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBEDDING_PROVIDER=openai \
  -e MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-... \
  scitrera/memorylayer-server
```

### LLM Provider Configuration

Some features (reflection, smart extraction, context environment queries) require an LLM provider. LLM providers are configured via profiles:

```bash
# OpenAI LLM
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=openai \
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=sk-...

# Anthropic Claude
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=anthropic \
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=sk-ant-...

# Google Gemini
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=google \
-e MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=...
```

Without an LLM provider, the server still handles core memory operations (remember, recall, forget, associate) but features that require synthesis or generation will be unavailable.

### Data Persistence

Mount a volume to `/data` to persist the SQLite database across container restarts:

```bash
-v memorylayer-data:/data          # Named volume
-v /path/on/host:/data             # Bind mount
```

### Health Check

The container includes a built-in health check at `GET /health` (every 30s, 10s startup grace period). Use `GET /health/ready` for readiness checks that verify storage connectivity.

## Storage

### SQLite (Default)

The default storage backend is SQLite with sqlite-vec for vector operations.

The SQLite database is a single file that contains all memories, embeddings, associations, and session data.

### Database Location

The default storage path is `memorylayer.db` (relative to the data directory). The data directory defaults to `~/.config/memorylayer-server/` and can be overridden with `MEMORYLAYER_DATA_DIR`, so the database is typically created at `~/.config/memorylayer-server/memorylayer.db`.

```bash
# Override the data directory
export MEMORYLAYER_DATA_DIR="/var/lib/memorylayer"

# Or specify an explicit database path
export MEMORYLAYER_SQLITE_STORAGE_PATH="/var/lib/memorylayer/data.db"
```
