# MemoryLayer.ai Server

**API-first memory infrastructure for LLM-powered agents.**

MemoryLayer provides cognitive memory capabilities for AI agents, including episodic, semantic, procedural, and working memory with vector-based retrieval, graph-based associations, and server-side computation sandboxes.

## Features

- **Cognitive Memory Architecture** — Episodic, semantic, procedural, and working memory types
- **Vector Search** — SQLite with sqlite-vec for efficient similarity search
- **Knowledge Graph** — 60+ relationship types organized into 11 categories for memory associations
- **Context Environment** — Server-side Python sandboxes for memory analysis and computation
- **Session Management** — Working memory with TTL and commit to long-term storage
- **REST API** — Full-featured HTTP API for all memory operations
- **Multiple Embedding Providers** — OpenAI, Google GenAI, sentence-transformers (local), and mock (testing)
- **Health Endpoints** — `/health` and `/health/ready` for monitoring and readiness checks

## Installation

```bash
# Basic installation
pip install memorylayer-server

# With OpenAI embeddings
pip install memorylayer-server[openai]

# With Google GenAI embeddings
pip install memorylayer-server[google]

# With local embeddings (sentence-transformers)
pip install memorylayer-server[local]

# All embedding providers
pip install memorylayer-server[all]
```

**Package name:** `memorylayer-server` (PyPI)
**Import name:** `memorylayer_server`

## Quick Start

### Start the HTTP Server

```bash
# Start on default port (61001)
memorylayer serve

# Custom port
memorylayer serve --port 8080

# Bind to all interfaces
memorylayer serve --host 0.0.0.0

# Debug mode
memorylayer serve --verbose
```

### Docker

The official Docker image comes with all optional dependencies pre-installed and defaults to `local` embeddings (no API key required):

```bash
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  scitrera/memorylayer-server
```

**With OpenAI embeddings:**

```bash
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBEDDING_PROVIDER=openai \
  -e MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-... \
  scitrera/memorylayer-server
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

# Create associations
await client.associate(
    source_id=memory.id,
    target_id=other_memory.id,
    relationship="related_to",
    strength=0.9
)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_SERVER_HOST` | `127.0.0.1` | Server bind address |
| `MEMORYLAYER_SERVER_PORT` | `61001` | Server port |
| `MEMORYLAYER_DATA_DIR` | `~/.config/memorylayer-server` | Data directory |
| `MEMORYLAYER_SQLITE_STORAGE_PATH` | `memorylayer.db` | SQLite database path (relative to data dir) |
| `MEMORYLAYER_EMBEDDING_PROVIDER` | `local` | Embedding provider (`openai`, `google`, `local`, `mock`) |
| `MEMORYLAYER_EMBEDDING_OPENAI_API_KEY` | — | OpenAI API key |
| `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` | — | Google API key |

### Embedding Providers

**Local (sentence-transformers)** — Default provider, no API key required:

```bash
pip install memorylayer-server[local]
export MEMORYLAYER_EMBEDDING_PROVIDER=local
memorylayer serve
```

**OpenAI:**

```bash
pip install memorylayer-server[openai]
export MEMORYLAYER_EMBEDDING_PROVIDER=openai
export MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-...
memorylayer serve
```

**Google GenAI:**

```bash
pip install memorylayer-server[google]
export MEMORYLAYER_EMBEDDING_PROVIDER=google
export MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY=...
memorylayer serve
```

**Mock (testing only):**

```bash
export MEMORYLAYER_EMBEDDING_PROVIDER=mock
memorylayer serve
```

### LLM Provider (Optional)

Some features (reflection, smart extraction, context environment queries) require an LLM provider configured via profiles:

```bash
# OpenAI
export MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=openai
export MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=sk-...

# Anthropic Claude
export MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=anthropic
export MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=sk-ant-...

# Google Gemini
export MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=google
export MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY=...
```

Without an LLM provider, core memory operations (remember, recall, forget, associate) work normally, but synthesis features will be unavailable.

### Context Environment

The Context Environment provides server-side Python sandboxes for memory analysis and computation. See [Context Environment documentation](https://docs.memorylayer.ai/guides/context-environment/) for details.

**Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_CONTEXT_EXECUTOR` | `smolagents` | Executor backend (`smolagents` or `restricted`) |
| `MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS` | `30` | Timeout per code execution |
| `MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS` | `50000` | Max captured stdout characters |
| `MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS` | `4096` | Max tokens for server-side LLM queries |
| `MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES` | `268435456` | Memory limit per sandbox (256 MB) |
| `MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS` | `10` | Max iterations for RLM loops |
| `MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS` | `120` | Total timeout for RLM loops |

## Storage

The default storage backend is **SQLite** with **sqlite-vec** for vector operations. The database file defaults to `~/.config/memorylayer-server/memorylayer.db` and contains all memories, embeddings, associations, and session data.

**Override the data directory:**

```bash
export MEMORYLAYER_DATA_DIR=/var/lib/memorylayer
```

**Override the database path:**

```bash
export MEMORYLAYER_SQLITE_STORAGE_PATH=/var/lib/memorylayer/data.db
```

## Recall Modes

The active recall mode is **RAG** (vector similarity + graph traversal). LLM and Hybrid modes are deprecated.

## MCP Integration

The Model Context Protocol (MCP) server is a **separate TypeScript package** (`@scitrera/memorylayer-mcp-server`), not part of this Python server CLI.

To use MemoryLayer with Claude Code or Claude Desktop:

1. Start the HTTP server: `memorylayer serve`
2. Install and configure the MCP server: `npm install -g @scitrera/memorylayer-mcp-server`

See the [MCP Server documentation](https://docs.memorylayer.ai/integrations/mcp-server/) for setup instructions.

## Health Checks

- **`GET /health`** — Basic health check (returns immediately)
- **`GET /health/ready`** — Readiness check (verifies storage connectivity)

The Docker image includes a built-in health check at `/health` (every 30s, 10s startup grace period).

## Documentation

- **Website:** [https://memorylayer.ai](https://memorylayer.ai)
- **Docs:** [https://docs.memorylayer.ai](https://docs.memorylayer.ai)
- **GitHub:** [https://github.com/scitrera/memorylayer](https://github.com/scitrera/memorylayer)

## License

Apache 2.0 License
