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
- **Multiple Embedding Providers** — OpenAI, Google GenAI, embed-server (self-hosted GPU via `memorylayer-embed-server`), and mock (testing)
- **Health Endpoints** — `/health` and `/health/ready` for monitoring and readiness checks

## Installation

```bash
# Basic installation
pip install memorylayer-server

# With OpenAI embeddings
pip install memorylayer-server[openai]

# With Google GenAI embeddings
pip install memorylayer-server[google]

# Self-hosted embeddings: install + run memorylayer-embed-server separately
# (no extras here — the main server only speaks HTTP to embed-server)
# pip install memorylayer-embed-server[gpu]

# All cloud embedding providers + LLM + document parsers
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

The official Docker image comes with all optional dependencies pre-installed and pins `MEMORYLAYER_EMBEDDING_PROVIDER=embed_server`, which delegates all GPU/ML work to a peer `memorylayer-embed-server` container — set `MEMORYLAYER_EMBED_SERVER_URL` accordingly, or override the provider entirely (`openai`/`google` for cloud, `hash` for a dependency-free lexical default):

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

The server exposes a REST API. Use any HTTP client, or install the Python SDK (`pip install memorylayer-client`) for a typed client:

```python
from memorylayer import MemoryLayerClient

async with MemoryLayerClient(base_url="http://localhost:61001") as client:
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
| `MEMORYLAYER_DATA_DIR` | `~` | State **root**, not the data directory itself. Data is written to `<root>/.config/memorylayer-server/`, so the default resolves to `~/.config/memorylayer-server` and `MEMORYLAYER_DATA_DIR=/data` resolves to `/data/.config/memorylayer-server`. |
| `MEMORYLAYER_SQLITE_STORAGE_PATH` | `memorylayer.db` | SQLite database path (relative to the resolved data directory) |
| `MEMORYLAYER_EMBEDDING_PROVIDER` | `hash` | Embedding provider (`hash`, `openai`, `google`, `embed_server`, `mock`). The default `hash` is lexical and dependency-free so the server runs with no setup — **set a real provider for production** (see below). |
| `MEMORYLAYER_SESSION_SERVICE` | `persistent` | `persistent` (survives restarts, uses the storage backend) or `in-memory` (ephemeral) |
| `MEMORYLAYER_EMBEDDING_OPENAI_API_KEY` | — | OpenAI API key |
| `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` | — | Google API key |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `384` (hash / embed_server) | Vector width. **Must match the model actually producing the embeddings** — see [Embedding dimensions](#embedding-dimensions) |
| `MEMORYLAYER_EMBED_SERVER_URL` | `http://localhost:61051` | Base URL for `memorylayer-embed-server` (used by `embed_server` provider) |
| `MEMORYLAYER_EMBED_TRANSPORT` | `http` | `http` for direct calls or `aether` for cross-DC mTLS via Aether |

### Deterministic memory and generation policy

`MEMORYLAYER_ENRICHMENT_POLICY` controls generative model use at the central LLM
service boundary. Its compatibility default is `generative`. `deterministic`
rejects every completion before provider selection while remember, RAG recall,
checkpoint capture, context packs, and context deltas continue to work.
`adaptive` allows only activities named in
`MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES` (default: `reflection,synthesis`).
Explicit LLM/agentic recall modes return `generation_not_allowed` when policy
does not authorize their required activity.

| Variable | Default | Rollback value / effect |
|----------|---------|-------------------------|
| `MEMORYLAYER_ENRICHMENT_POLICY` | `generative` | `generative` restores compatibility behavior; `deterministic` is the zero-generation guarantee |
| `MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES` | `reflection,synthesis` | Empty disables all adaptive generation |
| `MEMORYLAYER_GENERATION_MAX_CALLS` | `8` | Per-operation call ceiling |
| `MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS` | `100000` | Per-operation estimated input ceiling |
| `MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS` | `16384` | Per-operation reserved/actual output ceiling |
| `MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED` | `true` | `false` disables new extractive tier writes |
| `MEMORYLAYER_SESSION_CHECKPOINT_CAPTURE_ENABLED` | `true` | `false` disables the checkpoint endpoint without deleting captures |
| `MEMORYLAYER_SESSION_CHECKPOINT_MAX_BYTES` | `1048576` | Maximum UTF-8 raw segment size |
| `MEMORYLAYER_CONTEXT_PACK_ENABLED` | `true` | `false` disables pack/delta reads without deleting events |
| `MEMORYLAYER_CONTEXT_CURSOR_SECRET` | built-in local default | Set a stable private value shared by all server replicas |
| `MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS` | `30` | Retained delta window; expired cursors require a full pack |
| `MEMORYLAYER_RELATIONAL_RECALL_ENABLED` | `true` | `false` leaves stored relation/evidence rows intact but disables the recall arm |
| `MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED` | `true` | `false` returns compatibility confidence fields |
| `MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT` | `0` | `0` preserves count/detail behavior unless a caller supplies a budget |

Session recovery APIs are additive:

- `POST /v1/sessions/{session_id}/checkpoints` durably stores the exact raw
  transcript segment before deterministic indexing. Idempotency is scoped to
  the session and key.
- `POST /v1/sessions/{session_id}/context-pack` returns a ready-to-inject,
  deterministic rendering and opaque delta cursor under a hard token budget.
- `POST /v1/sessions/{session_id}/context-delta` returns at-least-once changes,
  including tombstones, and advances only through delivered events.

Recall accepts `budget_tokens`, `include_confidence`, and `include_relations`.
Remember accepts explicit typed entity relations with source spans. Structural
relations are stored separately from similarity associations and remain backed
by active source-memory evidence.

Connectors can also acquire general professional-work relations without an LLM
by placing a typed profile under `metadata.knowledge_work`. The profile maps
owners, assignees, authors, contributors, reviewers, approvers, projects,
dependencies, decisions, references, evidence, impacts, and topics onto the
canonical entity registry and relation store. See
[`docs/DESIGN_knowledge_work_relations.md`](docs/DESIGN_knowledge_work_relations.md).
Connector-shaped requests may instead declare `metadata.connector_type` and
provide ordinary source fields such as `title`, `assignee`, `creator`,
`reviewers`, `project`, `dependencies`, or their supported vendor forms. The
single-memory API, batch API, email adapter, and document pipeline normalize
that explicit envelope into the same auditable profile. Arbitrary metadata
without a connector declaration is never interpreted.

Prometheus metrics separate `memorylayer_generation_*`,
`memorylayer_embedding_*`, and `memorylayer_reranker_*` work. Generation metric
labels are limited to activity, policy, and outcome; provider/profile attribution
is emitted in logs.

### Embedding dimensions

**Choose your embedding model before ingesting data.** The vector width it produces
is written into every stored memory, so it is a property of your *data*, not just of
configuration — and nothing in the schema stops you from mixing widths.

| Provider | Default model | Dimensions |
|----------|---------------|------------|
| `hash` (default) | — (lexical feature hashing) | 384 |
| `embed_server` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `embed_server` (GPU/vLLM) | `Qwen/Qwen3-VL-Embedding-2B` | 2048 |
| `openai` | `text-embedding-3-small` | 1536 |
| `google` | `gemini-embedding-001` | 768 |

Set `MEMORYLAYER_EMBEDDING_DIMENSIONS` to match whatever model you actually run —
and set it on the `memorylayer-embed-server` peer too, if you use one.

**What goes wrong if you don't.** Two memories embedded at different widths cannot
be compared:

- On SQLite with `sqlite-vec` (the default), `vec_distance_cosine()` raises on a
  width mismatch and **the entire query returns no results** — not just the
  mismatched rows. A few wrong-width memories break recall for the whole workspace.
- If the `sqlite-vec` extension is unavailable, the pure-Python fallback scores
  mismatched vectors as `0.0`, so older memories **silently stop matching**.

No schema migration is involved — `memories.embedding` is a plain `BLOB` with no
dimension in the DDL — but changing dimensions on a populated deployment means
**re-embedding the existing memories**, or starting a fresh workspace.

Matching widths is necessary but not sufficient: the default `hash` provider is
also 384-d, so switching from `hash` to MiniLM produces vectors that compare
*without error* while being semantically unrelated to the ones already stored.
Re-embed when you change the model, not only when you change the width.

### Embedding Providers

**The default (`hash`) works offline with zero setup, but is not a semantic model.**
It hashes tokens into a vector, so it matches on shared words rather than meaning —
fine for local development, tests, and the retrieval-eval harness; not for production
recall quality. The server logs a warning at startup while it is in use. Pick one of
the real providers below for anything beyond trying it out.

The legacy in-process providers `local` (sentence-transformers), `colpali` (colpali-engine),
and `qwen3-vl` (qwen-vl-utils) were removed. All self-hosted/multi-vector embedding now
routes through the `embed_server` provider, which delegates to the standalone
`memorylayer-embed-server` package. Setting any of those legacy values for
`MEMORYLAYER_EMBEDDING_PROVIDER` raises a startup error with migration guidance.

**Embed-server (self-hosted)** — Run `memorylayer-embed-server` as a peer
process or container; the main server only speaks HTTP to it. This is what the
published Docker image is pinned to:

```bash
# In a GPU-equipped peer:
pip install memorylayer-embed-server[gpu]
memorylayer-embed-server serve --port 61051

# In the main server process:
export MEMORYLAYER_EMBEDDING_PROVIDER=embed_server
export MEMORYLAYER_EMBED_SERVER_URL=http://embed-host:61051
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

**Profile configuration variables** (replace `DEFAULT` with any profile name):

| Variable | Description |
|----------|-------------|
| `MEMORYLAYER_LLM_PROFILE_<NAME>_PROVIDER` | Provider (`openai`, `anthropic`, `google`) |
| `MEMORYLAYER_LLM_PROFILE_<NAME>_API_KEY` | API key |
| `MEMORYLAYER_LLM_PROFILE_<NAME>_MODEL` | Model name override |
| `MEMORYLAYER_LLM_PROFILE_<NAME>_BASE_URL` | Custom API base URL |
| `MEMORYLAYER_LLM_PROFILE_<NAME>_MAX_TOKENS` | Max response tokens |
| `MEMORYLAYER_LLM_PROFILE_<NAME>_TEMPERATURE` | Sampling temperature |

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
| `MEMORYLAYER_CONTEXT_MAX_OPERATIONS` | `1000000` | Max operations per sandbox execution |

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

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.
