<p align="center">
  <strong>memorylayer.ai</strong>
</p>

<p align="center">
  <em>Persistent, queryable memory for stateless LLMs.</em>
</p>

<p align="center">
  <a href="https://memorylayer.ai">Website</a> &middot;
  <a href="https://docs.memorylayer.ai">Docs</a> &middot;
  <a href="https://github.com/scitrera/memorylayer">GitHub</a>
</p>

---

LLMs forget everything between sessions. MemoryLayer fixes that.

Store memories with a single call, recall them with semantic search, and let the knowledge graph surface connections that vector similarity alone can't find. Works with any LLM framework or directly via REST API.

```python
from memorylayer import sync_client

with sync_client() as memory:
    memory.remember("User prefers dark mode and TypeScript")

    results = memory.recall("What are the user's preferences?")
```

## Why MemoryLayer

- **Cognitive memory types** -- episodic, semantic, procedural, and working memory mirror how humans organize knowledge
- **Knowledge graph** -- 63 typed relationships across 11 categories enable multi-hop causal queries
- **Semantic tiering** -- memories are progressively summarized so you retrieve the right detail level without wasting context
- **Context sandbox** -- process hundreds of memories server-side in a persistent Python sandbox without consuming your context window
- **Recursive reasoning** -- inspired by [RLM](https://arxiv.org/abs/2512.24601), the server iteratively executes code and LLM queries over memory data
- **Smart extraction** -- every memory stored automatically extracts facts, builds associations, deduplicates, and categorizes
- **Adaptive decay** -- memory importance adjusts over time based on usage and feedback
- **Document ingestion** -- upload PDFs / DOCX / images and turn them into memories; optional ColPali multi-vector page search via the embed-server peer
- **Repository Planning Graph** -- sync code structure, traverse symbols and dependencies, maintain task overlays, and detect file/symbol conflicts
- **Skills + MCP registries** -- workspace-scoped libraries of agent skills and MCP server entries with 4-tier scope precedence (user / workspace / tenant / global)
- **MCP integration** -- first-class Model Context Protocol server (25 tools by default, 38 in the `full` profile) for Claude Code, Claude Desktop, OpenCode, Cursor, and other MCP-compatible tools
- **Optional Aether transport** -- run behind an [Aether](https://aetherlayer.ai) mesh for mTLS, signed identity headers, on-behalf-of delegation, durable task scheduling, and cross-datacenter routing

## Packages

| Package                                                                      | Install | Description                                             |
|------------------------------------------------------------------------------|---------|---------------------------------------------------------|
| **[memorylayer-core-python](./memorylayer-core-python)**                     | `pip install memorylayer-server` | FastAPI server with SQLite + sqlite-vec storage; optional Turso/libSQL backend |
| **[memorylayer-server-rpg-python](./memorylayer-server-rpg-python)**         | `pip install memorylayer-server-rpg` | Repository Planning Graph plugin: sync, traversal, overlays, conflicts, maintenance, and enrichment |
| **[memorylayer-embed-server](./memorylayer-embed-server)**                   | `pip install "memorylayer-embed-server[local]"` | Stateless embedding peer. `[local]` = CPU (sentence-transformers + ColPali); `[gpu]` adds vLLM, OCR, transcription |
| **[memorylayer-sdk-python](./memorylayer-sdk-python)**                       | `pip install memorylayer-client` | Python client SDK (async/sync, optional Aether transport) |
| **[memorylayer-sdk-typescript](./memorylayer-sdk-typescript)**               | `npm i @scitrera/memorylayer-sdk` | TypeScript/JavaScript client SDK                        |
| **[memorylayer-mcp-typescript](./memorylayer-mcp-typescript)**               | `npm i @scitrera/memorylayer-mcp-server` | MCP server -- 25 tools (default), up to 38 in `full`    |
| **[memorylayer-sdk-langchain-python](./memorylayer-sdk-langchain-python)**   | `pip install memorylayer-langchain` | LangChain integration                                   |
| **[memorylayer-sdk-llamaindex-python](./memorylayer-sdk-llamaindex-python)** | `pip install memorylayer-llamaindex` | LlamaIndex integration                                  |
| **[memorylayer-cc-plugin](./memorylayer-cc-plugin)**                         | see README | Claude Code plugin -- captures memory before compaction |
| **[memorylayer-opencode-plugin](./memorylayer-opencode-plugin)**             | `npm i @scitrera/memorylayer-opencode-plugin` | OpenCode plugin -- session briefings, recall hooks, compaction capture |
| **[memorylayer-explorer](./memorylayer-explorer)**                          | see README | (Work in Progress) WebUI                                |

## Quick Start

### 1. Start the server

```bash
pip install memorylayer-server
memorylayer serve
```

That's it — no API key, no peer container. The server stores to SQLite under
`~/.config/memorylayer-server` and is ready at `http://localhost:61001`.

To add repository code graphs, install the RPG plugin alongside the server:

```bash
pip install memorylayer-server-rpg
```

The server discovers it automatically and exposes `/v1/rpg`.

It starts on the `hash` embedding provider, which is **lexical, not semantic**: it
matches on shared words, which is enough to try the API but not for real retrieval
quality. The server says so in its startup log. When you want real embeddings:

```bash
# Self-hosted on CPU — no GPU, no API key
pip install "memorylayer-embed-server[local]" && memorylayer-embed serve --port 61051 &
export MEMORYLAYER_EMBEDDING_PROVIDER=embed_server
export MEMORYLAYER_EMBED_SERVER_URL=http://localhost:61051
export MEMORYLAYER_EMBEDDING_DIMENSIONS=384      # must match the embed model
memorylayer serve

# Cloud (pick one)
pip install "memorylayer-server[openai]"        # or [google], [all]
export MEMORYLAYER_EMBEDDING_PROVIDER=openai    # or google
export MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-...
memorylayer serve

# Self-hosted on GPU — adds vLLM, OCR, transcription
pip install "memorylayer-embed-server[gpu]" && memorylayer-embed serve --port 61051 &
```

The CPU option downloads `all-MiniLM-L6-v2` (~90 MB, 384-d) on first use and runs
in-process — it is the embed server's default provider.

### The one-command version

```bash
docker compose up
```

Brings up the server plus a CPU embed-server peer with real semantic embeddings —
no GPU, no API key. See [`docker-compose.yml`](./docker-compose.yml). The first
request is slow while model weights download; they are cached in a volume after
that.

#### Running the containers yourself

The server image ships all optional dependencies, includes the RPG plugin, and
exposes `/v1/rpg` without an additional install. It is pinned to
`MEMORYLAYER_EMBEDDING_PROVIDER=embed_server`, so it expects a
`memorylayer-embed-server` peer — point it at one, or override the provider:

```bash
# Cloud embeddings
docker run -d -p 61001:61001 -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBEDDING_PROVIDER=openai \
  -e MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-... \
  scitrera/memorylayer-server

# Self-hosted embed-server peer
docker run -d -p 61001:61001 -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBED_SERVER_URL=http://embed-host:61051 \
  scitrera/memorylayer-server
```

The embed-server image ships in two variants sharing one repository:

| tag | contents |
|---|---|
| `scitrera/memorylayer-embed-server:<version>` | **CPU.** sentence-transformers (384-d) + ColPali. Runs anywhere. |
| `scitrera/memorylayer-embed-server:<version>-cuda13` | **CUDA 13.** Adds vLLM, OCR and transcription. Needs an NVIDIA GPU. |

The tag names the CUDA major version because it is not an implementation detail —
the wheels and vLLM's JIT-compiled kernels are built against that toolkit.

### 2. Connect a client

**Python:**

```python
from memorylayer import MemoryLayerClient, MemoryType

async with MemoryLayerClient(base_url="http://localhost:61001") as client:
    # Store
    await client.remember(
        content="User prefers Python for backend development",
        type=MemoryType.SEMANTIC,
        importance=0.8,
        tags=["preferences", "programming"]
    )

    # Recall
    results = await client.recall(
        query="What programming languages does the user like?",
        limit=5
    )
```

**TypeScript:**

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-project"
});

await client.remember("User prefers TypeScript for new projects", {
  type: "semantic",
  importance: 0.8
});
```

### 3. Or use with Claude Code (MCP)

Add `.mcp.json` to your project root:

```json
{
  "mcpServers": {
    "memorylayer": {
      "command": "npx",
      "args": ["@scitrera/memorylayer-mcp-server"],
      "env": {
        "MEMORYLAYER_URL": "http://localhost:61001"
      }
    }
  }
}
```

The MCP server auto-detects your workspace from the git repo name. Claude gets 25 tools by default (38 in the `full` profile) -- remember, recall, reflect, sessions, context sandbox / RLM, chat threads, and skills/MCP-server registry helpers.

For the full Claude Code experience, also install the **[MemoryLayer plugin](./memorylayer-cc-plugin)** which adds pre-compaction memory capture, session briefings, and automatic memory triggers:

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add scitrera/memorylayer

# Install the plugin
claude plugin install memorylayer@memorylayer.ai
```

## Enterprise

MemoryLayer also offers an enterprise edition that builds on the open source core:

- **Scale** -- PostgreSQL + Redis backends, hot / warm / cold storage tiering, vector-graph compression
- **Security** -- OIDC + RBAC via [Aether](https://aetherlayer.ai), audit trails, KMS-backed token issuance, custom ontologies
- **Multimodal** -- unified handling of text, images, audio, video, and documents (visual-tokenizer add-on for the embed-server)
- **Data connector pack** -- S3, GitHub, Google Drive, Dropbox, Slack, Teams, Discord, web scraper, manual upload, and the VFS-backed `local_fs` watcher behind the same `/v1/data-providers` API
- **Cross-cluster orchestration** -- durable task scheduling, scheduled syncs, multi-DC GPU peer placement
- **Advanced sandbox** -- state checkpointing, stronger isolation, extended tool libraries
- **Collections, datasets, trajectories** -- higher-level grouping primitives for managing memory at scale
- **Admin API + dashboards** -- `/v1/users`, `/v1/applications`, `/v1/admin/*` for tenant-wide visibility and control

Visit [memorylayer.ai](https://memorylayer.ai) for details.

## Scitrera Forge

[Scitrera Forge](https://scitrera.ai) is a separate, **access-list-only** product that builds on MemoryLayer's open-source Repository Planning Graph (RPG) to orchestrate sandboxed agent swarms over real codebases. MemoryLayer provides the structural graph and memory substrate; Forge adds managed orchestration and execution. Currently restricted to approved organizations.

## License

Apache 2.0 -- see [LICENSE](./LICENSE) for details.
