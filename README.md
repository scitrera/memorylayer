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
- **Knowledge graph** -- 60+ typed relationships across 11 categories enable multi-hop causal queries
- **Semantic tiering** -- memories are progressively summarized so you retrieve the right detail level without wasting context
- **Context sandbox** -- process hundreds of memories server-side in a persistent Python sandbox without consuming your context window
- **Recursive reasoning** -- inspired by [RLM](https://arxiv.org/abs/2512.24601), the server iteratively executes code and LLM queries over memory data
- **Smart extraction** -- every memory stored automatically extracts facts, builds associations, deduplicates, and categorizes
- **Adaptive decay** -- memory importance adjusts over time based on usage and feedback
- **MCP integration** -- first-class Model Context Protocol server for Claude Code, Claude Desktop, Cursor, and other MCP-compatible tools

## Packages

| Package                                                                      | Install | Description                                             |
|------------------------------------------------------------------------------|---------|---------------------------------------------------------|
| **[memorylayer-core-python](./memorylayer-core-python)**                     | `pip install memorylayer-server` | FastAPI server with SQLite + sqlite-vec storage         |
| **[memorylayer-sdk-python](./memorylayer-sdk-python)**                       | `pip install memorylayer-client` | Python client SDK (async/sync)                          |
| **[memorylayer-sdk-typescript](./memorylayer-sdk-typescript)**               | `npm i @scitrera/memorylayer-sdk` | TypeScript/JavaScript client SDK                        |
| **[memorylayer-mcp-typescript](./memorylayer-mcp-typescript)**               | `npm i @scitrera/memorylayer-mcp-server` | MCP server -- 21 tools for LLM agents                   |
| **[memorylayer-sdk-langchain-python](./memorylayer-sdk-langchain-python)**   | `pip install memorylayer-langchain` | LangChain integration                                   |
| **[memorylayer-sdk-llamaindex-python](./memorylayer-sdk-llamaindex-python)** | `pip install memorylayer-llamaindex` | LlamaIndex integration                                  |
| **[memorylayer-cc-plugin](./memorylayer-cc-plugin)**                         | see README | Claude Code plugin -- captures memory before compaction |
| **[memorylayer-explorer](./memorylayer-explorer)**                          | see README | (Work in Progress) WebUI                                |

## Quick Start

### 1. Start the server

```bash
pip install memorylayer-server[local]
memorylayer serve
```

Or with Docker (no setup required):

```bash
docker run -d -p 61001:61001 -v memorylayer-data:/data scitrera/memorylayer-server
```

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

The MCP server auto-detects your workspace from the git repo name. Claude gets 21 memory tools -- remember, recall, reflect, associate, graph queries, sessions, and a full context sandbox.

For the full Claude Code experience, also install the **[MemoryLayer plugin](./memorylayer-cc-plugin)** which adds pre-compaction memory capture, session briefings, and automatic memory triggers:

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add scitrera/memorylayer

# Install the plugin
claude plugin install memorylayer@memorylayer.ai
```

## Enterprise

MemoryLayer also offers an enterprise edition that builds on the open source core:

- **Scale** -- PostgreSQL + Redis backends, hot/warm/cold data tiering, vector-graph compression
- **Security** -- RBAC, audit trails, custom ontologies
- **Multimodal** -- unified handling of text, images, audio, video, and documents
- **Advanced sandbox** -- state checkpointing, stronger isolation, extended tool libraries

Visit [memorylayer.ai](https://memorylayer.ai) for details.

## License

Apache 2.0 -- see [LICENSE](./LICENSE) for details.
