# MemoryLayer.ai

**API-first memory infrastructure for LLM-powered agents.**

MemoryLayer provides cognitive memory capabilities for AI agents, including episodic, semantic, procedural, and working
memory with vector-based retrieval and graph-based associations.

## Features

- **Cognitive Memory Architecture**: Episodic, Semantic, Procedural, and Working memory types
- **Vector Search**: SQLite with sqlite-vec for efficient similarity search
- **Graph Associations**: 25+ relationship types for memory connections
- **MCP Integration**: Model Context Protocol server for Claude and other LLMs
- **REST API**: FastAPI-based HTTP server
- **Multiple Embedding Providers**: OpenAI, Qwen3-VL, vLLM, sentence-transformers

## Installation

```bash
# Basic installation
pip install memorylayer-server

# With OpenAI embeddings
pip install memorylayer-server[openai]

# With local embeddings (sentence-transformers)
pip install memorylayer-server[local]

# With multimodal support (Qwen3-VL)
pip install memorylayer-server[multimodal]

# All embedding providers
pip install memorylayer-server[embeddings]
```

## Quick Start

### HTTP Server

```bash
# Start the REST API server
memorylayer serve --port 8080
```

### MCP Server

```bash
# Start MCP server for Claude integration
memorylayer mcp
```

## API Usage

```python
from memorylayer import MemoryLayerClient

client = MemoryLayerClient(base_url="http://localhost:8080")

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

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
