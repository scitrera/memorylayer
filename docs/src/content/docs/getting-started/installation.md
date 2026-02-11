---
title: Installation
description: Install the MemoryLayer server and client SDKs
sidebar:
  order: 2
---

## Server Installation

The MemoryLayer server is a Python FastAPI application that provides the REST API and MCP server.

### Basic Installation

```bash
pip install memorylayer-server
```

### With Embedding Providers

Choose an embedding provider based on your needs:

```bash
# OpenAI embeddings (recommended for production)
pip install memorylayer-server[openai]

# Local embeddings (no API key needed)
pip install memorylayer-server[local]

# Google GenAI embeddings
pip install memorylayer-server[google]

# All embedding providers (openai + local + google)
pip install memorylayer-server[embeddings]
```

### Start the Server

```bash
# HTTP server (default port 61001)
memorylayer serve

# Custom port
memorylayer serve --port 8080

# Debug mode
memorylayer serve --verbose
```

### Verify Installation

```bash
curl http://localhost:61001/health
```

You should see a JSON response confirming the server is running.

### Docker

Run the server as a Docker container with no dependencies to install:

```bash
# Basic — uses local sentence-transformers embeddings (no API key needed)
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  scitrera/memorylayer-server

# With OpenAI embeddings
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBEDDING_PROVIDER=openai \
  -e MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-... \
  scitrera/memorylayer-server

# With Google GenAI embeddings
docker run -d \
  --name memorylayer \
  -p 61001:61001 \
  -v memorylayer-data:/data \
  -e MEMORYLAYER_EMBEDDING_PROVIDER=google \
  -e MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY=... \
  scitrera/memorylayer-server
```

The Docker image includes all embedding providers and LLM integrations. Data is persisted in the `/data` volume. See the [Configuration](/server/configuration/) page for all available environment variables.

## Client SDKs

### Python SDK

```bash
pip install memorylayer-client
```

```python
from memorylayer import MemoryLayerClient

async with MemoryLayerClient(
    base_url="http://localhost:61001",
    workspace_id="my-workspace"
) as client:
    memory = await client.remember("Hello, MemoryLayer!")
```

### TypeScript SDK

```bash
npm install @scitrera/memorylayer-sdk
```

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace",
});

const memory = await client.remember("Hello, MemoryLayer!");
```

## Framework Integrations

### MCP Server (Claude Code / Claude Desktop)

```bash
npm install @scitrera/memorylayer-mcp-server
```

See [MCP Server Integration](/integrations/mcp-server/) for setup instructions.

### LangChain

```bash
pip install memorylayer-langchain
```

See [LangChain Integration](/integrations/langchain/) for usage guide.

### LlamaIndex

```bash
pip install memorylayer-llamaindex
```

See [LlamaIndex Integration](/integrations/llamaindex/) for usage guide.

## System Requirements

- **Server**: Python 3.12+
- **Python SDK**: Python 3.12+
- **TypeScript SDK**: Node.js 18+
- **MCP Server**: Node.js 18+

## Development Installation

For contributing to MemoryLayer, clone the repository and install in development mode:

```bash
# From the repository root
python -m venv .venv && source .venv/bin/activate

# Server
pip install -e "oss/memorylayer-core-python[dev]"

# Python SDK
pip install -e "oss/memorylayer-sdk-python[dev]"

# TypeScript SDK
cd oss/memorylayer-sdk-typescript
npm install
npm run build
```
