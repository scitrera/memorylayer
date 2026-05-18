# MemoryLayer Embed Server

Stateless GPU embedding / transcription server for [MemoryLayer.ai](https://memorylayer.ai). Runs as a peer process or container alongside the main `memorylayer-server` and serves all heavy ML work (text embeddings, multi-vector / ColPali embeddings, OCR, transcription) over plain HTTP or [Aether](https://aetherlayer.ai) mTLS.

The main `memorylayer-server` core no longer ships any in-process embedding models. The previously bundled `local` (sentence-transformers), `colpali` (colpali-engine), and `qwen3-vl` providers were removed in v0.1.x and are now served exclusively via this package through the `embed_server` provider.

## Installation

```bash
# Core install (no GPU dependencies)
pip install memorylayer-embed-server

# GPU bundle: OCR + vLLM + ColPali
pip install "memorylayer-embed-server[gpu]"

# Everything: GPU + Google embeddings + observability
pip install "memorylayer-embed-server[all]"
```

Optional extras:

| Extra | Purpose |
|-------|---------|
| `ocr` | OCR via Transformers (GLM-OCR, etc.) — `transformers`, `torch`, `accelerate` |
| `vllm` | High-throughput vLLM-served text models |
| `colpali` | ColPali / late-interaction visual embedding |
| `google` | Google GenAI embedding/transcription proxy |
| `observability` | Prometheus `/metrics` + OpenTelemetry tracing |
| `gpu` | `ocr + vllm + colpali` |
| `all` | `gpu + google + observability` |
| `dev` | pytest + ruff |

Visual-tokenizer (Qwen3.5) lives in the proprietary `memorylayer-embed-server-enterprise` package; install that separately if you need it.

## Quick Start

```bash
# Start on the default port (61051)
memorylayer-embed serve

# Custom host/port
memorylayer-embed serve --host 0.0.0.0 --port 61051

# Verbose logging
memorylayer-embed -v serve
```

Verify the server is up:

```bash
curl http://localhost:61051/health
```

Point a `memorylayer-server` instance at it:

```bash
export MEMORYLAYER_EMBEDDING_PROVIDER=embed_server
export MEMORYLAYER_EMBED_SERVER_URL=http://localhost:61051
memorylayer serve
```

For cross-datacenter / mTLS deployments, use [Aether](https://aetherlayer.ai) transport:

```bash
export MEMORYLAYER_EMBED_TRANSPORT=aether
export MEMORYLAYER_EMBED_AETHER_TARGET=sv::memorylayer-embed::default
```

Aether is the optional service-mesh layer that provides mTLS, signed identity headers, on-behalf-of delegation, and cross-datacenter service discovery. See [aetherlayer.ai](https://aetherlayer.ai) for the full product overview.

## CLI

| Command | Description |
|---------|-------------|
| `memorylayer-embed serve` | Start the HTTP server (`--host`, `--port`) |
| `memorylayer-embed version` | Print the version |

Global flag `-v` / `--verbose` enables debug logging.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_EMBED_SERVER_HOST` | `127.0.0.1` | Bind address |
| `MEMORYLAYER_EMBED_SERVER_PORT` | `61051` | Listening port |
| `MEMORYLAYER_EMBED_MODEL_TEXT` | _(provider default)_ | Override the default text-embedding model |
| `MEMORYLAYER_EMBED_MODEL_COLPALI` | _(provider default)_ | Override the default ColPali model |

Refer to the provider modules under `src/memorylayer_embed_server/` for the full list of model-specific environment variables.

## Docker

A `Dockerfile` ships with this package. Test variants (`Dockerfile.test`, `Dockerfile.real-test`, `Dockerfile.real-test-full`) are used by the integration test harness. Expose port `61051`, mount any model cache directory you want to persist, and pass `MEMORYLAYER_EMBED_*` env vars at runtime.

## Health Checks

- `GET /health` — process is up
- `GET /health/ready` — model(s) loaded and ready to serve

The Docker image's healthcheck targets `/health`.

## Versioning

This package is released in lockstep with `memorylayer-server` (currently `0.1.22`). The version pin in `dependencies` keeps client and server protocol versions aligned.

## License

Apache 2.0 — see [LICENSE](../LICENSE).

## Links

- [memorylayer.ai](https://memorylayer.ai)
- [Documentation](https://docs.memorylayer.ai)
- [GitHub](https://github.com/scitrera/memorylayer)
