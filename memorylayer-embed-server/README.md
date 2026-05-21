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
| `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` | `vllm` | `vllm` (in-process), `vllm_subprocess`, `openai`, `google`, `colpali`, `mock` |
| `MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER` | `vllm_subprocess` | `colpali_inprocess` (in-process colpali-engine) or `vllm_subprocess` (out-of-process vLLM) |
| `MEMORYLAYER_EMBED_MODEL_TEXT` | _(provider default)_ | Override the default text-embedding model |
| `MEMORYLAYER_EMBEDDING_COLPALI_MODEL` | `ModernVBERT/colmodernvbert` | Multi-vector model. The vLLM path auto-upgrades the unloadable LoRA-adapter checkpoint to `colmodernvbert-merged`. |
| `MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR` | `1` | Hierarchical token-pool factor. `2` halves vectors with negligible recall loss, `3` cuts ~66% with ~97.8% perf retention per the ColPali paper. Must be the same for query- and doc-side calls. |
| `MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES` | `ColModernVBertForRetrieval` | `--hf-overrides` arch list for the multi-vec vLLM subprocess. Override when swapping to `ColQwen3_5` etc. |
| `MEMORYLAYER_EMBEDDING_VLLM_MV_MAX_LENGTH` | _(model default)_ | Per-multi-vec max sequence length. Leave unset to let vLLM derive from the model's config (avoids tripping ColModernVBert's 7999 limit). |
| `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` | `0.25` | Per-vLLM-subprocess GPU memory budget. Lower when sharing the GPU. |

Refer to the provider modules under `src/memorylayer_embed_server/` for the full list of model-specific environment variables.

### Multi-vector serving back-ends

The multi-vector / ColPali path has two interchangeable back-ends. Both speak the same wire shape on `/v1/embeddings/multi`, `/v1/embeddings/images`, and `/v1/score`:

- **`colpali_inprocess` (default)** — colpali-engine via HF transformers, in the embed-server process. Lightweight; loads the small `ModernVBERT/colmodernvbert` LoRA adapter (~250 MB). Best for tests and tiny deployments.
- **`vllm_subprocess` (production default)** — out-of-process `vllm serve --runner pooling` for batched, paged-attention throughput. Spawns one child process per multi-vec model; default model is `ModernVBERT/colmodernvbert-merged` (~1 GB unquantized) routed through the `ColModernVBertForRetrieval` arch class.

Both back-ends honor `MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR`; queries and documents must use the same factor or MaxSim geometry breaks.

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
