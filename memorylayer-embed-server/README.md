# MemoryLayer Embed Server

Stateless GPU embedding / transcription server for [MemoryLayer.ai](https://memorylayer.ai). Runs as a peer process or container alongside the main `memorylayer-server` and serves all heavy ML work (text embeddings, multi-vector / ColPali embeddings, OCR, transcription) over plain HTTP or [Aether](https://aetherlayer.ai) mTLS.

The main `memorylayer-server` core no longer ships any in-process embedding models. The previously bundled `local` (sentence-transformers), `colpali` (colpali-engine), and `qwen3-vl` providers were removed in v0.1.x and are now served exclusively via this package through the `embed_server` provider.

## Installation

```bash
# Local / CPU: sentence-transformers single-vector embeddings (the default provider)
pip install "memorylayer-embed-server[local]"

# Core install (server skeleton only — no embedding backend)
pip install memorylayer-embed-server

# GPU bundle: OCR + vLLM + ColPali
pip install "memorylayer-embed-server[gpu]"

# Everything: local + GPU + Google embeddings + observability
pip install "memorylayer-embed-server[all]"
```

The default single-vector provider is `sentence_transformers`, which needs the
`local` extra. A bare `pip install memorylayer-embed-server` gives you the server
skeleton with no embedding backend; it starts, but logs an error naming the extra
to install. Pick a different backend with
`MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` (see [Configuration](#configuration)).

Optional extras:

| Extra | Purpose |
|-------|---------|
| `local` | **Default single-vector backend** — sentence-transformers on CPU (`all-MiniLM-L6-v2`, 384-d) — `sentence-transformers`, `torch` |
| `ocr` | OCR via Transformers (GLM-OCR, etc.) — `transformers`, `torch`, `accelerate` |
| `vllm` | High-throughput vLLM-served text models |
| `colpali` | ColPali / late-interaction visual embedding |
| `google` | Google GenAI embedding/transcription proxy |
| `observability` | Prometheus `/metrics` + OpenTelemetry tracing |
| `gpu` | `ocr + vllm + colpali` |
| `all` | `local + gpu + google + observability` |
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
| `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` | `sentence_transformers` | `sentence_transformers` (local/CPU, 384-d), `vllm_subprocess` (GPU, 2048-d), `vllm` (in-process), `openai`, `google`, `colpali`, `mock`. **Changing this changes the vector dimension — see [Embedding dimensions](#embedding-dimensions).** |
| `MEMORYLAYER_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Single-vector model for the active provider |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `384` | Vector width. Must match the model — see [Embedding dimensions](#embedding-dimensions) |
| `MEMORYLAYER_EMBEDDING_ST_DEVICE` | _(auto)_ | `cpu`, `cuda`, `mps`… for the sentence-transformers provider. Pin to `cpu` to leave the GPU for ColPali |
| `MEMORYLAYER_EMBEDDING_ST_BATCH_SIZE` | `32` | Encode batch size for the sentence-transformers provider |
| `MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER` | `vllm_subprocess` | `colpali_inprocess` (in-process colpali-engine) or `vllm_subprocess` (out-of-process vLLM) |
| `MEMORYLAYER_EMBED_MODEL_TEXT` | _(provider default)_ | Override the default text-embedding model |
| `MEMORYLAYER_EMBEDDING_COLPALI_MODEL` | `ModernVBERT/colmodernvbert` | Multi-vector model. The vLLM path auto-upgrades the unloadable LoRA-adapter checkpoint to `colmodernvbert-merged`. |
| `MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR` | `1` | Hierarchical token-pool factor. `2` halves vectors with negligible recall loss, `3` cuts ~66% with ~97.8% perf retention per the ColPali paper. Must be the same for query- and doc-side calls. |
| `MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES` | `ColModernVBertForRetrieval` | `--hf-overrides` arch list for the multi-vec vLLM subprocess. Override when swapping to `ColQwen3_5` etc. |
| `MEMORYLAYER_EMBEDDING_VLLM_MV_MAX_LENGTH` | _(model default)_ | Per-multi-vec max sequence length. Leave unset to let vLLM derive from the model's config (avoids tripping ColModernVBert's 7999 limit). |
| `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` | `0.25` | Per-vLLM-subprocess GPU memory budget. Lower when sharing the GPU. |

Refer to the provider modules under `src/memorylayer_embed_server/` for the full list of model-specific environment variables.

### Embedding dimensions

**Pick a single-vector model before you ingest anything, and treat it as a
long-lived decision.** The vector width it produces is written into every stored
memory in `memorylayer-server`, so it is a property of your *data*, not just of
this server's configuration.

| Provider | Default model | Dimensions |
|----------|---------------|------------|
| `sentence_transformers` (default) | `sentence-transformers/all-MiniLM-L6-v2` | **384** |
| `vllm_subprocess` / `vllm` | `Qwen/Qwen3-VL-Embedding-2B` | **2048** |
| `openai` | `text-embedding-3-small` | 1536 |
| `google` | `gemini-embedding-001` | 768 |

Set `MEMORYLAYER_EMBEDDING_DIMENSIONS` on **both** this server and the main
`memorylayer-server` to match whichever model you run. The provider verifies the
value against the loaded model at startup and warns if they disagree (the model
wins — writing a wrong-width vector is worse than a noisy log).

**Why it matters more than a normal setting.** Two memories embedded at different
widths cannot be compared, and nothing in the storage schema prevents you from
mixing them:

- On SQLite with `sqlite-vec` (the default), `vec_distance_cosine()` raises on a
  width mismatch and **the whole query returns nothing** — not just the offending
  row. A handful of wrong-width memories break recall for the entire workspace.
- Without the `sqlite-vec` extension, the pure-Python fallback scores mismatched
  vectors as `0.0`, so the older memories **silently stop matching**.

Changing dimensions therefore needs **no schema migration** (the column is a plain
`BLOB`), but it does require **re-embedding existing memories** — or starting a
fresh workspace. Same-width is not enough either: swapping between two different
384-d models produces vectors that compare without error but mean nothing to each
other, which is the quiet version of the same bug.

### Multi-vector serving back-ends

The multi-vector / ColPali path has two interchangeable back-ends. Both speak the same wire shape on `/v1/embeddings/multi`, `/v1/embeddings/images`, and `/v1/score`:

- **`vllm_subprocess` (default)** — out-of-process `vllm serve --runner pooling` for batched, paged-attention throughput. Spawns one child process per multi-vec model; default model is `ModernVBERT/colmodernvbert-merged` (~1 GB unquantized) routed through the `ColModernVBertForRetrieval` arch class. **Needs a GPU.**
- **`colpali_inprocess`** — colpali-engine via HF transformers, in the embed-server process. Lightweight; loads the small `ModernVBERT/colmodernvbert` LoRA adapter (~250 MB). Best for tests and tiny deployments. **Runs on CPU** (auto-selects `cpu` when CUDA is absent, `float32` there), so this is the multi-vector back-end for a GPU-free box.

Both back-ends honor `MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR`; queries and documents must use the same factor or MaxSim geometry breaks.

Because the multi-vector default needs a GPU, a CPU-only deployment must opt
into the in-process back-end explicitly — the single- and multi-vector sides are
selected independently:

```bash
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=sentence_transformers  # (default) MiniLM on CPU
MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER=colpali_inprocess       # ColPali on CPU
MEMORYLAYER_EMBEDDING_ST_DEVICE=cpu      # sentence-transformers device
MEMORYLAYER_EMBEDDING_DEVICE=cpu         # ColPali device (separate knob)
```

GPU remains strongly preferred for multi-vector: on CPU a page-image
multivector takes several seconds, which is fine for dev and ad-hoc queries but
impractical for bulk document ingestion.

### LLM hosting (optional, OpenAI-compatible)

When `MEMORYLAYER_EMBED_LLM_ENABLED=true`, the server hosts one or more
`vllm serve` chat profiles and exposes `POST /v1/chat/completions`,
`POST /v1/completions`, and `GET /v1/models`. Profiles are declared via
`MEMORYLAYER_EMBED_LLM_PROFILES` and configured with per-profile env vars
`MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_*` (see `config.py` for the full list).

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
