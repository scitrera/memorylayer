# Embedding Providers in `memorylayer-embed-server`

This document is the canonical reference for which embedding providers the
embed-server supports, how to select between them, and every configuration
knob each one exposes.

The embed-server runs **two providers in parallel**:

* a **single-vector** provider (one dense vector per input — used for
  storage in a normal vector DB and for query embeddings on retrieval),
  served at `POST /v1/embeddings`;
* a **multi-vector** provider (one matrix of token vectors per input —
  used for ColPali-style late-interaction retrieval), served at
  `POST /v1/embeddings/multi`, `POST /v1/embeddings/images`, and used by
  `POST /v1/score`.

These two providers are independent. You can mix and match them — e.g.
real ColPali for multi-vector retrieval while single-vector traffic goes
out to OpenAI.

---

## Quick reference

| Single-vector provider | Multi-vector provider | When to use |
|---|---|---|
| `mock`             | `mock`     | CI / unit tests / docker-compose smoke tests. No GPU, no network. |
| `vllm`             | `colpali`  | Default production: vLLM hosts a real text-embedding model (Qwen3-VL-Embedding-2B by default) **in-process**, ColPali handles multi-vector. Needs GPU + vllm/colpali extras. |
| `vllm_subprocess`  | `colpali`  | Same shape as `vllm`, but vLLM runs as a child process (separate asyncio loop + CUDA context). Better isolation: vLLM crashes don't take down the embed-server; memory accounting is independent. The embed-server talks to it over OpenAI-compat HTTP at `http://127.0.0.1:18000/v1`. |
| `openai`           | `colpali`  | Self-hosted multi-vector retrieval, single-vector outsourced to OpenAI (or any OpenAI-compat endpoint — vLLM, LocalAI, Ollama, …). No vLLM dependency on-box. |
| `google`           | `colpali`  | Same shape as `openai` but using Google GenAI for single-vector. |
| `colpali`          | `colpali`  | One model for both — ColPali serves multi-vector natively and mean-pools its token vectors for the single-vector path. No vLLM needed. |

The selector for the single-vector provider is the env var
`MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` (default `vllm`). The
multi-vector side is always ColPali in production, or mocked when
`MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true`.

---

## Selection precedence

When the embed-server boots, `_setup_dual_embedding_service` decides
which provider to instantiate using the rules below, applied in order:

1. If `MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true` →
   both providers are deterministic numpy mocks. **All other selection
   flags are ignored.**
2. If `MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE=true` (or the equivalent
   `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=colpali`) →
   the ColPali multi-vector provider is reused as the single-vector
   provider. The single-vector init step is skipped, so the vLLM /
   OpenAI / Google plugins are never imported.
3. Otherwise the single-vector provider is dispatched by
   `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` (default `vllm`). Valid
   values: `vllm`, `vllm_subprocess`, `openai`, `google`, `mock`,
   `colpali`.

The multi-vector provider is always ColPali except in mock mode.

If a provider import fails (e.g. `vllm` not installed in the image),
that provider is left `None` and a warning is logged — the server boots,
but requests targeting that capability return 503.

---

## Provider details

### `mock` — deterministic numpy mocks

* Module: `memorylayer_embed_server.services.embedding.mock_providers`
* Classes: `MockSingleVectorProvider`, `MockMultiVectorProvider`
* No external deps beyond numpy. No GPU. No network.
* Outputs are SHA-256-seeded from the input so the same text/image
  always produces the same vector — useful for golden tests.
* Single-vector mock subclasses `MultimodalEmbeddingProvider` (mirrors
  the real vLLM Qwen3-VL provider) so the `/v1/embeddings/images?mode=single`
  route exercises the same code path.

**Single-vector mock parameters** (constructor-level; not env-tunable):

| Param | Default |
|---|---|
| `dimensions` | `384` |

**Multi-vector mock parameters**:

| Param | Default |
|---|---|
| `dimensions` | `128` (per token) |
| `num_tokens` | `16` |

**Enable via env**:

```bash
export MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true
```

---

### `vllm` — in-process vLLM (`AsyncLLM`)

* Module: `memorylayer_embed_server.services.embedding.vllm`
* Class: `VLLMEmbeddingProvider`
* Uses `vllm.v1.engine.async_llm.AsyncLLM` with `runner="pooling"` and
  `PoolingParams(task="embed")` so the response is the sentence-pooled
  embedding (not per-token).
* Multimodal: supports image embedding via the
  `{"prompt": "...", "multi_modal_data": {"image": <PIL.Image>}}`
  prompt shape, so models like `Qwen/Qwen3-VL-Embedding-2B` are first-class.
* Requires the `[vllm]` extra (`pip install memorylayer-embed-server[vllm]`)
  and a CUDA-capable host. `Dockerfile.real-test-full` installs this.

**Env knobs**:

| Env var | Default | Notes |
|---|---|---|
| `MEMORYLAYER_EMBEDDING_VLLM_MODEL` | (unset) | Provider-specific model override. Wins over `MEMORYLAYER_EMBEDDING_MODEL`. Set this when vLLM coexists with ColPali (or any other model-using provider) so the two don't collide on the shared key. |
| `MEMORYLAYER_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Fallback if the vLLM-specific override is unset. HuggingFace repo id. |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `2048` | Truncates the output to this many dims. |
| `MEMORYLAYER_EMBEDDING_VLLM_DTYPE` | `bfloat16` | vLLM `dtype` arg. |
| `MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH` | `32768` | `max_model_len` for the vLLM engine. |
| `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` | `0.25` | **Important**: vLLM's stock default is `0.92` which is catastrophic on unified-memory systems (DGX Spark, Jetson) and when sharing the GPU with ColPali. Tune up on dedicated inference boxes. |
| `MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER` | `false` | Set to `true` to disable torch.compile + CUDA-graph capture. Use this on CUDA `*-runtime-*` base images that lack `nvcc`. Costs latency in exchange for fast cold-start and no compile dependency. |

**Enable**:

```bash
export MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm   # (also the default)
```

---

### `vllm_subprocess` — out-of-process vLLM

* Module: `memorylayer_embed_server.services.embedding.vllm_subprocess`
* Class: `VLLMSubprocessEmbeddingProvider`
* Spawns `vllm serve` as a child process (its own asyncio loop, its own
  CUDA context). The embed-server talks to it via the `openai` async
  client at `http://127.0.0.1:18000/v1`.
* Why pick this over in-process `vllm`?
  * **Crash isolation** — vLLM CUDA OOM, kernel hangs, or assertion
    failures kill only the subprocess; the embed-server's FastAPI loop
    keeps serving (multi-vector ColPali requests continue, the
    subprocess restarts on the next request).
  * **Independent memory accounting** — vLLM's `gpu_memory_utilization`
    only competes with itself; the embed-server's other torch
    work (ColPali) lives in a separate process.
  * **Asyncio-loop independence** — vLLM's own event loop runs in the
    child; the embed-server's loop only handles cheap HTTP calls.
* Why not? Subprocess startup is slower than in-process import (one-time
  cost). Otherwise the surface matches the in-process `vllm` provider,
  including multimodal — see "Multimodal" below.
* **Multimodal**: `embed_image` / `embed_multimodal(text+image)` POST a
  chat-style `messages` payload to `/v1/embeddings` (vLLM's VLM
  extension). Image bytes / file paths are base64-encoded into a
  `data:image/...` URL; pre-encoded `data:` URLs and `http(s)://` URLs
  pass through unchanged (vLLM fetches them server-side). Requires a
  vision-language embedding model in the subprocess (e.g.
  `Qwen/Qwen3-VL-Embedding-2B`).
* Same config knobs as the in-process `vllm` provider; the dispatcher
  reads the same env vars when constructing the subprocess argv.
* Requires the `[vllm]` extra so `vllm serve` is on PATH inside the
  container.

**Env knobs** (the in-process-vLLM knobs all apply here too):

| Env var | Default | Notes |
|---|---|---|
| `MEMORYLAYER_EMBEDDING_VLLM_MODEL` | (unset) | Provider-specific model override (shared with the in-process `vllm` provider). |
| `MEMORYLAYER_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Fallback. |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `2048` | Output truncation (applied client-side). |
| `MEMORYLAYER_EMBEDDING_VLLM_DTYPE` | `bfloat16` | Passed to `vllm serve --dtype`. |
| `MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH` | `32768` | `--max-model-len`. |
| `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` | `0.25` | `--gpu-memory-utilization`. |
| `MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER` | `false` | `--enforce-eager`. |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST` | `127.0.0.1` | Bind/connect host for the child vllm serve. |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT` | `18000` | Child vllm serve port. Pick a free port if 18000 collides. |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC` | `600` | How long to wait for the child to return 200 on `/health` before giving up. Cold model load can take minutes. |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_CMD` | `vllm` | Binary name (looked up on PATH). Override to a full path or a wrapper script (e.g. for legacy `--task embed` flag mapping). |

**Lifecycle**

* The subprocess is started lazily on the first embedding request
  (or eagerly during model preload when
  `MEMORYLAYER_EMBED_PRELOAD_MODELS=true`).
* `vllm serve` stderr is forwarded line-by-line into the embed-server's
  logger so its output shows up in the same stream as the embed-server.
* On embed-server shutdown, the child process group receives `SIGTERM`
  (followed by `SIGKILL` after 10 s if it hasn't exited). This is wired
  through `shutdown_services` and runs before the rest of the
  framework's plugin teardown.

**Enable**:

```bash
export MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_subprocess
# All other vllm knobs apply as for the in-process variant:
export MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL=0.25
export MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER=true     # if base lacks nvcc
```

---

### `openai` — OpenAI or any OpenAI-compatible HTTP endpoint

* Module (lives in OSS server, reused by embed-server):
  `memorylayer_server.services.embedding.openai`
* Class: `OpenAIEmbeddingProvider`
* Uses the `openai` python SDK (`AsyncOpenAI`). Pure HTTP — no GPU.
* Works against the real OpenAI API and against any OpenAI-compatible
  endpoint (a sibling vLLM serve, LocalAI, Ollama, fastembed-server,
  etc.) by setting `MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL`.

**Env knobs**:

| Env var | Default | Notes |
|---|---|---|
| `MEMORYLAYER_EMBEDDING_OPENAI_MODEL` | (unset) | Provider-specific model override. Wins over `MEMORYLAYER_EMBEDDING_MODEL`. Set this when the OpenAI provider runs alongside ColPali so the two don't collide on the shared key. |
| `MEMORYLAYER_EMBEDDING_MODEL` | `text-embedding-3-small` | Fallback if the OpenAI-specific override is unset. Forwarded as `model=` to the API. |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `1536` | OpenAI returns this many dims for `-small`. Match your endpoint's output. |
| `MEMORYLAYER_EMBEDDING_OPENAI_API_KEY` | `x` (placeholder) | Required for OpenAI. Many OpenAI-compat servers don't enforce auth — any non-empty value works. |
| `MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL` | `None` (→ OpenAI cloud) | Set to e.g. `http://localhost:8000/v1` for a local vLLM serve, `http://ollama:11434/v1` for Ollama, etc. |

**Enable**:

```bash
export MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=openai
export MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-...
# Optional — pin to a non-OpenAI endpoint:
export MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL=http://host.docker.internal:8000/v1
export MEMORYLAYER_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export MEMORYLAYER_EMBEDDING_DIMENSIONS=1024
```

Requires installing the OSS server with the `openai` extra (the
embed-server's `Dockerfile.real-test` already does this).

---

### `google` — Google GenAI (Gemini embeddings)

* Module (lives in OSS server, reused by embed-server):
  `memorylayer_server.services.embedding.google`
* Class: `GoogleEmbeddingProvider`
* Uses the `google-genai` python SDK. Pure HTTP — no GPU.

**Env knobs**:

| Env var | Default | Notes |
|---|---|---|
| `MEMORYLAYER_EMBEDDING_GOOGLE_MODEL` | (unset) | Provider-specific model override. Wins over `MEMORYLAYER_EMBEDDING_MODEL`. Set this when the Google provider runs alongside ColPali so the two don't collide on the shared key. |
| `MEMORYLAYER_EMBEDDING_MODEL` | `gemini-embedding-001` | Fallback if the Google-specific override is unset. Forwarded to the Gemini API. |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `768` | Match the model's output. |
| `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` | `None` | Required. |

**Enable**:

```bash
export MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=google
export MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY=...
```

Requires installing the OSS server with the `google` extra
(`Dockerfile.real-test` includes this).

---

### `colpali` — ColPali multi-vector (in-process)

* Module: `memorylayer_embed_server.services.embedding.colpali`
* Class: `ColPaliEmbeddingProvider`
* Always serves the **multi-vector** path (`/v1/embeddings/multi`,
  `/v1/embeddings/images?mode=multi`, `/v1/score` MaxSim helper).
* Can also serve the **single-vector** path when
  `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=colpali` (or the legacy
  `MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE=true`). In that mode the
  provider mean-pools its token-level output into a single dense vector
  — useful if you want a GPU embedding path but don't want vLLM in the
  image.
* Auto-detects which ColPali family from the model name
  (`qwen2.5` → ColQwen2.5, `qwen2` → ColQwen2, `modernvbert` →
  ColModernVBert, `colpali` → original ColPali). Defaults to
  ColModernVBert (MIT-licensed, smallest).
* Requires the `[colpali]` extra (`pip install memorylayer-embed-server[colpali]`)
  which pulls in `colpali-engine` + `torch`. GPU strongly recommended.

**Env knobs**:

| Env var | Default | Notes |
|---|---|---|
| `MEMORYLAYER_EMBEDDING_COLPALI_MODEL` | (unset) | Provider-specific model override. Wins over `MEMORYLAYER_EMBEDDING_MODEL`. Set this when ColPali coexists with vLLM / OpenAI / Google so the two don't collide on the shared key. |
| `MEMORYLAYER_EMBEDDING_MODEL` | `ModernVBERT/colmodernvbert` | Fallback if the ColPali-specific override is unset. HuggingFace repo id. |
| `MEMORYLAYER_EMBEDDING_DEVICE` | auto (`cuda` if available, else `cpu`) | Pass `cuda`, `cpu`, or any torch device spec. |
| `MEMORYLAYER_EMBEDDING_REVISION` | `main` | Model revision (branch / tag / commit). |
| `MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT` | `4` | Max in-flight ColPali GPU requests. Excess requests wait on an `asyncio.Semaphore`. Tune up on dedicated GPUs, down on shared/unified-memory boxes. |
| `MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC` | `0` (wait forever) | If positive, requests that wait longer than this for a slot are rejected with HTTP 503 + `Retry-After: 1`. |

**Concurrency / load-shedding semantics**

* The provider holds a `asyncio.Semaphore(MAX_CONCURRENT)`. Each GPU-bound
  method (`embed_text_multivector`, `embed_batch_multivector`,
  `embed_image_multivector`, `embed_images_batch_multivector`) acquires
  one slot for its full duration. CPU work — image decode, prompt
  prep — happens before the slot is acquired, so the GPU critical
  section stays tight.
* Waits longer than 100 ms log at `WARNING` so contention is visible.
  Sub-100 ms acquisitions log at `DEBUG`.
* When `MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC > 0` and a
  request can't acquire a slot in time, the provider raises
  `ColPaliQueueTimeoutError`. The embed-server's FastAPI app installs a
  global exception handler that turns this into `503 Service
  Unavailable` with `Retry-After: 1` and a JSON body containing
  `wait_seconds` + `max_concurrent` so clients can shed load.
* The single-vector vLLM path is intentionally **not** gated — vLLM has
  its own continuous-batching scheduler that handles admission control.
  Adding a Python-level semaphore on top would serialize requests it's
  happy to batch.

**Metrics emitted (OTel-compatible)**

When `MEMORYLAYER_METRICS_SERVICE=prometheus` is set on the embed-server
(and the `[observability]` extra is installed), the slot wrapper emits:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `embed_server_colpali_gpu_slot_total` | counter | `result={acquired,timeout}` | One per `_gpu_slot()` attempt, terminal outcome. |
| `embed_server_colpali_gpu_slot_wait_seconds` | histogram | `result={acquired,timeout}` | Time spent waiting for the slot before acquire/timeout. |
| `embed_server_colpali_gpu_in_flight` | gauge | — | Current in-flight ColPali GPU request count. |
| `embed_server_colpali_gpu_utilization` | gauge | — | `in_flight / max_concurrent`, 0..1. |

These are scraped from each embed-server pod's `/metrics` endpoint
(also gated on `MEMORYLAYER_METRICS_SERVICE=prometheus`). Any
OpenTelemetry Collector with a Prometheus receiver can forward them to
OTLP-native backends; no separate exporter library is required on the
embed-server. Set `MEMORYLAYER_OTEL_ENABLED=true` for distributed
tracing alongside metrics.

**Load-aware routing via `/health/load`**

Independent of the metrics backend, the embed-server always exposes a
JSON load summary at `GET /health/load`:

```json
{
  "providers": {
    "colpali_multi_vector": {
      "in_flight": 1,
      "max_concurrent": 4,
      "utilization": 0.25
    }
  },
  "utilization": 0.25
}
```

The top-level `utilization` is the max across registered providers.
A load-aware proxy (HAProxy with an external check, Envoy with
metadata, custom nginx module, etc.) can poll this and route to the
replica with the lowest utilization. No Prometheus / OTel pipeline
required for the LB path.

> 💡 **About model collisions.** Each provider also reads the shared
> `MEMORYLAYER_EMBEDDING_MODEL` as a fallback. When two providers run
> together (e.g. vLLM + ColPali in the default production setup),
> setting that shared key would point both at the same model — almost
> certainly wrong. Use the provider-specific overrides
> (`MEMORYLAYER_EMBEDDING_VLLM_MODEL`, `MEMORYLAYER_EMBEDDING_COLPALI_MODEL`,
> `MEMORYLAYER_EMBEDDING_OPENAI_MODEL`, `MEMORYLAYER_EMBEDDING_GOOGLE_MODEL`)
> when running side-by-side, or leave both unset so each provider uses
> its own default.

---

## Multi-vector provider

The multi-vector side has only two states:

* **Mock** — when `MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true`.
* **ColPali** — every other time. Configured via the env vars in the
  ColPali section above.

There is no analogue to `EMBED_SERVER_SINGLE_VECTOR_PROVIDER` for the
multi-vector side — if you don't want ColPali at runtime, the
multi-vector endpoints will 503.

---

## Full env-var reference

### Selection / mode

| Env var | Default | Effect |
|---|---|---|
| `MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS` | `false` | Replace **both** providers with deterministic mocks. Highest priority. |
| `MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE` | `false` | Reuse ColPali as the single-vector provider (mean-pooled). Skips the single-vector init step entirely. |
| `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER` | `vllm` | One of `vllm`, `openai`, `google`, `colpali`, `mock`. `colpali` is an alias for `MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE=true`. |

### Server boot / lifecycle

| Env var | Default | Effect |
|---|---|---|
| `MEMORYLAYER_EMBED_SERVER_HOST` | `0.0.0.0` | uvicorn bind host. |
| `MEMORYLAYER_EMBED_SERVER_PORT` | `61051` | uvicorn port. |
| `MEMORYLAYER_EMBED_PRELOAD_MODELS` | `true` | If true, warm up registered providers during FastAPI lifespan (single + multi). |
| `EMBED_SERVER_RUN_SIDECAR` | `true` | Container entrypoint flag. `true` → `proxy-sidecar` runs as PID 1 and supervises the embed FastAPI. `false` → run the embed FastAPI directly on plain HTTP. |

### Observability

Install with `pip install memorylayer-embed-server[observability]`
(or `[all]`). The embed-server reuses the OSS server's pluggable
`MetricsService` and `OTelInitPlugin`.

| Env var | Default | Effect |
|---|---|---|
| `MEMORYLAYER_METRICS_SERVICE` | `noop` | `noop` discards observations. `prometheus` activates the ColPali metrics + the `/metrics` Prometheus exposition endpoint. |
| `MEMORYLAYER_OTEL_ENABLED` | `false` | Initialize the OpenTelemetry SDK. Auto-instruments FastAPI / SQLAlchemy / Redis / httpx for traces. |
| `MEMORYLAYER_OTEL_EXPORTER` | `none` | `none` (in-process only), `console` (stdout), or `otlp` (push to a collector via gRPC). |
| `MEMORYLAYER_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint when `MEMORYLAYER_OTEL_EXPORTER=otlp`. |
| `MEMORYLAYER_OTEL_SERVICE_NAME` | `memorylayer` | Resource attribute for spans. Set to e.g. `memorylayer-embed-server` to distinguish from the main server in your tracing UI. |

`GET /health/load` is always available regardless of these settings —
load balancers polling it don't need a metrics pipeline.

### Single-vector — vLLM (in-process)

| Env var | Default |
|---|---|
| `MEMORYLAYER_EMBEDDING_VLLM_MODEL` | (unset; falls back to `MEMORYLAYER_EMBEDDING_MODEL`) |
| `MEMORYLAYER_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `2048` |
| `MEMORYLAYER_EMBEDDING_VLLM_DTYPE` | `bfloat16` |
| `MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH` | `32768` |
| `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` | `0.25` |
| `MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER` | `false` |

### Single-vector — vLLM (subprocess)

Inherits all of the in-process knobs above, plus:

| Env var | Default |
|---|---|
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST` | `127.0.0.1` |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT` | `18000` |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC` | `600` |
| `MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_CMD` | `vllm` |

### Single-vector — OpenAI / OpenAI-compat

| Env var | Default |
|---|---|
| `MEMORYLAYER_EMBEDDING_OPENAI_MODEL` | (unset; falls back to `MEMORYLAYER_EMBEDDING_MODEL`) |
| `MEMORYLAYER_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `1536` |
| `MEMORYLAYER_EMBEDDING_OPENAI_API_KEY` | `x` (placeholder; required for real OpenAI) |
| `MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL` | `None` (uses OpenAI cloud) |

### Single-vector — Google

| Env var | Default |
|---|---|
| `MEMORYLAYER_EMBEDDING_GOOGLE_MODEL` | (unset; falls back to `MEMORYLAYER_EMBEDDING_MODEL`) |
| `MEMORYLAYER_EMBEDDING_MODEL` | `gemini-embedding-001` |
| `MEMORYLAYER_EMBEDDING_DIMENSIONS` | `768` |
| `MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY` | `None` (required) |

### Multi-vector — ColPali

| Env var | Default |
|---|---|
| `MEMORYLAYER_EMBEDDING_COLPALI_MODEL` | (unset; falls back to `MEMORYLAYER_EMBEDDING_MODEL`) |
| `MEMORYLAYER_EMBEDDING_MODEL` | `ModernVBERT/colmodernvbert` |
| `MEMORYLAYER_EMBEDDING_DEVICE` | auto |
| `MEMORYLAYER_EMBEDDING_REVISION` | `main` |
| `MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT` | `4` |
| `MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC` | `0` (wait forever) |

### Optional services (orthogonal to embedding selection)

| Env var | Default | Effect |
|---|---|---|
| `MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED` | `true` | Set up the transcription cascade (GLM-OCR → DeepSeek-OCR → Gemini Flash). Only used by `/v1/transcribe`. |
| `MEMORYLAYER_EMBED_GLM_OCR_MODEL` | `zai-org/GLM-OCR` | |
| `MEMORYLAYER_EMBED_DEEPSEEK_OCR_MODEL` | `deepseek-ai/DeepSeek-OCR-2` | |
| `MEMORYLAYER_EMBED_GEMINI_MODEL` | `gemini-3-flash-preview` | |

The visual-tokenizer service (`/v1/visual-tokenize`) lives in the
proprietary `memorylayer-embed-server-enterprise` overlay package; its
config keys are documented separately.

---

## Recipes

### Lightweight CI test (no GPU, no network)

```bash
docker compose \
    -f tests/integration/docker-compose.embed-chain.yml \
    up --build -d
```

Effectively:

```
MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true
EMBED_SERVER_RUN_SIDECAR=false
```

`Dockerfile.test` ships these as defaults.

### Heavy real-model — ColPali for both single and multi

```
MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=false
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=colpali
MEMORYLAYER_EMBEDDING_MODEL=ModernVBERT/colmodernvbert
MEMORYLAYER_EMBEDDING_DEVICE=cuda
```

Smallest GPU footprint that exercises real models. See
`Dockerfile.real-test` and `docker-compose.embed-chain.real.yml`.

### Full real — vLLM (Qwen3-VL) for single, ColPali for multi

```
MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=false
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm
# Each provider's per-default model is correct (Qwen3-VL-Embedding-2B
# for vLLM; ColModernVBert for ColPali). Leave MEMORYLAYER_EMBEDDING_MODEL
# unset, or pin per-provider:
#   MEMORYLAYER_EMBEDDING_VLLM_MODEL=Qwen/Qwen3-VL-Embedding-2B
#   MEMORYLAYER_EMBEDDING_COLPALI_MODEL=ModernVBERT/colmodernvbert
MEMORYLAYER_EMBEDDING_DEVICE=cuda
MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL=0.25     # tune up on dedicated boxes
MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER=true    # only if base image lacks nvcc
```

See `Dockerfile.real-test-full` and
`docker-compose.embed-chain.real-full.yml`. Don't forget `ipc: host`
(or `--ipc=host`) for vLLM.

### Hybrid — OpenAI-compat for single, ColPali for multi (no vLLM)

Against real OpenAI (use the provider-specific model key so it can't
clobber ColPali's model):

```
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=openai
MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=sk-...
MEMORYLAYER_EMBEDDING_OPENAI_MODEL=text-embedding-3-small
MEMORYLAYER_EMBEDDING_DIMENSIONS=1536
MEMORYLAYER_EMBEDDING_DEVICE=cuda   # for ColPali
# ColPali keeps its own default (ModernVBERT/colmodernvbert); override
# explicitly with MEMORYLAYER_EMBEDDING_COLPALI_MODEL if needed.
```

Against a self-hosted OpenAI-compat endpoint (Ollama, LocalAI, sibling
vLLM serve, …):

```
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=openai
MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL=http://embed-llm:8000/v1
MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=x
MEMORYLAYER_EMBEDDING_OPENAI_MODEL=BAAI/bge-large-en-v1.5
MEMORYLAYER_EMBEDDING_DIMENSIONS=1024
MEMORYLAYER_EMBEDDING_DEVICE=cuda
```

See `docker-compose.embed-chain.real-openai-single.yml`.

### Hybrid — Google Gemini for single, ColPali for multi

```
MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=google
MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY=...
MEMORYLAYER_EMBEDDING_GOOGLE_MODEL=gemini-embedding-001   # optional, isolates from ColPali
MEMORYLAYER_EMBEDDING_DEVICE=cuda   # for ColPali
```

---

## Troubleshooting

* **`/v1/embeddings` returns 503 ("Single-vector embedding service not configured")**
  Your selected provider failed to initialize. Check the embed-server
  logs at boot — the failure is logged as a `WARNING`. Common causes:
  vLLM not installed in the image, missing API key, dim/model mismatch.

* **`pydantic_core.ValidationError: ... input should be a valid number ... input_value=[...,...]`**
  vLLM is returning per-token output instead of pooled. The provider
  applies `PoolingParams(task="embed")`; if you see this error you're
  likely on a model that doesn't expose a pooling task — try a
  different model or run with `colpali` mode instead.

* **`torch._inductor.exc.InductorError: PermissionError: 'nvcc'`**
  Your CUDA base image is the `*-runtime-*` flavour (no nvcc) and
  torch.compile / inductor wants it. Either switch to a `*-devel-*`
  base or set `MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER=true`.

* **`ValueError: Free memory on device cuda:0 (X / Y GiB) on startup is less than desired GPU memory utilization`**
  vLLM is trying to grab `gpu_memory_utilization * total_VRAM`. Lower
  `MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL` (default `0.25` already
  accounts for sharing the GPU with ColPali).

* **vLLM engine bootstrap hangs / fails with shared-memory errors**
  Add `ipc: host` (compose) or `--ipc=host` (docker run). vLLM uses
  POSIX shared memory between its engine and worker processes; without
  host IPC the default container `/dev/shm` is too small.

* **Single-vector responses look identical to test inputs**
  You may be in mock mode without realising it — check
  `MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS`. Mocks are seeded from a
  SHA-256 of the input, so the same input always yields the same vector.

* **ColPali requests return 503 with `Retry-After: 1`**
  GPU queue saturated. The error body's `wait_seconds` tells you how
  long the request waited; `max_concurrent` is the current cap. Either
  bump `MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT`, scale the
  embed-server replicas, or back off at the client.

* **"ColPali GPU slot acquired after Xs wait" WARN log entries**
  Healthy backpressure signal — requests are queueing because the cap
  is reached. Track frequency and average wait time as proxy metrics for
  "do we need more embed-servers". (A proper metrics integration is a
  documented future task.)

---

## LLM Inference Profiles

The embed-server can optionally host **chat / completions LLM workloads**
in addition to embeddings. When enabled, one or more `vllm serve` child
processes run alongside the embedding stack, each serving a distinct
LLM, and the FastAPI app exposes OpenAI-compatible chat / completions /
models endpoints. Routing is by the OpenAI-standard `model` request
field — clients written for the OpenAI API "just work."

This is a gated, additive feature: existing deployments are
byte-for-byte unaffected unless they set
`MEMORYLAYER_EMBED_LLM_ENABLED=true`.

### When to use it

Co-locate a small chat LLM with embeddings on the same GPU pod when:

* You want one process and one network endpoint for both retrieval and
  generation (e.g. a single-box on-prem deployment).
* You want vLLM's continuous batching + paged attention for a small
  model that doesn't justify its own dedicated pod.
* You want the same Aether mTLS + OBO posture for both embedding and
  chat traffic.

If you already run a dedicated inference cluster (Triton, sglang,
hosted OpenAI, etc.), the core server can point at it directly via
`MEMORYLAYER_LLM_PROFILE_<NAME>_BASE_URL` — no need to use the
embed-server's LLM hosting.

### Selection / enabling

```
MEMORYLAYER_EMBED_LLM_ENABLED=true                 # gate; default false
MEMORYLAYER_EMBED_LLM_PROFILES=qwen,llama          # comma list of profile names
MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE=qwen         # used when request `model` doesn't match any profile/alias
MEMORYLAYER_EMBED_LLM_PRELOAD=false                # eagerly spawn at boot (default: lazy)
MEMORYLAYER_EMBED_LLM_PORT_RANGE=18100-18199       # private port pool for internal vllm subprocesses
```

Internal vLLM ports are **auto-assigned** from the configured port
range — operators never set ports manually. External callers only ever
see the embed-server's public port (or its Aether topic).

### Per-profile env vars

For each declared profile NAME (uppercased in the env var):

```
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_MODEL=Qwen/Qwen2.5-7B-Instruct
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_ALIASES=qwen-2.5,qwen-7b         # comma list
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_DTYPE=auto                       # vLLM dtype
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_MAX_MODEL_LEN=8192               # optional; omit for model default
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_GPU_MEM_UTIL=0.3                 # 0..1
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_ENFORCE_EAGER=false              # disable torch.compile / inductor
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_TENSOR_PARALLEL_SIZE=1           # multi-GPU TP on a single node
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_STARTUP_TIMEOUT_SEC=600          # cold-load deadline
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_EXTRA_ARGS=--quantization fp8    # shell-split free-form
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_CMD=vllm                         # binary on PATH; override for wrappers
MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_HOST=127.0.0.1                   # loopback by default
```

The profile name itself is added as an OpenAI `--served-model-name`
alongside the underlying model name and any explicit aliases. So a
profile named `qwen` with `MODEL=Qwen/Qwen2.5-7B-Instruct` and
`ALIASES=qwen-7b` will answer to all four strings (`qwen`,
`qwen-7b`, `Qwen/Qwen2.5-7B-Instruct`, and `qwen2.5` if listed).

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions. Supports `stream=true` (SSE pass-through). |
| `POST /v1/completions` | Legacy text completions. Same routing rules. |
| `GET /v1/models` | OpenAI-compatible model list: every profile name + alias + underlying model. |

All payloads are forwarded **verbatim** to the underlying vLLM — tool
calls (`tools`, `tool_choice`), structured output (`response_format`),
multimodal (`messages[].content[].type=image_url`), reasoning
(`reasoning_effort`), and any other vLLM-supported field passes
through unchanged. The proxy uses raw `httpx`, not the OpenAI SDK, so
nothing is dropped due to schema gaps.

### Routing precedence

When a request arrives at `/v1/chat/completions`, the router resolves
the incoming `model` field against the alias map (case-insensitive):

1. Exact match against any profile name, alias, or underlying model.
2. Fall back to `MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE` if set.
3. Otherwise → `404` with a `model_not_found` error including the
   list of available routing names.

### Health + load reporting

`GET /health/ready` includes `services.llm` showing profile count
and the resolved default profile.

`GET /health/load` merges per-profile snapshots under
`providers.llm_<profile_name>`:

```json
{
  "providers": {
    "colpali_multi_vector": {"in_flight": 0, "max_concurrent": 4, "utilization": 0.0},
    "llm_qwen":  {"in_flight": 2, "max_concurrent": 0, "utilization": 0.0},
    "llm_llama": {"in_flight": 0, "max_concurrent": 0, "utilization": 0.0}
  },
  "utilization": 0.0
}
```

vLLM owns scheduling internally; `max_concurrent: 0` means "no LB-side
cap" and operators routing on this metric should treat `0` as
unlimited. `in_flight` is the number of requests the embed-server has
forwarded to vLLM that haven't returned yet.

### Lifecycle

* **Subprocess start**: lazy by default (on first request). Set
  `MEMORYLAYER_EMBED_LLM_PRELOAD=true` to warm all profiles at boot.
* **Shutdown**: every profile's `vllm serve` child gets SIGTERM via
  its process group, then SIGKILL after 10s. Triggered before the
  framework's plugin teardown so GPU memory is released cleanly.
* **Health-check**: each profile polls `http://127.0.0.1:<auto-port>/health`
  with the configured `STARTUP_TIMEOUT_SEC` deadline.

### Recipes

#### Single small LLM co-located with ColPali embeddings

```yaml
environment:
  # Embeddings (unchanged)
  MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER: vllm_subprocess

  # LLM (new)
  MEMORYLAYER_EMBED_LLM_ENABLED: "true"
  MEMORYLAYER_EMBED_LLM_PROFILES: tiny
  MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE: tiny
  MEMORYLAYER_EMBED_LLM_PROFILE_TINY_MODEL: Qwen/Qwen2.5-0.5B-Instruct
  MEMORYLAYER_EMBED_LLM_PROFILE_TINY_GPU_MEM_UTIL: "0.15"
```

#### Two LLMs (small + medium) sharing one GPU

```yaml
environment:
  MEMORYLAYER_EMBED_LLM_ENABLED: "true"
  MEMORYLAYER_EMBED_LLM_PROFILES: tiny,medium
  MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE: tiny

  MEMORYLAYER_EMBED_LLM_PROFILE_TINY_MODEL: Qwen/Qwen2.5-0.5B-Instruct
  MEMORYLAYER_EMBED_LLM_PROFILE_TINY_GPU_MEM_UTIL: "0.15"

  MEMORYLAYER_EMBED_LLM_PROFILE_MEDIUM_MODEL: Qwen/Qwen2.5-7B-Instruct
  MEMORYLAYER_EMBED_LLM_PROFILE_MEDIUM_ALIASES: qwen-7b,qwen
  MEMORYLAYER_EMBED_LLM_PROFILE_MEDIUM_GPU_MEM_UTIL: "0.5"
  MEMORYLAYER_EMBED_LLM_PROFILE_MEDIUM_MAX_MODEL_LEN: "8192"
```

A client sending `model="qwen"` reaches the medium profile (via the
`qwen` alias). `model="tiny"` reaches the small profile. Unknown
models fall back to `tiny` via the configured default.

#### Tool-capable model + reasoning model

```yaml
environment:
  MEMORYLAYER_EMBED_LLM_ENABLED: "true"
  MEMORYLAYER_EMBED_LLM_PROFILES: tools,reason

  MEMORYLAYER_EMBED_LLM_PROFILE_TOOLS_MODEL: NousResearch/Hermes-3-Llama-3.1-8B
  MEMORYLAYER_EMBED_LLM_PROFILE_TOOLS_GPU_MEM_UTIL: "0.4"
  MEMORYLAYER_EMBED_LLM_PROFILE_TOOLS_EXTRA_ARGS: "--tool-call-parser hermes --enable-auto-tool-choice"

  MEMORYLAYER_EMBED_LLM_PROFILE_REASON_MODEL: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  MEMORYLAYER_EMBED_LLM_PROFILE_REASON_GPU_MEM_UTIL: "0.4"
```

### Wiring it from the core server

The core server's existing LLM profile machinery supports an explicit
`embed_server` provider type:

```
MEMORYLAYER_LLM_PROFILE_INFERENCE_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_INFERENCE_MODEL=qwen           # matches a profile/alias on the embed-server
MEMORYLAYER_LLM_PROFILE_INFERENCE_EMBED_SERVER_URL=http://embed-1:61051   # optional override
MEMORYLAYER_LLM_PROFILE_INFERENCE_EMBED_SERVER_TRANSPORT=http             # 'http' or 'aether'
MEMORYLAYER_LLM_ASSIGN_REFLECT=inference
```

Each LLM profile is independent — different profiles can point at
different embed-server peers (or one cloud + one self-hosted). See
the core server's [LLM profiles
docs](../../memorylayer-core-python/docs/llm-profiles.md) for the full
recipe set.

### Limitations / out of scope

1. No per-profile bearer-token auth on `/v1/chat/completions` yet —
   the embed-server's security model assumes a trusted peer (or Aether
   mTLS).
2. Multi-host tensor parallelism is out of scope. Use single-host TP
   via `TENSOR_PARALLEL_SIZE`; for multi-host setups, run a dedicated
   inference cluster and use the core OpenAI provider with `BASE_URL`.
3. Streaming over Aether uses
   `proxy_http_async(stream_response=True)`. Older builds of
   `scitrera-aether-client` that don't expose that flag will raise at
   the first streaming chat call — upgrade the SDK or fall back to
   HTTP transport for the LLM path.
