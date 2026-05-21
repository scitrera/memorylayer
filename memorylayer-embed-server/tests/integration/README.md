# Embed-server integration test (docker chain)

End-to-end test that exercises the chain:

```
pytest driver ──► memorylayer-server (61101) ──► memorylayer-embed-server
                                    private docker network
```

Two variants are supported.

---

## Fast variant — mock providers (default; CI-friendly)

Uses ``Dockerfile.test`` for the embed-server, which runs with
``MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS=true`` (deterministic numpy
providers, no torch, no GPU, no model downloads). Build is seconds,
runtime is seconds.

```bash
# From repo root
pytest -m integration oss/memorylayer-embed-server/tests/integration/
```

The pytest fixture takes care of ``docker compose up --build -d`` and
``docker compose down -v``. Set ``KEEP_STACK=1`` to leave the stack
running after the test for inspection:

```bash
KEEP_STACK=1 pytest -m integration -k test_memorylayer_server_health \
    oss/memorylayer-embed-server/tests/integration/
docker compose -f oss/memorylayer-embed-server/tests/integration/docker-compose.embed-chain.yml ps
```

Compose surface:

| service             | host port | network port | image                          |
|---------------------|-----------|--------------|--------------------------------|
| `memorylayer-server`| 61101     | 61001        | built from local `oss/`        |
| `embed-server`      | —         | 61051        | `memorylayer-embed-server:test`|

---

## Variant — real ColPali + OpenAI-compat single (no vLLM)

Useful when the operator wants self-hosted multi-vector retrieval
(ColPali) but the lighter HTTP-out path for single-vector embeddings —
no vLLM dependency, no in-process LLM. The embed-server's
``EMBED_SERVER_SINGLE_VECTOR_PROVIDER`` selects between ``vllm``,
``openai``, ``google``, ``colpali`` (shared multi-vector provider), and
``mock``.

```bash
OPENAI_API_KEY=sk-... \
COMPOSE_FILE_OVERRIDE=docker-compose.embed-chain.real-openai-single.yml \
    pytest -m integration oss/memorylayer-embed-server/tests/integration/
```

Or point at an OpenAI-compatible endpoint (a sibling vLLM server,
LocalAI, Ollama, …) for fully self-hosted single-vector:

```bash
MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
MEMORYLAYER_EMBEDDING_OPENAI_API_KEY=x \
MEMORYLAYER_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 \
MEMORYLAYER_EMBEDDING_DIMENSIONS=1024 \
COMPOSE_FILE_OVERRIDE=docker-compose.embed-chain.real-openai-single.yml \
    pytest -m integration oss/memorylayer-embed-server/tests/integration/
```

---

## Heavy variant — real ColPali + Qwen3-VL (manual)

For validating model behavior end-to-end (not run in CI). Reuses the
production ``oss/memorylayer-embed-server/Dockerfile`` (CUDA + torch +
colpali-engine + vLLM) and a tiny memory-light ColPali model.

1. Build the production embed-server image (this can take a while and
   needs a working sibling checkout of ``scitrera-aether3-go`` staged
   at ``proprietary/.build-staging/`` — see the production Dockerfile
   for the exact requirements):

   ```bash
   docker build -f oss/memorylayer-embed-server/Dockerfile \
                -t memorylayer-embed-server:real .
   ```

2. Override the compose file's `embed-server.image` and disable the
   build context override:

   ```yaml
   # docker-compose.embed-chain.override.yml
   services:
     embed-server:
       image: memorylayer-embed-server:real
       build: null   # disable the test-image build
       environment:
         EMBED_SERVER_RUN_SIDECAR: "false"   # still skip the Aether sidecar for the chain test
         MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS: "false"
         MEMORYLAYER_EMBEDDING_MODEL: "ModernVBERT/colmodernvbert"  # smallest ColPali
         MEMORYLAYER_EMBED_PRELOAD_MODELS: "false"                  # skip warm-up on first request
   ```

3. Run the same pytest tests against the heavy stack:

   ```bash
   docker compose \
       -f oss/memorylayer-embed-server/tests/integration/docker-compose.embed-chain.yml \
       -f docker-compose.embed-chain.override.yml \
       up --build -d
   KEEP_STACK=1 pytest -m integration -k test_create_memory \
       oss/memorylayer-embed-server/tests/integration/
   docker compose ... down -v
   ```

Expect the first request to be slow (model download + warm-up). The
mock-provider variant remains the right tool for CI.

---

## Variant — real-GPU subprocess (no docker)

For a faster iteration loop than the docker chain, the
``test_real_gpu_embeddings.py`` and ``test_real_gpu_transcription.py``
modules spawn ``memorylayer-embed`` as a subprocess on a free loopback
port and drive it directly over HTTP. No docker, no second
memorylayer-server container — just the embed-server with real models
on the host GPU.

These tests require the embed-server's own venv with the ``[colpali]``
and ``[ocr]`` extras installed:

```bash
cd oss/memorylayer-embed-server
uv venv
uv pip install --python .venv/bin/python --torch-backend=auto \
    -e ../memorylayer-core-python \
    -e ".[colpali,ocr,dev,observability]"
```

Then opt into the slow-integration markers:

```bash
# ColPali via in-process colpali-engine (ModernVBERT/colmodernvbert LoRA adapter, ~250MB)
.venv/bin/pytest -m "slow and integration" \
    tests/integration/test_real_gpu_embeddings.py -v

# ColPali via vLLM /pooling subprocess (ModernVBERT/colmodernvbert-merged, ~1GB).
# Exercises the production multi-vector path: EMBED_SERVER_MULTI_VECTOR_PROVIDER=vllm_subprocess.
# Cold-start downloads model weights and warms vLLM — expect 1–3 minutes.
.venv/bin/pytest -m "slow and integration" \
    tests/integration/test_real_gpu_vllm_multivector.py -v

# Transcription cascade (GLM-OCR — multi-GB download on cold cache)
.venv/bin/pytest -m "slow and integration" \
    tests/integration/test_real_gpu_transcription.py -v
```

Useful environment overrides (read by the test modules):

| Variable                                      | Purpose                                                 |
|-----------------------------------------------|---------------------------------------------------------|
| ``MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER``   | ``vllm_subprocess`` (default) or ``colpali_inprocess``  |
| ``MEMORYLAYER_EMBEDDING_COLPALI_MODEL``       | Override the ColPali model under test                   |
| ``MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR`` | Hierarchical token-pool factor (1=off, paper recommends 2 or 3) |
| ``MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES`` | Comma-separated arch override for the vLLM provider (e.g. ``ColModernVBertForRetrieval``) |
| ``MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL``   | Per-vLLM-subprocess GPU memory budget (default 0.25)    |
| ``MEMORYLAYER_EMBEDDING_DEVICE``              | Force ``cuda`` / ``cpu`` (in-process colpali path only) |
| ``MEMORYLAYER_EMBED_GLM_OCR_MODEL``           | Substitute a different OCR model in the transcription cascade |
| ``MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS``      | Cap OCR output length (test default: 512)               |
| ``MEMORYLAYER_EMBED_TEST_REQUEST_TIMEOUT``    | Per-request HTTP timeout in seconds (default 600/900)   |
| ``MEMORYLAYER_EMBED_TEST_BOOT_TIMEOUT``       | embed-server boot timeout in seconds (default 600 for vLLM tests) |
| ``MEMORYLAYER_EMBED_TEST_DISCARD_LOGS``       | ``1`` cleans up the subprocess log files on teardown. By default logs are kept under ``$TMPDIR`` for post-mortem. |

Each module uses a ``module``-scoped fixture that boots one subprocess
for the file's tests and tears it down at the end. The two suites are
intentionally split so the embedding subprocess and transcription
subprocess do not compete for GPU memory.

---

## Troubleshooting

* **Compose build fails on aether staging**: the `Dockerfile.test`
  variant does NOT need ``proprietary/.build-staging/`` — only the heavy
  production Dockerfile does. If you see ``COPY proprietary/.build-staging/...``
  fail you're building the wrong file.
* **Port 61101 in use**: edit the host port in the compose file or stop
  whatever else is listening on it.
* **Tests time out waiting for `/health`**: run
  ``docker compose -f ... logs memorylayer-server embed-server`` to see
  why one of the containers crashed at boot.
