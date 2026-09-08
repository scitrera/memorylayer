# Evaluation: driving the vLLM lanes with sparkrun instead of our own subprocess runner

**Status:** Spike complete. Feasibility confirmed; not implemented.
**Date:** 2026-08-04
**Sparkrun:** 0.3.2 (local checkout; 0.3.1 on PyPI) at `scitrera-dgx-spark-commander/oss-sparkrun`

## Question

`services/_vllm_runner.py` (475 lines) spawns `vllm serve` as a child process, polls
its health endpoint, and tears it down. Sparkrun already does that, is already used to
run these exact models, and **the recipes already live in this repo** under
`sparkrun_recipes/`. Should the embed server use sparkrun as a library — with the
`local` (no-container) executor — rather than maintaining its own launcher?

## Spike results

Run on a host with **no GPU**, so a real launch was impossible. `RunOptions.dry_run`
plus `RunResult.serve_command` render the full command without executing it, which is
enough to settle the wiring questions. Script: `.slop/sparkrun_spike.py`.

| | question | result |
|---|---|---|
| Q1 | Load a recipe shipped with our code, no registry/cluster config? | **PASS** — `Recipe.load(path)` works standalone |
| Q2 | Does the rendered command match what `_vllm_runner` builds? | **PASS** — same model, `--runner pooling`, `--convert embed` |
| Q3 | Does a per-launch port override reach the command? | **PASS** — also `host`, `gpu_memory_utilization` |
| Q4 | Idempotent + attributable launch? | **PASS** — `ensure=True`, `owner=` accepted |

Rendered from `sparkrun_recipes/sv_qwen3vl_2b_v1.yaml` with overrides applied:

```
vllm serve Qwen/Qwen3-VL-Embedding-2B --host 127.0.0.1 --port 18000 --dtype auto
  -tp 1 -pp 1 --max-model-len 32768 --max-num-batched-tokens 8192
  --trust-remote-code --gpu-memory-utilization 0.25 --kv-cache-dtype auto
  --load-format instanttensor --runner pooling --convert embed
```

That is the command our runner builds today, from a file we already maintain.

## Why the fit is good

- **The recipes exist and already match.** `sv_qwen3vl_2b_v1` (single-vector),
  `mv_vbert_v1` (multi-vector), `transcribe_unlimited-ocr` (transcription) cover all
  three of our vLLM lanes. Today the same settings are duplicated as
  `MEMORYLAYER_EMBEDDING_VLLM_*` env vars.
- **The API is built for embedding.** Its own docstring: *"The API never writes to
  stdout/stderr and never calls sys.exit."* Typed errors, stable dataclasses.
- **`RunOptions.owner`'s docstring describes this exact use case** — an automated
  supervisor distinguishing its own jobs from a human's, with `llm-gateway` as the
  example. `ensure=True` gives idempotent relaunch, which a restarting server wants.
- **No Docker-in-Docker.** The `local` executor `setsid`-launches natively, writes a
  pidfile, redirects output to a logfile. `should_run_locally("localhost")` is true
  with no SSH user, so no ssh hop either.
- **Dependency overlap is clean.** Sparkrun pins `scitrera-app-framework==0.0.69` —
  exactly what this monorepo already pins. Marginal additions: `textual`, `vpd`,
  `platformdirs`, `linkify-it-py`, `mdit-py-plugins`, `uc-micro-py`.

## What does NOT go away

`_vllm_runner.py` is not 475 lines of generic process management. Sparkrun replaces
the spawn / health-poll / teardown core. These stay ours:

- **Concurrency sizing.** We parse vLLM's `Maximum concurrency for N tokens per
  request: X` log line to size the client-side semaphore, so backpressure lands at our
  HTTP boundary rather than inside vLLM's scheduler queue. Nothing in sparkrun does
  this.
- **Dynamic port assignment.** `find_free_port` gives each LLM profile a loopback port
  at boot. Recipes carry a static `port:`, so it becomes an override — meaning the
  recipe's port is advisory rather than true.
- The provider classes themselves (OpenAI-compat HTTP clients, batching, dimension
  handling) are untouched either way.

## Risks, characterized

**1. The `local` executor is explicitly "Experimental."** Its own module docstring says
so. It is the one piece of sparkrun we would depend on, and the least exercised — the
container path is the well-trodden one.

**2. Reading the concurrency line changes mechanism.** Today we read a stderr pipe we
own. Under sparkrun the process writes to a logfile and `api.logs(...)` tails it with
`tail -F`, yielding structured `LogLine` records. So the line *is* reachable — but via
an extra `tail` subprocess, through a **blocking iterator** that an async FastAPI server
must drive on a thread. Whether the line reliably appears within the startup window is
the one thing the spike could not test without a GPU.

**3. Teardown semantics genuinely change.** `api.stop()` uses a process-group kill
(`kill -- -<pgid>`), which is clean. But `setsid` deliberately detaches the child, so it
**survives the parent's death**:
- *In a container* this is moot — PID-namespace teardown kills it with the container.
- *On bare metal / dev*, a crashed embed server leaves vLLM holding VRAM. With
  `ensure=True` the restart reuses it instead of reloading, which is arguably better —
  but only if the config is unchanged. It is a real behaviour change either way.

**4. The exact framework pin cuts both ways.** `scitrera-app-framework==0.0.69` is
pinned, not floored. The day this monorepo bumps the framework, sparkrun blocks it
until sparkrun releases too.

**5. The recipes are container-shaped.** Each carries a `container:` field that the
local executor ignores, and `sv_qwen3vl_2b_v1` binds `--host 0.0.0.0` by default — wrong
for an internal child, though overridable (verified). Reused locally, the recipe files
stop being fully truthful about what runs.

**6. It must be an optional extra.** This is a GPU-path dependency; the CPU image must
not grow `textual` and friends. `memorylayer-embed-server[sparkrun]`.

## Recommendation

**Proceed, on one lane, behind a provider flag.** Add
`MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_sparkrun` alongside the existing
`vllm_subprocess`, keep `vllm_subprocess` as the default, and measure on a GPU host:

1. Cold-start time vs the current runner.
2. Whether the concurrency line arrives in time via `api.logs(follow=True)` — the
   likeliest thing to break, and the reason to do single-vector first.
3. Teardown when the parent is SIGKILLed, on bare metal and in a container.

If it holds, extend to multi-vector and transcription and make it the default for the
`-cuda13` image. If it does not, the loss is one provider file rather than the launch
path.

The prize is real: three lanes' worth of tuning currently lives as env-var soup that
duplicates recipes you already maintain, and the recipe form is both more expressive
and shared with the rest of the stack.
