# LLM Profiles

`memorylayer-server` routes chat / completion traffic through a
**profile-based** LLM registry. Each profile names one provider
instance (with its own model, base URL, transport, etc.), and each
internal activity (memory extraction, reflection, semantic tiering,
etc.) is assigned to one of those profiles via
`MEMORYLAYER_LLM_ASSIGN_<ACTIVITY>=<profile_name>`.

This doc covers the env-var schema and a few recipes for talking to
self-hosted models via a `memorylayer-embed-server` peer.

## Env-var schema

For each profile NAME (uppercased in the env var):

```
MEMORYLAYER_LLM_PROFILE_<NAME>_PROVIDER=openai|anthropic|google|embed_server|noop
MEMORYLAYER_LLM_PROFILE_<NAME>_MODEL=<model-id>
MEMORYLAYER_LLM_PROFILE_<NAME>_API_KEY=<vendor-key>            # optional (some providers)
MEMORYLAYER_LLM_PROFILE_<NAME>_BASE_URL=<https://...>         # OpenAI-compatible only
MEMORYLAYER_LLM_PROFILE_<NAME>_MAX_TOKENS=4096                 # default cap; per-request override wins
MEMORYLAYER_LLM_PROFILE_<NAME>_TEMPERATURE=0.7                 # default; per-request override wins
```

For `provider=embed_server` (talks to a `memorylayer-embed-server`
peer that hosts one or more LLM profiles internally):

```
MEMORYLAYER_LLM_PROFILE_<NAME>_EMBED_SERVER_URL=http://embed-peer:61051       # optional override
MEMORYLAYER_LLM_PROFILE_<NAME>_EMBED_SERVER_TRANSPORT=http                    # 'http' or 'aether'
MEMORYLAYER_LLM_PROFILE_<NAME>_EMBED_SERVER_AETHER_TARGET=sv::memorylayer-embed::us-west
MEMORYLAYER_LLM_PROFILE_<NAME>_EMBED_SERVER_TIMEOUT=600
```

When **all four** of the embed-server-specific overrides are omitted,
the provider piggybacks on the singleton `EmbedServerClient` already
wired for embeddings (same `MEMORYLAYER_EMBED_SERVER_URL`,
`MEMORYLAYER_EMBED_TRANSPORT`, etc.). Setting any one of them forces
construction of a **dedicated** client just for this profile — so
multiple LLM profiles can point at multiple different embed-server
peers concurrently, independent of the embedding-side client.

## Activity-to-profile routing

```
MEMORYLAYER_LLM_ASSIGN_REFLECT=<profile_name>
MEMORYLAYER_LLM_ASSIGN_TIER_GENERATION=<profile_name>
MEMORYLAYER_LLM_ASSIGN_ONTOLOGY=<profile_name>
```

When an activity isn't explicitly assigned, the registry falls back
to the profile named `default`. If no `default` profile exists, the
registry installs a `NoOpLLMProvider` that raises
`LLMNotConfiguredError` on any call.

## Provider matrix

| Provider | Tools / function calling | Structured output | Reasoning |
|---|---|---|---|
| `openai` | ✓ (canonical OpenAI shape) | `response_format` forwarded | `reasoning_effort` (o-series) |
| `anthropic` | ✓ (auto-translated from canonical) | n/a | `reasoning_effort` → `thinking` budget |
| `google` | ✓ (caller supplies Google-shape) | `response_format` → `response_mime_type` | `reasoning_effort` → `thinking_config` |
| `embed_server` | ✓ (passthrough) | passthrough | passthrough |
| `noop` | — | — | — |

Tools / structured output / reasoning fields all live on
`LLMRequest` (`tools`, `tool_choice`, `response_format`,
`reasoning_effort`, `max_completion_tokens`, `extra_body`); see
`src/memorylayer_server/models/llm.py` for the full schema.

## Recipes

### 1. Single embed-server peer hosting a chat LLM

The embed-server runs both ColPali (for visual embeddings) and a
Qwen-7B chat profile on the same GPU box. The core server uses one
LLM profile pointing at it.

**Embed-server side** (see embed-server docs for full env reference):

```
MEMORYLAYER_EMBED_LLM_ENABLED=true
MEMORYLAYER_EMBED_LLM_PROFILES=qwen
MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_GPU_MEM_UTIL=0.4
```

**Core side**:

```
MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL=qwen          # matches the embed-server profile name
# No URL override — piggyback on the existing MEMORYLAYER_EMBED_SERVER_URL
# already configured for embeddings.
```

### 2. Two embed-server peers, one profile each

Embed-server "A" runs a small fast model for tier generation; embed-server
"B" runs a stronger model for reflection synthesis. The two peers are
independent and may live in different regions.

```
# Fast tier-generation profile → peer A
MEMORYLAYER_LLM_PROFILE_TIER_GEN_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_TIER_GEN_MODEL=tiny
MEMORYLAYER_LLM_PROFILE_TIER_GEN_EMBED_SERVER_URL=http://embed-a:61051

# Strong reflection profile → peer B
MEMORYLAYER_LLM_PROFILE_REFLECT_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_REFLECT_MODEL=qwen-7b
MEMORYLAYER_LLM_PROFILE_REFLECT_EMBED_SERVER_URL=http://embed-b:61051

# Route activities
MEMORYLAYER_LLM_ASSIGN_TIER_GENERATION=tier_gen
MEMORYLAYER_LLM_ASSIGN_REFLECT=reflect
```

### 3. Cross-region embed-server via Aether mTLS

The chat LLM lives in an isolated GPU network reachable only via the
Aether mesh. Encrypted transport with workload identity + OBO:

```
MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL=qwen-7b
MEMORYLAYER_LLM_PROFILE_DEFAULT_EMBED_SERVER_TRANSPORT=aether
MEMORYLAYER_LLM_PROFILE_DEFAULT_EMBED_SERVER_AETHER_TARGET=sv::memorylayer-embed::us-west
MEMORYLAYER_LLM_PROFILE_DEFAULT_EMBED_SERVER_TIMEOUT=120
```

Both non-streaming and streaming chat completions flow through
`proxy_http_async` (the SDK uses `stream_response=True` under the
hood for streaming requests).

### 4. Mixed cloud + self-hosted

Cheap activities go to OpenAI; sensitive activities stay on a
self-hosted embed-server peer.

```
MEMORYLAYER_LLM_PROFILE_CLOUD_PROVIDER=openai
MEMORYLAYER_LLM_PROFILE_CLOUD_MODEL=gpt-5-nano
MEMORYLAYER_LLM_PROFILE_CLOUD_API_KEY=sk-...

MEMORYLAYER_LLM_PROFILE_PRIVATE_PROVIDER=embed_server
MEMORYLAYER_LLM_PROFILE_PRIVATE_MODEL=qwen-7b
MEMORYLAYER_LLM_PROFILE_PRIVATE_EMBED_SERVER_URL=http://embed-private:61051

MEMORYLAYER_LLM_ASSIGN_TIER_GENERATION=cloud
MEMORYLAYER_LLM_ASSIGN_REFLECT=private
MEMORYLAYER_LLM_ASSIGN_ONTOLOGY=private
```

## Streaming

All four real providers (`openai`, `anthropic`, `google`,
`embed_server`) support streaming via
`LLMProviderRegistry.complete_stream()`. The
`embed_server` provider parses upstream Server-Sent Events and emits
`LLMStreamChunk(content=..., is_final=False)` text deltas, with
separate chunks for incremental `tool_calls_delta` and
`reasoning_content_delta` updates, plus a terminal `is_final=True`
chunk carrying `finish_reason`.

## Tool calling end-to-end

`LLMRequest.tools` accepts the canonical OpenAI shape:

```python
LLMRequest(
    messages=[LLMMessage(role=LLMRole.USER, content="weather in Boston?")],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
    tool_choice="auto",
)
```

The `openai` and `embed_server` providers forward this payload as-is.
The `anthropic` provider auto-translates to Anthropic's
`tools` + content-block schema. The `google` provider passes the
list through to `GenerateContentConfig.tools` — the caller is
responsible for providing Google-native `Tool` objects in that case
(or merging them via `extra_body`).

`LLMResponse.tool_calls` is always in canonical OpenAI shape on the
return path:

```python
[
    {
        "id": "call_abc",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}',
        },
    },
]
```

On the next turn, append the tool result as an
`LLMMessage(role=LLMRole.TOOL, content="...", tool_call_id="call_abc")`
and call `complete()` again.

## Reasoning content

`LLMRequest.reasoning_effort` accepts `"minimal"`, `"low"`,
`"medium"`, or `"high"`. Each provider maps it to its native budget:

| Provider | Mapping |
|---|---|
| `openai` | forwarded verbatim (o-series only) |
| `anthropic` | `thinking={"type": "enabled", "budget_tokens": …}` |
| `google` | `GenerateContentConfig(thinking_config=…)` |
| `embed_server` | forwarded as-is for vLLM to interpret |

The model's reasoning summary (when emitted) lands in
`LLMResponse.reasoning_content`. Streaming responses surface
incremental updates via `LLMStreamChunk.reasoning_content_delta`.

## See also

* [`memorylayer-embed-server` LLM hosting docs](../../memorylayer-embed-server/docs/embedding-providers.md#llm-inference-profiles)
* `src/memorylayer_server/models/llm.py` — full `LLMRequest` /
  `LLMResponse` / `LLMStreamChunk` schemas
* `src/memorylayer_server/services/llm/embed_server.py` — the
  `EmbedServerLLMProvider` implementation
