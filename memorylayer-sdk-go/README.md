# MemoryLayer Go SDK

Go client for [MemoryLayer.ai](https://memorylayer.ai) — memory infrastructure for LLM-powered agents.

It mirrors the [Python](../memorylayer-sdk-python) and [TypeScript](../memorylayer-sdk-typescript) SDKs: one `Client` exposing memory CRUD, semantic recall/reflect, the association graph, sessions, workspaces, chat threads, the context sandbox, the entity registry, API tokens, and the Enterprise document/dataset surfaces — plus the `Skills`, `McpServers`, `Kb`, and `Rpg` namespaces, and on-behalf-of (OBO) authority.

```
import memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
```

## Install

```bash
go get github.com/scitrera/memorylayer/memorylayer-sdk-go
```

The core module is **dependency-light** — it uses only the Go standard library (`net/http`, `encoding/json`). The Aether transport lives in a separate module so HTTP-only callers don't pull in the Aether SDK's gRPC dependency tree (see [Aether transport](#aether-transport)).

## Quick start

```go
package main

import (
	"context"
	"log"

	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
)

func main() {
	client, err := memorylayer.NewClient(
		memorylayer.WithBaseURL("https://api.memorylayer.ai"),
		memorylayer.WithAPIKey("your-api-key"),
		memorylayer.WithWorkspaceID("ws_123"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()

	mem, err := client.Remember(ctx, "User prefers Go for backend services.", memorylayer.RememberOptions{
		Type:       memorylayer.MemoryTypeSemantic,
		Subtype:    string(memorylayer.MemorySubtypePreference),
		Importance: 0.8,
		Tags:       []string{"preferences"},
	})

	results, err := client.Recall(ctx, "what language does the user like?", memorylayer.RecallOptions{
		Limit:               5,
		IncludeAssociations: memorylayer.Ptr(true),
	})

	reflection, err := client.Reflect(ctx, "summarize the user's preferences", memorylayer.ReflectOptions{MaxTokens: 200})

	_ = mem
	_ = results
	_ = reflection
}
```

A runnable example: [`examples/basic`](examples/basic).

## API shape

Each method takes a `context.Context` first, required positional arguments next, and an `…Options` struct last for optional parameters (the zero value is always valid — only non-empty fields are sent, so the server applies its own defaults). Pointer-typed option fields (`*bool`, `*int`, `*float64`) distinguish "unset" from a zero value; use `memorylayer.Ptr(v)` to set them:

```go
client.Recall(ctx, "q", memorylayer.RecallOptions{
	Limit:             10,
	MinRelevance:      memorylayer.Ptr(0.7),
	IncludeGlobal:     memorylayer.Ptr(true),
	BudgetTokens:      memorylayer.Ptr(600),
	IncludeConfidence: memorylayer.Ptr(true),
	IncludeRelations:  memorylayer.Ptr(true),
})
```

Compaction-safe session recovery uses `CreateCheckpoint`, `GetCheckpoint`,
`GetContextPack`, and `GetContextDelta`. Checkpoints require a SHA-256 content
hash and session-scoped idempotency key; packs and deltas enforce their requested
token budget and do not synthesize prose.

### Namespaces

```go
client.Skills.List(ctx, memorylayer.ListSkillsOptions{})
client.McpServers.Resolve(ctx, "github", "", nil)
client.Kb.Generate(ctx, memorylayer.KbGenerateOptions{Regenerate: true})
client.Rpg.GetSubgraph(ctx, memorylayer.RpgSubgraphOptions{Path: "src/main.go"})
```

## Errors

Failures return typed errors that embed `*APIError` (carrying `Message` and `StatusCode`). Match them with `errors.As`, or use the helpers:

```go
results, err := client.Recall(ctx, "q", memorylayer.RecallOptions{})
if err != nil {
	switch {
	case memorylayer.IsNotFound(err):
		// 404
	case memorylayer.IsEnterpriseRequired(err):
		// Enterprise-only endpoint on an OSS server
	default:
		var apiErr *memorylayer.APIError
		if errors.As(err, &apiErr) {
			log.Printf("status %d: %s", apiErr.StatusCode, apiErr.Message)
		}
	}
}
```

Error types: `AuthenticationError` (401), `AuthorizationError` (403), `NotFoundError` (404), `ValidationError` (422), `RateLimitError` (429), `ServerError` (5xx), `EnterpriseRequiredError` (501 on an Enterprise feature). Idempotent requests (GET/PUT/DELETE/PATCH) are retried with backoff on transient failures, honoring `Retry-After`; POST is never auto-retried.

## On-behalf-of (OBO) authority

Bind an authority and/or workspace to a lightweight proxy without mutating the shared client:

```go
alice := client.ActingFor("g_abc", &memorylayer.PrincipalRef{Type: "user", ID: "alice"})
alice.Recall(ctx, "preferences", memorylayer.RecallOptions{WorkspaceID: "ws_work"})

ws := client.ForWorkspace("ws_other")
ws.Remember(ctx, "scoped note", memorylayer.RememberOptions{})
```

The proxy also exposes the namespaces with the same authority + workspace bound, so they flow through to every namespace call:

```go
alice.Skills().List(ctx, memorylayer.ListSkillsOptions{})
alice.McpServers().Resolve(ctx, "github", "", nil)
alice.Kb().Generate(ctx, memorylayer.KbGenerateOptions{Regenerate: true})
```

Any individual method also accepts a per-call `Authority *AuthorityContext` in its options. The client emits authority as `X-Aether-*` headers; the HTTP transport forwards them and the Aether transport translates them into the proxy's structured authorization envelope.

## Aether transport

To route every request through an existing [Aether](https://github.com/scitrera/aether) connection (via `proxy_http`) instead of a direct HTTP call, install the companion module and pass its transport:

```bash
go get github.com/scitrera/memorylayer/memorylayer-sdk-go/aether
```

```go
import (
	aethersdk "github.com/scitrera/aether/sdk/go/aether"
	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
	mlaether "github.com/scitrera/memorylayer/memorylayer-sdk-go/aether"
)

// agent is an already-connected *aethersdk.AgentClient (or *ServiceClient).
transport := mlaether.NewTransport(agent, mlaether.WithTarget("sv::memorylayer"))

client, _ := memorylayer.NewClient(
	memorylayer.WithTransport(transport),
	memorylayer.WithAPIKey("..."),
	memorylayer.WithWorkspaceID("ws_123"),
)
```

The core and transport modules release together as `memorylayer-sdk-go/v0.2.0` and
`memorylayer-sdk-go/aether/v0.2.0`. The transport uses Aether API/SDK v0.2.3
and both modules require Go 1.25.14 or newer. The core module has no external
dependencies. The generated Go publish workflow creates both module tags at the
root release tag's commit after the Go tests pass.

The transport does **not** own the Aether connection — its `Close()` is a no-op, and disposing of the underlying client is the caller's responsibility. A runnable example: [`aether/example`](aether/example).

## License

Apache 2.0.
