// Package memorylayer is the Go SDK for MemoryLayer.ai — memory infrastructure
// for LLM-powered agents.
//
// It mirrors the design of the Python (memorylayer-client) and TypeScript
// (@scitrera/memorylayer-sdk) SDKs: a single Client exposing memory CRUD,
// semantic recall/reflect, the association graph, sessions, workspaces, chat
// threads, the context sandbox, the entity registry, API tokens, and the
// Enterprise document/dataset surfaces, plus the Skills, McpServers, and
// Knowledgebase (Kb), and Repository Planning Graph (Rpg) namespaces.
//
// # Transports
//
// The Client issues every request through a [Transport]. Two are available:
//
//   - The default HTTP transport (constructed automatically) speaks directly to
//     a MemoryLayer server over net/http. It has no third-party dependencies.
//   - An Aether transport, provided by the companion module
//     github.com/scitrera/memorylayer/memorylayer-sdk-go/aether, routes every
//     request through an existing Aether SDK connection via proxy_http. Pass it
//     with [WithTransport]. It is a separate module so HTTP-only callers do not
//     pull in the Aether SDK's gRPC dependency tree.
//
// # On-behalf-of (OBO) authority
//
// Every request may carry an [AuthorityContext]. The Client emits it as
// X-Aether-* headers; the HTTP transport forwards them and the Aether transport
// translates them into the proxy's structured authorization envelope. Use
// [Client.ActingFor] and [Client.ForWorkspace] to bind authority/workspace to a
// lightweight proxy without mutating the shared Client.
//
// # Basic usage
//
//	client, err := memorylayer.NewClient(
//		memorylayer.WithBaseURL("https://api.memorylayer.ai"),
//		memorylayer.WithAPIKey("your-api-key"),
//		memorylayer.WithWorkspaceID("ws_123"),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer client.Close()
//
//	mem, err := client.Remember(ctx, "User prefers Go", memorylayer.RememberOptions{
//		Type:       memorylayer.MemoryTypeSemantic,
//		Importance: 0.8,
//	})
//	results, err := client.Recall(ctx, "coding preferences", memorylayer.RecallOptions{Limit: 5})
package memorylayer

// Version is the SDK version, kept in step with the Python/TypeScript SDKs.
const Version = "0.2.0"
