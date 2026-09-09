// Command example wires the MemoryLayer Go SDK onto an Aether connection: every
// request is tunneled through the Aether gateway via proxy_http to the
// MemoryLayer service topic, instead of a direct HTTP call.
//
//	go run ./example -server localhost:50051 -workspace prod
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	aethersdk "github.com/scitrera/aether/sdk/go/aether"
	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
	mlaether "github.com/scitrera/memorylayer/memorylayer-sdk-go/aether"
)

func main() {
	server := flag.String("server", "localhost:50051", "Aether gateway address")
	workspace := flag.String("workspace", "default", "Aether workspace")
	target := flag.String("target", mlaether.DefaultTarget, "MemoryLayer Aether service topic")
	apiKey := flag.String("api-key", "", "MemoryLayer API key (bearer token)")
	mlWorkspace := flag.String("ml-workspace", "go-sdk-demo", "MemoryLayer workspace ID")
	flag.Parse()

	// 1. Build and connect an Aether client (the SDK does NOT own its lifecycle).
	agent, err := aethersdk.NewAgentClient(aethersdk.AgentOptions{
		ClientOptions:  aethersdk.ClientOptions{ServerAddr: *server},
		Workspace:      *workspace,
		Implementation: "memorylayer-go-sdk",
		Specifier:      "example",
	})
	if err != nil {
		log.Fatalf("new aether agent: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if err := agent.Connect(ctx); err != nil {
		log.Fatalf("aether connect: %v", err)
	}
	defer func() { _ = agent.Close() }()

	// 2. Route the MemoryLayer SDK through the Aether connection.
	transport := mlaether.NewTransport(agent, mlaether.WithTarget(*target))
	client, err := memorylayer.NewClient(
		memorylayer.WithTransport(transport),
		memorylayer.WithAPIKey(*apiKey),
		memorylayer.WithWorkspaceID(*mlWorkspace),
	)
	if err != nil {
		log.Fatalf("new memorylayer client: %v", err)
	}
	defer func() { _ = client.Close() }()

	// 3. Use the SDK exactly as with the HTTP transport.
	mem, err := client.Remember(ctx, "Routed through Aether.", memorylayer.RememberOptions{
		Type:       memorylayer.MemoryTypeEpisodic,
		Importance: 0.6,
	})
	if err != nil {
		log.Fatalf("remember: %v", err)
	}
	fmt.Printf("stored memory %s via Aether transport\n", mem.ID)

	// OBO: act on behalf of a user for a scoped query.
	alice := client.ActingFor("g_example_grant", &memorylayer.PrincipalRef{Type: "user", ID: "alice"})
	results, err := alice.Recall(ctx, "anything?", memorylayer.RecallOptions{Limit: 3})
	if err != nil {
		log.Fatalf("recall (obo): %v", err)
	}
	fmt.Printf("recall returned %d memories\n", results.TotalCount)
}
