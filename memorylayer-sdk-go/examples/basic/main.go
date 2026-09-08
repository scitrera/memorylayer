// Command basic demonstrates the MemoryLayer Go SDK over the default HTTP
// transport: store a memory, search for it, and reflect.
//
//	go run ./examples/basic -base-url http://localhost:61001 -api-key "$MEMORYLAYER_API_KEY"
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
)

func main() {
	baseURL := flag.String("base-url", memorylayer.DefaultBaseURL, "MemoryLayer server base URL")
	apiKey := flag.String("api-key", "", "API key (bearer token)")
	workspace := flag.String("workspace", "go-sdk-demo", "workspace ID")
	flag.Parse()

	client, err := memorylayer.NewClient(
		memorylayer.WithBaseURL(*baseURL),
		memorylayer.WithAPIKey(*apiKey),
		memorylayer.WithWorkspaceID(*workspace),
	)
	if err != nil {
		log.Fatalf("new client: %v", err)
	}
	defer func() { _ = client.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	mem, err := client.Remember(ctx, "The user prefers Go for backend services.", memorylayer.RememberOptions{
		Type:       memorylayer.MemoryTypeSemantic,
		Subtype:    string(memorylayer.MemorySubtypePreference),
		Importance: 0.8,
		Tags:       []string{"preferences", "languages"},
	})
	if err != nil {
		log.Fatalf("remember: %v", err)
	}
	fmt.Printf("stored memory %s\n", mem.ID)

	results, err := client.Recall(ctx, "what language does the user like?", memorylayer.RecallOptions{
		Limit:               5,
		IncludeAssociations: memorylayer.Ptr(true),
	})
	if err != nil {
		log.Fatalf("recall: %v", err)
	}
	fmt.Printf("recall returned %d memories:\n", results.TotalCount)
	for _, m := range results.Memories {
		fmt.Printf("  - %s\n", m.Content)
	}

	reflection, err := client.Reflect(ctx, "summarize the user's language preferences", memorylayer.ReflectOptions{
		MaxTokens: 200,
	})
	if err != nil {
		// Reflect needs an LLM-capable server; treat as non-fatal for the demo.
		fmt.Printf("reflect unavailable: %v\n", err)
		return
	}
	fmt.Printf("reflection: %s\n", reflection.Reflection)
}
