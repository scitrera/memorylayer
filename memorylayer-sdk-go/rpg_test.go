package memorylayer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"reflect"
	"testing"
)

func TestRpgSyncEncodesRequest(t *testing.T) {
	var gotMethod, gotPath, gotSubject string
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, req *http.Request) {
		gotMethod = req.Method
		gotPath = req.URL.Path
		gotSubject = req.Header.Get("X-Aether-Subject-ID")
		if err := json.NewDecoder(req.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"nodes_created":1,"edges_created":1,"source_commit":"abc123"}`))
	})

	if client.Rpg == nil {
		t.Fatal("NewClient did not initialize Rpg namespace")
	}
	result, err := client.Rpg.Sync(
		context.Background(),
		[]RpgNodeInput{{
			NodeID: "src/main.go", NodeType: "rpg_file", Path: "src/main.go", Name: "main.go",
		}},
		[]RpgEdgeInput{{
			SourceID: "src", TargetID: "src/main.go", Relationship: "contains", Strength: Ptr(0.0),
		}},
		RpgSyncOptions{
			FullSync:     true,
			SourceCommit: "abc123",
			ContextID:    "rpg-task-42",
			Authority: &AuthorityContext{
				GrantID: "g-rpg",
				Subject: &PrincipalRef{Type: "user", ID: "alice"},
			},
		},
	)
	if err != nil {
		t.Fatalf("Rpg.Sync: %v", err)
	}
	if result.NodesCreated != 1 || gotMethod != http.MethodPost || gotPath != "/v1/rpg/sync" {
		t.Fatalf("result=%#v method=%s path=%s", result, gotMethod, gotPath)
	}
	if gotSubject != "alice" {
		t.Errorf("subject header = %q", gotSubject)
	}
	if gotBody["context_id"] != "rpg-task-42" || gotBody["full_sync"] != true {
		t.Errorf("body = %#v", gotBody)
	}
	nodes := gotBody["nodes"].([]any)
	node := nodes[0].(map[string]any)
	if node["node_id"] != "src/main.go" || node["node_type"] != "rpg_file" {
		t.Errorf("node = %#v", node)
	}
	edges := gotBody["edges"].([]any)
	edge := edges[0].(map[string]any)
	if edge["strength"] != float64(0) {
		t.Errorf("explicit zero edge strength was not preserved: %#v", edge)
	}
}

func TestRpgGetSubgraphEncodesFilters(t *testing.T) {
	var gotQuery url.Values
	client, _ := newTestClient(t, func(w http.ResponseWriter, req *http.Request) {
		gotQuery = req.URL.Query()
		_, _ = w.Write([]byte(`{"nodes":[],"edges":[],"root_id":"src","depth":2}`))
	})

	result, err := client.Rpg.GetSubgraph(context.Background(), RpgSubgraphOptions{
		NodeID:            "src",
		Depth:             2,
		NodeTypes:         []string{"rpg_file", "rpg_class"},
		RelationshipTypes: []string{"contains", "imports"},
	})
	if err != nil {
		t.Fatalf("Rpg.GetSubgraph: %v", err)
	}
	if result.RootID == nil || *result.RootID != "src" {
		t.Fatalf("result = %#v", result)
	}
	if got := gotQuery.Get("node_types"); got != "rpg_file,rpg_class" {
		t.Errorf("node_types = %q", got)
	}
	if got := gotQuery.Get("relationship_types"); got != "contains,imports" {
		t.Errorf("relationship_types = %q", got)
	}
}

func TestRpgEnrichDefaultsAsync(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, req *http.Request) {
		_ = json.NewDecoder(req.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"workspace_id":"ws_default","context_id":"rpg","async_mode":true}`))
	})

	result, err := client.Rpg.Enrich(context.Background(), RpgEnrichOptions{})
	if err != nil {
		t.Fatalf("Rpg.Enrich: %v", err)
	}
	if !result.AsyncMode || !reflect.DeepEqual(gotBody, map[string]any{
		"async": true, "context_id": "rpg", "phases": nil,
	}) {
		t.Fatalf("result=%#v body=%#v", result, gotBody)
	}
}
