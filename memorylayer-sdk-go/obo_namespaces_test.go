package memorylayer

import (
	"context"
	"net/http"
	"testing"
)

// TestOBONamespacesInjectAuthorityAndWorkspace verifies that a bound-namespace
// proxy injects the proxy's OBO authority (as X-Aether-* headers) and its
// workspace into namespace requests.
func TestOBONamespacesInjectAuthorityAndWorkspace(t *testing.T) {
	var gotHeader http.Header
	var gotWorkspace string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Clone()
		gotWorkspace = r.URL.Query().Get("workspace_id")
		_, _ = w.Write([]byte(`{"skills":[]}`))
	})

	// Proxy bound to grant g_1 / user:alice, scoped to ws_team.
	proxy := client.ActingFor("g_1", &PrincipalRef{Type: "user", ID: "alice"}).ForWorkspace("ws_team")

	if _, err := proxy.Skills().List(context.Background(), ListSkillsOptions{}); err != nil {
		t.Fatalf("Skills.List: %v", err)
	}

	if gotWorkspace != "ws_team" {
		t.Errorf("workspace_id = %q, want ws_team", gotWorkspace)
	}
	if got := gotHeader.Get("X-Aether-Grant-ID"); got != "g_1" {
		t.Errorf("grant header = %q", got)
	}
	if got := gotHeader.Get("X-Aether-Subject-ID"); got != "alice" {
		t.Errorf("subject-id = %q", got)
	}
}

// TestOBONamespacesExplicitOverride verifies that explicit per-call workspace
// overrides the bound one.
func TestOBONamespacesExplicitOverride(t *testing.T) {
	var gotWorkspace string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotWorkspace = r.URL.Query().Get("workspace_id")
		_, _ = w.Write([]byte(`{"mcp_servers":[]}`))
	})

	proxy := client.ForWorkspace("ws_bound")
	if _, err := proxy.McpServers().List(context.Background(), ListMcpServersOptions{WorkspaceID: "ws_explicit"}); err != nil {
		t.Fatalf("McpServers.List: %v", err)
	}
	if gotWorkspace != "ws_explicit" {
		t.Errorf("workspace_id = %q, want ws_explicit (explicit override)", gotWorkspace)
	}
}
