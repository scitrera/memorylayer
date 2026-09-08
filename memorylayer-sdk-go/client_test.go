package memorylayer

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// newTestClient spins an httptest server with handler h and returns a client
// pointed at it plus a cleanup func.
func newTestClient(t *testing.T, h http.HandlerFunc, opts ...Option) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	base := []Option{WithBaseURL(srv.URL), WithAPIKey("test-key"), WithWorkspaceID("ws_default")}
	client, err := NewClient(append(base, opts...)...)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client, srv
}

func TestRememberEncodesRequest(t *testing.T) {
	var gotBody map[string]any
	var gotAuth, gotPath string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memory":{"id":"mem_1","workspace_id":"ws_default","content":"hi","type":"semantic","importance":0.8,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}}`))
	})

	mem, err := client.Remember(context.Background(), "hi", RememberOptions{
		Type:       MemoryTypeSemantic,
		Importance: 0.8,
		Tags:       []string{"a", "b"},
	})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if gotPath != "/v1/memories" {
		t.Errorf("path = %q, want /v1/memories", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("auth = %q", gotAuth)
	}
	if gotBody["content"] != "hi" || gotBody["type"] != "semantic" {
		t.Errorf("body = %#v", gotBody)
	}
	if gotBody["workspace_id"] != "ws_default" {
		t.Errorf("workspace_id = %v, want ws_default", gotBody["workspace_id"])
	}
	if mem.ID != "mem_1" || mem.Type != MemoryTypeSemantic {
		t.Errorf("memory = %#v", mem)
	}
}

func TestRecallTotalCountFallback(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		// No total_count in the response → SDK falls back to len(memories).
		_, _ = w.Write([]byte(`{"memories":[{"id":"m1","workspace_id":"ws","content":"x","type":"working","importance":0.1,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}]}`))
	})
	res, err := client.Recall(context.Background(), "q", RecallOptions{})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if res.TotalCount != 1 {
		t.Errorf("TotalCount = %d, want 1", res.TotalCount)
	}
}

func TestOBOHeaders(t *testing.T) {
	var h http.Header
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		h = r.Header.Clone()
		_, _ = w.Write([]byte(`{"memories":[]}`))
	})

	alice := client.ActingFor("g_1", &PrincipalRef{Type: "user", ID: "alice"})
	if _, err := alice.Recall(context.Background(), "q", RecallOptions{}); err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if got := h.Get("X-Aether-Grant-ID"); got != "g_1" {
		t.Errorf("grant header = %q", got)
	}
	if got := h.Get("X-Aether-Authority-Mode"); got != "on_behalf_of" {
		t.Errorf("mode header = %q", got)
	}
	if got := h.Get("X-Aether-Subject-Type"); got != "user" {
		t.Errorf("subject-type = %q", got)
	}
	if got := h.Get("X-Aether-Subject-ID"); got != "alice" {
		t.Errorf("subject-id = %q", got)
	}
}

func TestSessionHeader(t *testing.T) {
	var sid string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		sid = r.Header.Get("X-Session-ID")
		_, _ = w.Write([]byte(`{"memory":{"id":"m","workspace_id":"ws","content":"c","type":"working","importance":0.1,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}}`))
	})
	client.SetSession("sess_42")
	if _, err := client.GetMemory(context.Background(), "m"); err != nil {
		t.Fatalf("GetMemory: %v", err)
	}
	if sid != "sess_42" {
		t.Errorf("session header = %q", sid)
	}
	client.ClearSession()
	if _, err := client.GetMemory(context.Background(), "m"); err != nil {
		t.Fatalf("GetMemory: %v", err)
	}
	if sid != "" {
		t.Errorf("session header after clear = %q", sid)
	}
}

func TestErrorMapping(t *testing.T) {
	cases := []struct {
		status  int
		body    string
		feature string
		check   func(error) bool
	}{
		{401, `{"detail":"nope"}`, "", func(e error) bool { var x *AuthenticationError; return errors.As(e, &x) }},
		{403, `{"detail":"nope"}`, "", func(e error) bool { var x *AuthorizationError; return errors.As(e, &x) }},
		{404, `{"detail":"missing"}`, "", IsNotFound},
		{409, `{"detail":"conflict"}`, "", func(e error) bool { var x *ConflictError; return errors.As(e, &x) }},
		{412, `{"detail":"stale"}`, "", func(e error) bool { var x *PreconditionFailedError; return errors.As(e, &x) }},
		{422, `{"detail":"bad"}`, "", func(e error) bool { var x *ValidationError; return errors.As(e, &x) }},
		{428, `{"detail":"required"}`, "", func(e error) bool { var x *PreconditionRequiredError; return errors.As(e, &x) }},
		{429, `{"detail":"slow down"}`, "", func(e error) bool { var x *RateLimitError; return errors.As(e, &x) }},
		{500, `{"detail":"boom"}`, "", func(e error) bool { var x *ServerError; return errors.As(e, &x) }},
		{501, `{"detail":"nope"}`, "Docs", IsEnterpriseRequired},
		{501, `{"detail":"nope"}`, "", IsNotFound},
	}
	for _, tc := range cases {
		resp := &Response{StatusCode: tc.status, Body: []byte(tc.body)}
		err := mapError(resp, tc.feature)
		if err == nil || !tc.check(err) {
			t.Errorf("status %d feature %q: unexpected error %v", tc.status, tc.feature, err)
		}
	}
}

func TestForgetSendsHardParam(t *testing.T) {
	var hard string
	var method string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
		hard = r.URL.Query().Get("hard")
		w.WriteHeader(http.StatusNoContent)
	})
	if err := client.Forget(context.Background(), "m1", true); err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if method != http.MethodDelete || hard != "true" {
		t.Errorf("method=%s hard=%s", method, hard)
	}
}

func TestExportWorkspaceParsesNDJSON(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"type":"header","version":"2.0","workspace_id":"ws_x","exported_at":"2026-01-01"}`,
		`{"type":"memory","data":{"id":"m1"}}`,
		`{"type":"memory","data":{"id":"m2"}}`,
		`{"type":"association","data":{"id":"a1"}}`,
	}, "\n")
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(ndjson))
	})
	exp, err := client.ExportWorkspace(context.Background(), ExportWorkspaceOptions{WorkspaceID: "ws_x"})
	if err != nil {
		t.Fatalf("ExportWorkspace: %v", err)
	}
	if exp.Version != "2.0" || exp.WorkspaceID != "ws_x" {
		t.Errorf("header = %#v", exp)
	}
	if len(exp.Memories) != 2 || len(exp.Associations) != 1 {
		t.Errorf("counts: memories=%d associations=%d", len(exp.Memories), len(exp.Associations))
	}
	if exp.TotalMemories != 2 || exp.TotalAssociations != 1 {
		t.Errorf("totals: %d/%d", exp.TotalMemories, exp.TotalAssociations)
	}
}

func TestRetryOnTransientStatus(t *testing.T) {
	var calls int32
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable) // 503 → retried (GET is idempotent)
			return
		}
		_, _ = w.Write([]byte(`{"memory":{"id":"m","workspace_id":"ws","content":"c","type":"working","importance":0.1,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}}`))
	}, WithMaxRetries(2))
	if _, err := client.GetMemory(context.Background(), "m"); err != nil {
		t.Fatalf("GetMemory: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("calls = %d, want 2 (1 retry)", got)
	}
}

func TestMessageContentUnion(t *testing.T) {
	var msg struct {
		Content MessageContent `json:"content"`
	}
	if err := json.Unmarshal([]byte(`{"content":"plain text"}`), &msg); err != nil {
		t.Fatalf("unmarshal string: %v", err)
	}
	if !msg.Content.IsText() || msg.Content.Text != "plain text" {
		t.Errorf("string content = %#v", msg.Content)
	}
	if err := json.Unmarshal([]byte(`{"content":[{"type":"text","text":"a"}]}`), &msg); err != nil {
		t.Fatalf("unmarshal blocks: %v", err)
	}
	if msg.Content.IsText() || len(msg.Content.Blocks) != 1 || msg.Content.Blocks[0].Type != "text" {
		t.Errorf("blocks content = %#v", msg.Content)
	}
}

func TestUserChatWorkspaceSubstitution(t *testing.T) {
	var gotWS string
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotWS = r.URL.Query().Get("workspace_id")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		_, _ = w.Write([]byte(`{"messages":[]}`))
	})
	// ownership defaults to "user" → storage workspace becomes the sentinel and
	// the caller workspace is folded into per-message metadata.
	_, err := client.AppendMessages(context.Background(), "t1",
		[]map[string]any{{"role": "user", "content": "hi"}},
		AppendMessagesOptions{WorkspaceID: "ws_app"})
	if err != nil {
		t.Fatalf("AppendMessages: %v", err)
	}
	if gotWS != UserChatHomeWorkspace {
		t.Errorf("storage workspace = %q, want %q", gotWS, UserChatHomeWorkspace)
	}
	msgs, _ := gotBody["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v", gotBody["messages"])
	}
	m0, _ := msgs[0].(map[string]any)
	md, _ := m0["metadata"].(map[string]any)
	if md[MessageMetaAppWorkspaceKey] != "ws_app" {
		t.Errorf("app workspace metadata = %v, want ws_app", md[MessageMetaAppWorkspaceKey])
	}
}
