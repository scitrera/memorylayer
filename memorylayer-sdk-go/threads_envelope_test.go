package memorylayer

import (
	"context"
	"net/http"
	"testing"
)

// The server wraps single-thread responses as {"thread": {...}}. Decoding those
// straight into a ChatThread yields a zero-valued struct with NO error (the
// wrapper key is simply unknown to the struct), so a caller gets an empty ID and
// only finds out when the next call 404s. These pin the unwrapping.

func TestCreateThreadUnwrapsThreadEnvelope(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"thread":{"id":"thread_abc","workspace_id":"ws1","ownership":"workspace"}}`))
	})

	thread, err := client.CreateThread(context.Background(), CreateThreadOptions{
		WorkspaceID: "ws1",
		Ownership:   "workspace",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	if thread.ID != "thread_abc" {
		t.Fatalf("id = %q, want thread_abc", thread.ID)
	}
	if thread.WorkspaceID != "ws1" {
		t.Errorf("workspace_id = %q, want ws1", thread.WorkspaceID)
	}
}

func TestGetThreadUnwrapsThreadEnvelope(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"thread":{"id":"thread_get","workspace_id":"ws1"}}`))
	})

	thread, err := client.GetThread(context.Background(), "thread_get", "ws1")
	if err != nil {
		t.Fatalf("GetThread: %v", err)
	}
	if thread.ID != "thread_get" {
		t.Fatalf("id = %q, want thread_get", thread.ID)
	}
}

func TestUpdateThreadUnwrapsThreadEnvelope(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"thread":{"id":"thread_upd","title":"renamed"}}`))
	})

	title := "renamed"
	thread, err := client.UpdateThread(context.Background(), "thread_upd", UpdateThreadOptions{
		WorkspaceID: "ws1",
		Title:       &title,
	})
	if err != nil {
		t.Fatalf("UpdateThread: %v", err)
	}
	if thread.ID != "thread_upd" {
		t.Fatalf("id = %q, want thread_upd", thread.ID)
	}
	if thread.Title == nil || *thread.Title != "renamed" {
		t.Errorf("title = %v, want renamed", thread.Title)
	}
}

// A bare (unwrapped) thread body must still decode, so the SDK keeps working
// against any deployment that does not wrap. Mirrors decodeMemoryEnvelope's
// tolerance.
func TestGetThreadAcceptsBareThreadBody(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"thread_bare","workspace_id":"ws1"}`))
	})

	thread, err := client.GetThread(context.Background(), "thread_bare", "ws1")
	if err != nil {
		t.Fatalf("GetThread: %v", err)
	}
	if thread.ID != "thread_bare" {
		t.Fatalf("id = %q, want thread_bare", thread.ID)
	}
}
