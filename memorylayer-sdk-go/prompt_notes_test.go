package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
)

const promptNoteJSON = `{"id":"vrs_1","tenant_id":"_default","workspace_id":"ws_project","key":"coding/style","title":"Style","content":"Use CAS","enabled":true,"schema_version":1,"metadata":{},"revision":1,"etag":"\"vr-1-hash\"","created_at":"2026-08-09T00:00:00Z","updated_at":"2026-08-09T00:00:00Z"}`

func TestPromptNotesCreateSendsTypedConditionalMutation(t *testing.T) {
	var gotBody map[string]any
	var gotIdempotency, gotIfNoneMatch, gotGrant string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/prompt-notes" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		gotIdempotency = r.Header.Get("Idempotency-Key")
		gotIfNoneMatch = r.Header.Get("If-None-Match")
		gotGrant = r.Header.Get("X-Aether-Grant-ID")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"vr-1-hash"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"note":` + promptNoteJSON + `,"replayed":false}`))
	})

	enabled := true
	result, err := client.PromptNotes.Create(context.Background(), PromptNoteCreateInput{
		Key: "coding/style", Title: "Style", Content: "Use CAS", Enabled: &enabled,
	}, PromptNoteMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "op-create", Authority: &AuthorityContext{GrantID: "grant_1"},
	})
	if err != nil {
		t.Fatalf("PromptNotes.Create: %v", err)
	}
	if gotIdempotency != "op-create" || gotIfNoneMatch != "*" || gotGrant != "grant_1" {
		t.Errorf("headers idempotency=%q if-none-match=%q grant=%q", gotIdempotency, gotIfNoneMatch, gotGrant)
	}
	if gotBody["workspace_id"] != "ws_project" || gotBody["key"] != "coding/style" || gotBody["enabled"] != true {
		t.Errorf("body = %#v", gotBody)
	}
	if result.Note.ID != "vrs_1" || result.Note.Revision != 1 || result.ETag != `"vr-1-hash"` || result.Replayed {
		t.Errorf("result = %#v", result)
	}
}

func TestPromptNotesReplaceDeleteAndRestoreSendIfMatch(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("If-Match") != `"vr-1-hash"` || r.Header.Get("Idempotency-Key") == "" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		if requests == 1 && (r.Method != http.MethodPut || r.URL.Path != "/v1/prompt-notes/vrs_1") {
			t.Errorf("replace request = %s %s", r.Method, r.URL.Path)
		}
		if requests == 2 && (r.Method != http.MethodDelete || r.URL.Query().Get("workspace_id") != "ws_project") {
			t.Errorf("delete request = %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
		}
		if requests == 3 && (r.Method != http.MethodPost || r.URL.Path != "/v1/prompt-notes/vrs_1/restore" || r.URL.Query().Get("workspace_id") != "ws_project") {
			t.Errorf("restore request = %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
		}
		_, _ = w.Write([]byte(`{"note":` + promptNoteJSON + `,"replayed":true}`))
	})

	if _, err := client.PromptNotes.Replace(context.Background(), "vrs_1", PromptNoteReplaceInput{
		Title: "Style", Content: "Use CAS",
	}, PromptNoteMutationOptions{WorkspaceID: "ws_project", IdempotencyKey: "op-replace", ETag: `"vr-1-hash"`}); err != nil {
		t.Fatalf("PromptNotes.Replace: %v", err)
	}
	result, err := client.PromptNotes.Delete(context.Background(), "vrs_1", PromptNoteMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "op-delete", ETag: `"vr-1-hash"`,
	})
	if err != nil {
		t.Fatalf("PromptNotes.Delete: %v", err)
	}
	if !result.Replayed {
		t.Fatal("delete replay flag = false, want true")
	}
	if _, err := client.PromptNotes.Restore(context.Background(), "vrs_1", PromptNoteMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "op-restore", ETag: `"vr-1-hash"`,
	}); err != nil {
		t.Fatalf("PromptNotes.Restore: %v", err)
	}
}

func TestPromptNotesListAndHistoryPreserveCursors(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("workspace_id") != "ws_project" || r.URL.Query().Get("page_token") != "cursor_1" || r.URL.Query().Get("limit") != "25" {
			t.Errorf("query = %s", r.URL.RawQuery)
		}
		if requests == 1 {
			if r.URL.Query().Get("include_deleted") != "true" {
				t.Errorf("include_deleted = %q", r.URL.Query().Get("include_deleted"))
			}
			_, _ = w.Write([]byte(`{"notes":[` + promptNoteJSON + `],"next_page_token":"cursor_2"}`))
			return
		}
		_, _ = w.Write([]byte(`{"revisions":[{"note":` + promptNoteJSON + `,"action":"create","operation_id":"op-create"}],"next_page_token":null}`))
	})

	page, err := client.PromptNotes.List(context.Background(), PromptNoteListOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1", IncludeDeleted: true,
	})
	if err != nil || len(page.Notes) != 1 || page.NextPageToken != "cursor_2" {
		t.Fatalf("PromptNotes.List page=%#v err=%v", page, err)
	}
	history, err := client.PromptNotes.History(context.Background(), "vrs_1", PromptNoteHistoryOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1",
	})
	if err != nil || len(history.Revisions) != 1 || history.Revisions[0].OperationID != "op-create" {
		t.Fatalf("PromptNotes.History history=%#v err=%v", history, err)
	}
}

func TestPromptNotesCreateRetriesBecauseItHasIdempotencyKey(t *testing.T) {
	var calls int32
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"note":` + promptNoteJSON + `}`))
	}, WithMaxRetries(1))

	_, err := client.PromptNotes.Create(context.Background(), PromptNoteCreateInput{
		Key: "coding/style", Title: "Style", Content: "Use CAS",
	}, PromptNoteMutationOptions{IdempotencyKey: "stable-op"})
	if err != nil {
		t.Fatalf("PromptNotes.Create: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("calls = %d, want 2", got)
	}
}
