package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const versionedMemoryJSON = `{"id":"mem_1","logical_key":"preferences/editor","tenant_id":"_default","workspace_id":"ws_project","content":"Prefer helix.","content_hash":"hash","type":"semantic","subtype":"preference","importance":0.4,"tags":["editor"],"metadata":{"source":"instruction"},"refinement_metadata":{"confidence":"revised"},"access_count":0,"decay_factor":0.8,"pinned":true,"status":"active","revision":2,"etag":"\"memory-2-hash\"","created_at":"2026-08-09T00:00:00Z","updated_at":"2026-08-09T00:00:00Z"}`

func TestCreateMemoryVersionedSendsConditionalHeaders(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/memories" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Idempotency-Key") != "memory-create" || r.Header.Get("If-None-Match") != "*" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"memory-2-hash"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"memory":` + versionedMemoryJSON + `,"replayed":false}`))
	})

	result, err := client.CreateMemoryVersioned(context.Background(), SemanticMemoryCreateInput{
		LogicalKey: "preferences/editor", Content: "Prefer helix.", Type: MemoryTypeSemantic,
		RefinementMetadata: map[string]any{"confidence": "explicit"},
	}, MemoryMutationOptions{WorkspaceID: "ws_project", IdempotencyKey: "memory-create"})
	if err != nil {
		t.Fatalf("CreateMemoryVersioned: %v", err)
	}
	if gotBody["logical_key"] != "preferences/editor" || gotBody["workspace_id"] != "ws_project" {
		t.Errorf("body = %#v", gotBody)
	}
	if result.Memory.Revision != 2 || result.ETag != `"memory-2-hash"` {
		t.Errorf("result = %#v", result)
	}
}

func TestVersionedMemoryLifecycleHistoryAndTombstoneGet(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("workspace_id") != "ws_project" {
			t.Errorf("workspace query = %q", r.URL.RawQuery)
		}
		switch requests {
		case 1:
			if r.Method != http.MethodPut || r.URL.Path != "/v1/memories/mem_1/semantic" {
				t.Errorf("replace = %s %s", r.Method, r.URL.Path)
			}
		case 2:
			if r.URL.Path != "/v1/memories/mem_1/delete" {
				t.Errorf("delete path = %s", r.URL.Path)
			}
		case 3:
			if r.URL.Path != "/v1/memories/mem_1/restore" {
				t.Errorf("restore path = %s", r.URL.Path)
			}
		case 4:
			if r.Method != http.MethodGet || r.URL.Query().Get("include_deleted") != "true" {
				t.Errorf("get tombstone = %s?%s", r.URL.Path, r.URL.RawQuery)
			}
			w.Header().Set("ETag", `"memory-2-hash"`)
			_, _ = w.Write([]byte(`{"memory":` + versionedMemoryJSON + `}`))
			return
		case 5:
			if r.URL.Query().Get("page_token") != "cursor_1" || r.URL.Query().Get("limit") != "25" {
				t.Errorf("history query = %s", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"revisions":[{"memory":` + versionedMemoryJSON + `,"action":"replace","operation_id":"memory-replace"}],"next_page_token":"cursor_2"}`))
			return
		}
		if r.Header.Get("If-Match") != `"memory-1-hash"` || r.Header.Get("Idempotency-Key") == "" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		_, _ = w.Write([]byte(`{"memory":` + versionedMemoryJSON + `,"replayed":true}`))
	})

	opts := MemoryMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "memory-op", ETag: `"memory-1-hash"`,
	}
	if _, err := client.ReplaceSemanticMemory(context.Background(), "mem_1", SemanticMemoryReplaceInput{
		Content: "Prefer helix.", Type: MemoryTypeSemantic, Tags: []string{"editor"},
		RefinementMetadata: map[string]any{"confidence": "revised"}, Pinned: true,
	}, opts); err != nil {
		t.Fatalf("ReplaceSemanticMemory: %v", err)
	}
	if result, err := client.DeleteMemoryVersioned(context.Background(), "mem_1", opts); err != nil || !result.Replayed {
		t.Fatalf("DeleteMemoryVersioned result=%#v err=%v", result, err)
	}
	if _, err := client.RestoreMemoryVersioned(context.Background(), "mem_1", opts); err != nil {
		t.Fatalf("RestoreMemoryVersioned: %v", err)
	}
	if memory, err := client.GetMemory(context.Background(), "mem_1", GetMemoryOptions{
		WorkspaceID: "ws_project", IncludeDeleted: true,
	}); err != nil || memory.ETag != `"memory-2-hash"` {
		t.Fatalf("GetMemory memory=%#v err=%v", memory, err)
	}
	history, err := client.MemoryHistory(context.Background(), "mem_1", MemoryHistoryOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1",
	})
	if err != nil || len(history.Revisions) != 1 || history.NextPageToken != "cursor_2" {
		t.Fatalf("MemoryHistory page=%#v err=%v", history, err)
	}
}
