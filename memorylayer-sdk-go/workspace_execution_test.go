package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const workspaceViewJSON = `{"id":"vrs_view_1","tenant_id":"_default","workspace_id":"ws_project","view_id":"worktree-a","kind":"git_worktree","display_name":"Feature worktree","capabilities":["workspace.read","workspace.write"],"metadata":{},"schema_version":1,"revision":1,"sequence":10,"etag":"\"vr-1-view\"","created_at":"2026-08-10T12:00:00Z","updated_at":"2026-08-10T12:00:00Z"}`

const workspaceObservationJSON = `{"id":"vrs_obs_1","tenant_id":"_default","workspace_id":"ws_project","view_id":"worktree-a","observer_id":"sahara-tui-1","generation":"generation-a","sequence":2,"tool_host_id":"host-tui-1","execution_site":"client","root_ref":"local-root-1","capabilities":["workspace.read","workspace.write"],"vcs":{"kind":"git","head_revision":"0123456789abcdef","branch":"feature/a","dirty":true},"observed_at":"2026-08-10T12:00:00Z","metadata":{},"schema_version":1,"resource_revision":2,"resource_sequence":12,"etag":"\"vr-2-observation\"","created_at":"2026-08-10T11:00:00Z","updated_at":"2026-08-10T12:00:00Z"}`

func TestWorkspaceViewsCreateSendsConditionalTypedMutation(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/workspaces/ws_project/views" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Idempotency-Key") != "view-create-a" || r.Header.Get("If-None-Match") != "*" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"vr-1-view"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"view":` + workspaceViewJSON + `,"replayed":false}`))
	})

	displayName := "Feature worktree"
	result, err := client.WorkspaceViews.Create(context.Background(), WorkspaceViewCreateInput{
		ViewID: "worktree-a", Kind: "git_worktree", DisplayName: &displayName,
		Capabilities: []string{"workspace.write", "workspace.read"},
	}, WorkspaceViewMutationOptions{WorkspaceID: "ws_project", IdempotencyKey: "view-create-a"})
	if err != nil {
		t.Fatalf("WorkspaceViews.Create: %v", err)
	}
	if gotBody["view_id"] != "worktree-a" || gotBody["kind"] != "git_worktree" {
		t.Errorf("body = %#v", gotBody)
	}
	if result.View.ViewID != "worktree-a" || result.ETag != `"vr-1-view"` || result.Replayed {
		t.Errorf("result = %#v", result)
	}
}

func TestWorkspaceViewsObservationCreateAndReplaceUseExactHostBinding(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Method != http.MethodPut || r.URL.Path != "/v1/workspaces/ws_project/views/worktree-a/observations/sahara-tui-1" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if requests == 1 {
			if r.Header.Get("If-None-Match") != "*" || r.Header.Get("Idempotency-Key") != "observe-create" {
				t.Errorf("create headers = %#v", r.Header)
			}
			w.WriteHeader(http.StatusCreated)
		} else if r.Header.Get("If-Match") != `"vr-1-observation"` || r.Header.Get("Idempotency-Key") != "observe-replace" {
			t.Errorf("replace headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		var payload map[string]any
		_ = json.Unmarshal(body, &payload)
		if payload["tool_host_id"] != "host-tui-1" || payload["root_ref"] != "local-root-1" || payload["generation"] != "generation-a" {
			t.Errorf("observation body = %#v", payload)
		}
		w.Header().Set("ETag", `"vr-2-observation"`)
		_, _ = w.Write([]byte(`{"observation":` + workspaceObservationJSON + `,"replayed":false}`))
	})

	toolHostID, site, rootRef := "host-tui-1", "client", "local-root-1"
	input := WorkspaceViewObservationInput{
		Generation: "generation-a", Sequence: 2,
		ToolHostID: &toolHostID, ExecutionSite: &site, RootRef: &rootRef,
	}
	created, err := client.WorkspaceViews.CreateObservation(
		context.Background(), "worktree-a", "sahara-tui-1", input,
		WorkspaceObservationMutationOptions{WorkspaceID: "ws_project", IdempotencyKey: "observe-create"},
	)
	if err != nil || created.Observation.ObserverID != "sahara-tui-1" {
		t.Fatalf("CreateObservation result=%#v err=%v", created, err)
	}
	replaced, err := client.WorkspaceViews.ReplaceObservation(
		context.Background(), "worktree-a", "sahara-tui-1", input,
		WorkspaceObservationMutationOptions{
			WorkspaceID: "ws_project", IdempotencyKey: "observe-replace", ETag: `"vr-1-observation"`,
		},
	)
	if err != nil || replaced.ETag != `"vr-2-observation"` || replaced.Observation.Sequence != 2 {
		t.Fatalf("ReplaceObservation result=%#v err=%v", replaced, err)
	}
}

func TestWorkspaceViewsListAndHistoriesPreserveScopedCursors(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("page_token") != "cursor_1" || r.URL.Query().Get("limit") != "25" {
			t.Errorf("query = %s", r.URL.RawQuery)
		}
		switch requests {
		case 1:
			if r.URL.Path != "/v1/workspaces/ws_project/views" || r.URL.Query().Get("include_deleted") != "true" {
				t.Errorf("list request = %s?%s", r.URL.Path, r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"views":[` + workspaceViewJSON + `],"next_page_token":"cursor_2"}`))
		case 2:
			_, _ = w.Write([]byte(`{"revisions":[{"view":` + workspaceViewJSON + `,"action":"create","operation_id":"view-create"}],"next_page_token":null}`))
		case 3:
			_, _ = w.Write([]byte(`{"observations":[` + workspaceObservationJSON + `],"next_page_token":"cursor_2"}`))
		case 4:
			_, _ = w.Write([]byte(`{"revisions":[{"observation":` + workspaceObservationJSON + `,"action":"replace","operation_id":"observe-replace"}],"next_page_token":null}`))
		}
	})

	page, err := client.WorkspaceViews.List(context.Background(), WorkspaceViewListOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1", IncludeDeleted: true,
	})
	if err != nil || len(page.Views) != 1 || page.NextPageToken != "cursor_2" {
		t.Fatalf("List page=%#v err=%v", page, err)
	}
	history, err := client.WorkspaceViews.History(context.Background(), "worktree-a", WorkspaceViewHistoryOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1",
	})
	if err != nil || len(history.Revisions) != 1 || history.Revisions[0].OperationID != "view-create" {
		t.Fatalf("History page=%#v err=%v", history, err)
	}
	observations, err := client.WorkspaceViews.ListObservations(context.Background(), "worktree-a", WorkspaceObservationListOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1",
	})
	if err != nil || len(observations.Observations) != 1 || observations.NextPageToken != "cursor_2" {
		t.Fatalf("ListObservations page=%#v err=%v", observations, err)
	}
	observationHistory, err := client.WorkspaceViews.ObservationHistory(
		context.Background(), "worktree-a", "sahara-tui-1",
		WorkspaceObservationListOptions{WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1"},
	)
	if err != nil || len(observationHistory.Revisions) != 1 || observationHistory.Revisions[0].OperationID != "observe-replace" {
		t.Fatalf("ObservationHistory page=%#v err=%v", observationHistory, err)
	}
}

func TestWorkspaceViewsDeleteAndRestoreUseConditionalTombstones(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("If-Match") == "" || r.Header.Get("Idempotency-Key") == "" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		switch requests {
		case 1:
			if r.Method != http.MethodDelete || r.URL.Path != "/v1/workspaces/ws_project/views/worktree-a" {
				t.Errorf("delete view = %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"view":` + workspaceViewJSON + `,"replayed":false}`))
		case 2:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/workspaces/ws_project/views/worktree-a/restore" {
				t.Errorf("restore view = %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"view":` + workspaceViewJSON + `,"replayed":false}`))
		case 3:
			if r.Method != http.MethodDelete || r.URL.Path != "/v1/workspaces/ws_project/views/worktree-a/observations/sahara-tui-1" {
				t.Errorf("delete observation = %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"observation":` + workspaceObservationJSON + `,"replayed":false}`))
		case 4:
			if r.Method != http.MethodPost || r.URL.Path != "/v1/workspaces/ws_project/views/worktree-a/observations/sahara-tui-1/restore" {
				t.Errorf("restore observation = %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"observation":` + workspaceObservationJSON + `,"replayed":false}`))
		}
	})
	viewOpts := WorkspaceViewMutationOptions{WorkspaceID: "ws_project", IdempotencyKey: "view-delete", ETag: `"vr-1-view"`}
	if _, err := client.WorkspaceViews.Delete(context.Background(), "worktree-a", viewOpts); err != nil {
		t.Fatal(err)
	}
	viewOpts.IdempotencyKey = "view-restore"
	if _, err := client.WorkspaceViews.Restore(context.Background(), "worktree-a", viewOpts); err != nil {
		t.Fatal(err)
	}
	observationOpts := WorkspaceObservationMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "observation-delete", ETag: `"vr-2-observation"`,
	}
	if _, err := client.WorkspaceViews.DeleteObservation(context.Background(), "worktree-a", "sahara-tui-1", observationOpts); err != nil {
		t.Fatal(err)
	}
	observationOpts.IdempotencyKey = "observation-restore"
	if _, err := client.WorkspaceViews.RestoreObservation(context.Background(), "worktree-a", "sahara-tui-1", observationOpts); err != nil {
		t.Fatal(err)
	}
}
