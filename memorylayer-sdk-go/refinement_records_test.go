package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const refinementRecordJSON = `{"id":"vrs_refine","tenant_id":"_default","workspace_id":"ws_project","key":"refinements/r1/proposal","refinement_id":"r1","phase":"proposal","trigger":"user","scope":"workspace","summary":"Add reviewer","rationale":"Repeated unsupported claims","expected_outcome":"Claims cite evidence","evidence":[{"kind":"user_instruction","reference":"message:m1","description":"User requested evidence"}],"edits":[{"action":"create","resource_kind":"agent_specification","resource_key":"review/evidence","reason":"Reusable review"}],"outcome":"proposed","task_ref":{"system":"aether","id":"task-1"},"schema_version":1,"metadata":{},"revision":1,"etag":"\"vr-refine-hash\"","created_at":"2026-08-09T00:00:00Z"}`

func TestRefinementRecordsCreateIsTypedConditionalAndCarriesAuthority(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/refinement-records" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("If-None-Match") != "*" || r.Header.Get("Idempotency-Key") != "refine-create" || r.Header.Get("X-Aether-Grant-ID") != "grant_1" {
			t.Errorf("headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"vr-refine-hash"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"record":` + refinementRecordJSON + `}`))
	})

	result, err := client.RefinementRecords.Create(context.Background(), RefinementRecordCreateInput{
		Key: "refinements/r1/proposal", RefinementID: "r1", Phase: "proposal", Trigger: "user", Scope: "workspace",
		Summary: "Add reviewer", Rationale: "Repeated unsupported claims", ExpectedOutcome: "Claims cite evidence",
		Evidence: []RefinementEvidence{{Kind: "user_instruction", Reference: "message:m1", Description: "User requested evidence"}},
		Edits: []RefinementEdit{{
			Action: "create", ResourceKind: "agent_specification", ResourceKey: "review/evidence", Reason: "Reusable review",
			Content: map[string]any{"name": "Evidence reviewer", "instructions": "Check cited artifacts."},
		}},
		Outcome: "proposed",
	}, RefinementRecordCreateOptions{WorkspaceID: "ws_project", IdempotencyKey: "refine-create", Authority: &AuthorityContext{GrantID: "grant_1"}})
	if err != nil {
		t.Fatalf("RefinementRecords.Create: %v", err)
	}
	if gotBody["workspace_id"] != "ws_project" || gotBody["phase"] != "proposal" {
		t.Errorf("body = %#v", gotBody)
	}
	edits := gotBody["edits"].([]any)
	content := edits[0].(map[string]any)["content"].(map[string]any)
	if content["name"] != "Evidence reviewer" {
		t.Errorf("edit content = %#v", content)
	}
	if result.Record.ID != "vrs_refine" || result.ETag != `"vr-refine-hash"` {
		t.Errorf("result = %#v", result)
	}
}

func TestRefinementRecordsListAndGetPreserveWorkspaceAndCursor(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("workspace_id") != "ws_project" {
			t.Errorf("workspace = %q", r.URL.Query().Get("workspace_id"))
		}
		if requests == 1 {
			if r.URL.Query().Get("page_token") != "cursor" || r.URL.Query().Get("limit") != "25" {
				t.Errorf("query = %s", r.URL.RawQuery)
			}
			if got := r.URL.Query()["phase"]; len(got) != 2 || got[0] != "application" || got[1] != "rollback" ||
				r.URL.Query().Get("outcome") != "failed" || r.URL.Query().Get("scope") != "workspace" ||
				r.URL.Query().Get("resource_kind") != "memory" || r.URL.Query().Get("refinement_id") != "r1" ||
				r.URL.Query().Get("search") != "evidence" {
				t.Errorf("filters = %s", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"records":[` + refinementRecordJSON + `],"next_page_token":"next","scanned_count":9,"scan_truncated":true}`))
			return
		}
		w.Header().Set("ETag", `"vr-refine-hash"`)
		_, _ = w.Write([]byte(`{"record":` + refinementRecordJSON + `}`))
	})

	page, err := client.RefinementRecords.List(context.Background(), RefinementRecordListOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor",
		Phases: []string{"application", "rollback"}, Outcomes: []string{"failed"}, Scopes: []string{"workspace"},
		ResourceKinds: []string{"memory"}, RefinementID: "r1", Search: "evidence",
	})
	if err != nil || len(page.Records) != 1 || page.NextPageToken != "next" || page.ScannedCount != 9 || !page.ScanTruncated {
		t.Fatalf("list page=%#v err=%v", page, err)
	}
	record, err := client.RefinementRecords.Get(context.Background(), "vrs_refine", RefinementRecordGetOptions{WorkspaceID: "ws_project"})
	if err != nil || record.RefinementID != "r1" || record.TaskRef == nil || record.TaskRef.ID != "task-1" {
		t.Fatalf("get record=%#v err=%v", record, err)
	}
}
