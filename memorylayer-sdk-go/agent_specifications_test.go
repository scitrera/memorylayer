package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const agentSpecificationJSON = `{"id":"vrs_agent","tenant_id":"_default","workspace_id":"ws_project","key":"review/evidence","name":"Evidence reviewer","description":"Review evidence","instructions":"Cite evidence","invocation_guidance":"Use before completion","model":"review-model","max_turns":4,"allowed_tools":["read_file"],"denied_tools":["shell"],"skills":["review"],"mcp_servers":[],"permission_mode":"read_only","exec_policy_hint":"","background":false,"enabled":true,"schema_version":1,"metadata":{},"revision":1,"etag":"\"vr-agent-hash\"","created_at":"2026-08-09T00:00:00Z","updated_at":"2026-08-09T00:00:00Z"}`

func TestAgentSpecificationsCreateUsesTypedConditionalMutation(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/agent-specifications" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Idempotency-Key") != "agent-create" || r.Header.Get("If-None-Match") != "*" || r.Header.Get("X-Aether-Grant-ID") != "grant_1" {
			t.Errorf("headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"vr-agent-hash"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"specification":` + agentSpecificationJSON + `,"replayed":false}`))
	})

	enabled := true
	result, err := client.AgentSpecifications.Create(context.Background(), AgentSpecificationCreateInput{
		Key: "review/evidence", Name: "Evidence reviewer", Description: "Review evidence", Instructions: "Cite evidence",
		AllowedTools: []string{"read_file"}, PermissionMode: "read_only", Enabled: &enabled,
	}, AgentSpecificationMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "agent-create", Authority: &AuthorityContext{GrantID: "grant_1"},
	})
	if err != nil {
		t.Fatalf("AgentSpecifications.Create: %v", err)
	}
	if gotBody["workspace_id"] != "ws_project" || gotBody["key"] != "review/evidence" {
		t.Errorf("body = %#v", gotBody)
	}
	if result.Specification.ID != "vrs_agent" || result.ETag != `"vr-agent-hash"` || result.Replayed {
		t.Errorf("result = %#v", result)
	}
}

func TestAgentSpecificationsListHistoryAndCASPaths(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		switch requests {
		case 1:
			if r.Method != http.MethodGet || r.URL.Path != "/v1/agent-specifications" || r.URL.Query().Get("workspace_id") != "ws_project" {
				t.Errorf("list request = %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"specifications":[` + agentSpecificationJSON + `],"next_page_token":"next"}`))
		case 2:
			_, _ = w.Write([]byte(`{"revisions":[{"specification":` + agentSpecificationJSON + `,"action":"create","operation_id":"agent-create"}]}`))
		case 3, 4, 5:
			if r.Header.Get("If-Match") != `"vr-agent-hash"` || r.Header.Get("Idempotency-Key") == "" {
				t.Errorf("conditional headers = %#v", r.Header)
			}
			if requests == 5 && (r.Method != http.MethodPost || r.URL.Path != "/v1/agent-specifications/vrs_agent/restore") {
				t.Errorf("restore request = %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"specification":` + agentSpecificationJSON + `}`))
		}
	})

	page, err := client.AgentSpecifications.List(context.Background(), AgentSpecificationListOptions{WorkspaceID: "ws_project", Limit: 10})
	if err != nil || len(page.Specifications) != 1 || page.NextPageToken != "next" {
		t.Fatalf("list page=%#v err=%v", page, err)
	}
	history, err := client.AgentSpecifications.History(context.Background(), "vrs_agent", AgentSpecificationHistoryOptions{WorkspaceID: "ws_project"})
	if err != nil || len(history.Revisions) != 1 || history.Revisions[0].OperationID != "agent-create" {
		t.Fatalf("history=%#v err=%v", history, err)
	}
	if _, err := client.AgentSpecifications.Replace(context.Background(), "vrs_agent", AgentSpecificationReplaceInput{Name: "Evidence reviewer", Description: "Review evidence", Instructions: "Cite evidence"}, AgentSpecificationMutationOptions{IdempotencyKey: "agent-replace", ETag: `"vr-agent-hash"`}); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if _, err := client.AgentSpecifications.Delete(context.Background(), "vrs_agent", AgentSpecificationMutationOptions{IdempotencyKey: "agent-delete", ETag: `"vr-agent-hash"`}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := client.AgentSpecifications.Restore(context.Background(), "vrs_agent", AgentSpecificationMutationOptions{IdempotencyKey: "agent-restore", ETag: `"vr-agent-hash"`}); err != nil {
		t.Fatalf("restore: %v", err)
	}
}
