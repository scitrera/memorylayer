package memorylayer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const versionedSkillJSON = `{"id":"skl_1","tenant_id":"_default","workspace_id":"ws_project","name":"revisioned-skill","description":"Use CAS","version":"0.1.0","body":"## Instructions","metadata":{},"source_mode":"server","manifest_hash":"mh","bundle_hash":"bh","enabled":true,"revision":2,"etag":"\"skill-2-hash\"","created_at":"2026-08-09T00:00:00Z","updated_at":"2026-08-09T00:00:00Z"}`

func TestSkillsVersionedCreateSendsConditionalHeaders(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/skills" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Idempotency-Key") != "op-create" || r.Header.Get("If-None-Match") != "*" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("ETag", `"skill-2-hash"`)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"skill":` + versionedSkillJSON + `,"replayed":false}`))
	})

	result, err := client.Skills.CreateVersioned(context.Background(), SkillManifestCreateInput{
		Name: "revisioned-skill", Description: "Use CAS", Body: "## Instructions",
	}, SkillManifestMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "op-create",
	})
	if err != nil {
		t.Fatalf("CreateVersioned: %v", err)
	}
	if gotBody["workspace_id"] != "ws_project" || gotBody["name"] != "revisioned-skill" {
		t.Errorf("body = %#v", gotBody)
	}
	if result.Skill.Revision != 2 || result.ETag != `"skill-2-hash"` {
		t.Errorf("result = %#v", result)
	}
}

func TestSkillsVersionedLifecycleHistoryAndTombstoneGet(t *testing.T) {
	requests := 0
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("workspace_id") != "ws_project" {
			t.Errorf("workspace query = %q", r.URL.RawQuery)
		}
		switch requests {
		case 1:
			if r.Method != http.MethodPut || r.URL.Path != "/v1/skills/skl_1/manifest" {
				t.Errorf("replace = %s %s", r.Method, r.URL.Path)
			}
		case 2:
			if r.URL.Path != "/v1/skills/skl_1/delete" {
				t.Errorf("delete path = %s", r.URL.Path)
			}
		case 3:
			if r.URL.Path != "/v1/skills/skl_1/restore" {
				t.Errorf("restore path = %s", r.URL.Path)
			}
		case 4:
			if r.Method != http.MethodGet || r.URL.Query().Get("include_deleted") != "true" {
				t.Errorf("get tombstone = %s?%s", r.URL.Path, r.URL.RawQuery)
			}
			w.Header().Set("ETag", `"skill-2-hash"`)
			_, _ = w.Write([]byte(`{"skill":` + versionedSkillJSON + `}`))
			return
		case 5:
			if r.URL.Query().Get("page_token") != "cursor_1" || r.URL.Query().Get("limit") != "25" {
				t.Errorf("history query = %s", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`{"revisions":[{"skill":` + versionedSkillJSON + `,"action":"replace","operation_id":"op-replace"}],"next_page_token":"cursor_2"}`))
			return
		}
		if r.Header.Get("If-Match") != `"skill-1-hash"` || r.Header.Get("Idempotency-Key") == "" {
			t.Errorf("conditional headers = %#v", r.Header)
		}
		_, _ = w.Write([]byte(`{"skill":` + versionedSkillJSON + `,"replayed":true}`))
	})

	opts := SkillManifestMutationOptions{
		WorkspaceID: "ws_project", IdempotencyKey: "op", ETag: `"skill-1-hash"`,
	}
	if _, err := client.Skills.ReplaceManifest(context.Background(), "skl_1", SkillManifestReplaceInput{
		Description: "Use CAS", Metadata: map[string]any{}, SourceMode: "server", Enabled: true,
	}, opts); err != nil {
		t.Fatalf("ReplaceManifest: %v", err)
	}
	if result, err := client.Skills.DeleteVersioned(context.Background(), "skl_1", opts); err != nil || !result.Replayed {
		t.Fatalf("DeleteVersioned result=%#v err=%v", result, err)
	}
	if _, err := client.Skills.Restore(context.Background(), "skl_1", opts); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if skill, err := client.Skills.GetWithOptions(context.Background(), "skl_1", SkillGetOptions{
		WorkspaceID: "ws_project", IncludeDeleted: true,
	}); err != nil || skill.ETag != `"skill-2-hash"` {
		t.Fatalf("GetWithOptions skill=%#v err=%v", skill, err)
	}
	history, err := client.Skills.History(context.Background(), "skl_1", SkillHistoryOptions{
		WorkspaceID: "ws_project", Limit: 25, PageToken: "cursor_1",
	})
	if err != nil || len(history.Revisions) != 1 || history.NextPageToken != "cursor_2" {
		t.Fatalf("History page=%#v err=%v", history, err)
	}
}

func TestSkillsListIncludesAddendaParam(t *testing.T) {
	var gotPath, gotWorkspace, gotIncludeAddenda string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotWorkspace = r.URL.Query().Get("workspace_id")
		gotIncludeAddenda = r.URL.Query().Get("include_addenda")
		_, _ = w.Write([]byte(`{"skills":[]}`))
	})

	_, err := client.Skills.List(context.Background(), ListSkillsOptions{
		WorkspaceID:    "ws_project",
		IncludeAddenda: true,
	})
	if err != nil {
		t.Fatalf("Skills.List: %v", err)
	}
	if gotPath != "/v1/skills" {
		t.Errorf("path = %q, want /v1/skills", gotPath)
	}
	if gotWorkspace != "ws_project" {
		t.Errorf("workspace_id = %q, want ws_project", gotWorkspace)
	}
	if gotIncludeAddenda != "true" {
		t.Errorf("include_addenda = %q, want true", gotIncludeAddenda)
	}
}

func TestSkillsGetManifestWithOptionsIncludesQueryAndAuthority(t *testing.T) {
	var gotWorkspace, gotIncludeAddenda, gotGrant string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotWorkspace = r.URL.Query().Get("workspace_id")
		gotIncludeAddenda = r.URL.Query().Get("include_addenda")
		gotGrant = r.Header.Get("X-Aether-Grant-ID")
		if r.URL.Path != "/v1/skills/skl_1/manifest" {
			t.Errorf("path = %q, want /v1/skills/skl_1/manifest", r.URL.Path)
		}
		_, _ = w.Write([]byte("# skill"))
	})

	manifest, err := client.Skills.GetManifestWithOptions(context.Background(), "skl_1", GetSkillManifestOptions{
		WorkspaceID:    "ws_project",
		IncludeAddenda: true,
		Authority:      &AuthorityContext{GrantID: "grant_1"},
	})
	if err != nil {
		t.Fatalf("GetManifestWithOptions: %v", err)
	}
	if manifest != "# skill" {
		t.Errorf("manifest = %q, want # skill", manifest)
	}
	if gotWorkspace != "ws_project" || gotIncludeAddenda != "true" || gotGrant != "grant_1" {
		t.Errorf("workspace=%q include_addenda=%q grant=%q", gotWorkspace, gotIncludeAddenda, gotGrant)
	}
}

func TestSkillsResolveWithOptionsIncludesAddendaAndDecodesCandidates(t *testing.T) {
	var gotBody map[string]any
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"skill":{"id":"skl_1","workspace_id":"ws_project","name":"importer","description":"Import data","version":"0.1.0","body":"canonical","metadata":{},"source_mode":"server","manifest_hash":"mh","bundle_hash":"bh","enabled":true},"candidates":[{"id":"skl_1","workspace_id":"ws_project","name":"importer","description":"Import data","version":"0.1.0","body":"canonical","metadata":{},"source_mode":"server","manifest_hash":"mh","bundle_hash":"bh","enabled":true}]}`))
	})

	res, err := client.Skills.ResolveWithOptions(context.Background(), ResolveSkillOptions{
		Query:          "import csv",
		WorkspaceID:    "ws_project",
		IncludeAddenda: true,
	})
	if err != nil {
		t.Fatalf("ResolveWithOptions: %v", err)
	}
	if gotBody["query"] != "import csv" || gotBody["workspace_id"] != "ws_project" || gotBody["include_addenda"] != true {
		t.Errorf("body = %#v", gotBody)
	}
	if res.Skill == nil || res.Skill.ID != "skl_1" {
		t.Fatalf("skill = %#v", res.Skill)
	}
	if len(res.Skills) != 1 || res.Skills[0].ID != "skl_1" {
		t.Fatalf("candidates = %#v", res.Skills)
	}
}
