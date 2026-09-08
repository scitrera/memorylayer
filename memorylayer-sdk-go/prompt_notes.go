package memorylayer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
)

// PromptNote is a typed, revisioned prompt supplement stored by MemoryLayer.
type PromptNote struct {
	ID            string         `json:"id"`
	TenantID      string         `json:"tenant_id"`
	WorkspaceID   string         `json:"workspace_id"`
	Key           string         `json:"key"`
	Title         string         `json:"title"`
	Content       string         `json:"content"`
	Enabled       bool           `json:"enabled"`
	SchemaVersion int            `json:"schema_version"`
	Metadata      map[string]any `json:"metadata"`
	Revision      int            `json:"revision"`
	ETag          string         `json:"etag"`
	CreatedBy     *string        `json:"created_by,omitempty"`
	UpdatedBy     *string        `json:"updated_by,omitempty"`
	CreatedAt     string         `json:"created_at"`
	UpdatedAt     string         `json:"updated_at"`
	DeletedAt     *string        `json:"deleted_at,omitempty"`
}

// PromptNoteCreateInput is the typed content accepted by PromptNotesAPI.Create.
type PromptNoteCreateInput struct {
	Key           string         `json:"key"`
	Title         string         `json:"title"`
	Content       string         `json:"content"`
	Enabled       *bool          `json:"enabled,omitempty"`
	SchemaVersion int            `json:"schema_version,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// PromptNoteReplaceInput is a complete semantic replacement. The stable key is
// intentionally absent because it is immutable.
type PromptNoteReplaceInput struct {
	Title         string         `json:"title"`
	Content       string         `json:"content"`
	Enabled       *bool          `json:"enabled,omitempty"`
	SchemaVersion int            `json:"schema_version,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// PromptNoteMutationOptions carries conditional-write and authority metadata.
// ETag is required for Replace, Delete, and Restore. Create always sends
// If-None-Match: *.
type PromptNoteMutationOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	ETag           string
	Authority      *AuthorityContext
}

// PromptNoteGetOptions configures PromptNotesAPI.Get.
type PromptNoteGetOptions struct {
	WorkspaceID    string
	IncludeDeleted bool
	Authority      *AuthorityContext
}

// PromptNoteListOptions configures PromptNotesAPI.List.
type PromptNoteListOptions struct {
	WorkspaceID    string
	Limit          int
	PageToken      string
	IncludeDeleted bool
	Authority      *AuthorityContext
}

// PromptNoteHistoryOptions configures PromptNotesAPI.History.
type PromptNoteHistoryOptions struct {
	WorkspaceID string
	Limit       int
	PageToken   string
	Authority   *AuthorityContext
}

// PromptNoteMutationResult includes idempotent-replay status and response ETag.
type PromptNoteMutationResult struct {
	Note     PromptNote
	Replayed bool
	ETag     string
}

// PromptNotePage is one stable-cursor page of current prompt-note heads.
type PromptNotePage struct {
	Notes         []PromptNote `json:"notes"`
	NextPageToken string       `json:"next_page_token"`
}

// PromptNoteRevision is one immutable prompt-note snapshot.
type PromptNoteRevision struct {
	Note        PromptNote `json:"note"`
	Action      string     `json:"action"`
	OperationID string     `json:"operation_id"`
}

// PromptNoteRevisionPage is one stable-cursor history page.
type PromptNoteRevisionPage struct {
	Revisions     []PromptNoteRevision `json:"revisions"`
	NextPageToken string               `json:"next_page_token"`
}

// PromptNotesAPI is the typed /v1/prompt-notes namespace.
type PromptNotesAPI struct{ client *Client }

// Create stores a new prompt note. IdempotencyKey must be stable across retries.
func (p *PromptNotesAPI) Create(ctx context.Context, input PromptNoteCreateInput, opts PromptNoteMutationOptions) (*PromptNoteMutationResult, error) {
	payload, err := promptNotePayload(input, p.client.resolveWorkspace(opts.WorkspaceID))
	if err != nil {
		return nil, err
	}
	return p.mutate(ctx, http.MethodPost, "/prompt-notes", payload, opts, "*")
}

// Replace atomically replaces a prompt note when opts.ETag matches its head.
func (p *PromptNotesAPI) Replace(ctx context.Context, noteID string, input PromptNoteReplaceInput, opts PromptNoteMutationOptions) (*PromptNoteMutationResult, error) {
	payload, err := promptNotePayload(input, p.client.resolveWorkspace(opts.WorkspaceID))
	if err != nil {
		return nil, err
	}
	return p.mutate(ctx, http.MethodPut, "/prompt-notes/"+url.PathEscape(noteID), payload, opts, opts.ETag)
}

// Delete writes a durable tombstone when opts.ETag matches the current head.
func (p *PromptNotesAPI) Delete(ctx context.Context, noteID string, opts PromptNoteMutationOptions) (*PromptNoteMutationResult, error) {
	q := url.Values{}
	if ws := p.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	return p.mutateSpec(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      "/prompt-notes/" + url.PathEscape(noteID),
		query:     q,
		headers:   mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag),
		authority: opts.Authority,
	})
}

// Restore reactivates a tombstone when opts.ETag matches the deleted head. Its
// stable ID, key, content, metadata, and creation provenance are preserved.
func (p *PromptNotesAPI) Restore(ctx context.Context, noteID string, opts PromptNoteMutationOptions) (*PromptNoteMutationResult, error) {
	q := url.Values{}
	if ws := p.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	return p.mutateSpec(ctx, requestSpec{
		method:    http.MethodPost,
		path:      "/prompt-notes/" + url.PathEscape(noteID) + "/restore",
		query:     q,
		headers:   mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag),
		authority: opts.Authority,
	})
}

// Get fetches one active note, or a tombstone when IncludeDeleted is true.
func (p *PromptNotesAPI) Get(ctx context.Context, noteID string, opts PromptNoteGetOptions) (*PromptNote, error) {
	q := url.Values{}
	if ws := p.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out struct {
		Note PromptNote `json:"note"`
	}
	resp, err := p.client.doJSONResponse(ctx, requestSpec{
		method: http.MethodGet, path: "/prompt-notes/" + url.PathEscape(noteID), query: q, authority: opts.Authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	if out.Note.ETag == "" {
		out.Note.ETag = resp.Header.Get("ETag")
	}
	return &out.Note, nil
}

// List returns current heads in descending mutation-sequence order.
func (p *PromptNotesAPI) List(ctx context.Context, opts PromptNoteListOptions) (*PromptNotePage, error) {
	q := promptNotePageQuery(p.client.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out PromptNotePage
	if err := p.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: "/prompt-notes", query: q, authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// History returns immutable revisions in descending mutation-sequence order.
func (p *PromptNotesAPI) History(ctx context.Context, noteID string, opts PromptNoteHistoryOptions) (*PromptNoteRevisionPage, error) {
	q := promptNotePageQuery(p.client.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	var out PromptNoteRevisionPage
	if err := p.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: "/prompt-notes/" + url.PathEscape(noteID) + "/revisions", query: q, authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (p *PromptNotesAPI) mutate(ctx context.Context, method, path string, payload []byte, opts PromptNoteMutationOptions, expected string) (*PromptNoteMutationResult, error) {
	header := "If-Match"
	if expected == "*" {
		header = "If-None-Match"
	}
	return p.mutateSpec(ctx, requestSpec{
		method: method, path: path, body: payload, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, header, expected), authority: opts.Authority,
	})
}

func (p *PromptNotesAPI) mutateSpec(ctx context.Context, spec requestSpec) (*PromptNoteMutationResult, error) {
	var out struct {
		Note     PromptNote `json:"note"`
		Replayed bool       `json:"replayed"`
	}
	resp, err := p.client.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Note.ETag
	}
	return &PromptNoteMutationResult{Note: out.Note, Replayed: out.Replayed, ETag: etag}, nil
}

func promptNotePayload(input any, workspaceID string) ([]byte, error) {
	raw, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if workspaceID != "" {
		payload["workspace_id"] = workspaceID
	}
	return json.Marshal(payload)
}

func mutationHeaders(idempotencyKey, conditionalHeader, expected string) map[string]string {
	return map[string]string{"Idempotency-Key": idempotencyKey, conditionalHeader: expected}
}

func promptNotePageQuery(workspaceID string, limit int, pageToken string) url.Values {
	q := url.Values{}
	if workspaceID != "" {
		q.Set("workspace_id", workspaceID)
	}
	if limit > 0 {
		q.Set("limit", strconv.Itoa(limit))
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return q
}
