package memorylayer

import (
	"context"
	"net/http"
	"net/url"
)

type RefinementEvidence struct {
	Kind        string  `json:"kind"`
	Reference   string  `json:"reference"`
	Description string  `json:"description"`
	ContentHash *string `json:"content_hash,omitempty"`
}

type RefinementResourceSnapshot struct {
	ResourceID    *string        `json:"resource_id,omitempty"`
	ResourceKey   string         `json:"resource_key"`
	ETag          *string        `json:"etag,omitempty"`
	SchemaVersion int            `json:"schema_version,omitempty"`
	Content       map[string]any `json:"content,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
	Deleted       bool           `json:"deleted,omitempty"`
}

type RefinementEdit struct {
	Action       string                      `json:"action"`
	ResourceKind string                      `json:"resource_kind"`
	ResourceKey  string                      `json:"resource_key"`
	ResourceID   *string                     `json:"resource_id,omitempty"`
	ExpectedETag *string                     `json:"expected_etag,omitempty"`
	BeforeETag   *string                     `json:"before_etag,omitempty"`
	AfterETag    *string                     `json:"after_etag,omitempty"`
	Reason       string                      `json:"reason"`
	Content      map[string]any              `json:"content,omitempty"`
	Before       *RefinementResourceSnapshot `json:"before,omitempty"`
	After        *RefinementResourceSnapshot `json:"after,omitempty"`
	Applied      *bool                       `json:"applied,omitempty"`
	Error        *string                     `json:"error,omitempty"`
}

type RefinementExternalReference struct {
	System    string  `json:"system"`
	ID        string  `json:"id"`
	AttemptID *string `json:"attempt_id,omitempty"`
}

// RefinementRecord is one immutable proposal, decision, application, or rollback audit record.
type RefinementRecord struct {
	ID                 string                       `json:"id"`
	TenantID           string                       `json:"tenant_id"`
	WorkspaceID        string                       `json:"workspace_id"`
	Key                string                       `json:"key"`
	RefinementID       string                       `json:"refinement_id"`
	Phase              string                       `json:"phase"`
	Trigger            string                       `json:"trigger"`
	Scope              string                       `json:"scope"`
	Summary            string                       `json:"summary"`
	Rationale          string                       `json:"rationale"`
	ExpectedOutcome    string                       `json:"expected_outcome"`
	Evidence           []RefinementEvidence         `json:"evidence"`
	Edits              []RefinementEdit             `json:"edits"`
	Outcome            string                       `json:"outcome"`
	ParentRecordID     *string                      `json:"parent_record_id,omitempty"`
	RollbackOfRecordID *string                      `json:"rollback_of_record_id,omitempty"`
	TaskRef            *RefinementExternalReference `json:"task_ref,omitempty"`
	ApprovalRef        *RefinementExternalReference `json:"approval_ref,omitempty"`
	SchemaVersion      int                          `json:"schema_version"`
	Metadata           map[string]any               `json:"metadata"`
	Revision           int                          `json:"revision"`
	ETag               string                       `json:"etag"`
	CreatedBy          *string                      `json:"created_by,omitempty"`
	CreatedAt          string                       `json:"created_at"`
}

type RefinementRecordCreateInput struct {
	Key                string                       `json:"key"`
	RefinementID       string                       `json:"refinement_id"`
	Phase              string                       `json:"phase"`
	Trigger            string                       `json:"trigger"`
	Scope              string                       `json:"scope"`
	Summary            string                       `json:"summary"`
	Rationale          string                       `json:"rationale"`
	ExpectedOutcome    string                       `json:"expected_outcome"`
	Evidence           []RefinementEvidence         `json:"evidence"`
	Edits              []RefinementEdit             `json:"edits,omitempty"`
	Outcome            string                       `json:"outcome"`
	ParentRecordID     *string                      `json:"parent_record_id,omitempty"`
	RollbackOfRecordID *string                      `json:"rollback_of_record_id,omitempty"`
	TaskRef            *RefinementExternalReference `json:"task_ref,omitempty"`
	ApprovalRef        *RefinementExternalReference `json:"approval_ref,omitempty"`
	SchemaVersion      int                          `json:"schema_version,omitempty"`
	Metadata           map[string]any               `json:"metadata,omitempty"`
}

type RefinementRecordCreateOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	Authority      *AuthorityContext
}

type RefinementRecordGetOptions struct {
	WorkspaceID string
	Authority   *AuthorityContext
}

type RefinementRecordListOptions struct {
	WorkspaceID   string
	Limit         int
	PageToken     string
	Phases        []string
	Outcomes      []string
	Scopes        []string
	ResourceKinds []string
	RefinementID  string
	Search        string
	Authority     *AuthorityContext
}

type RefinementRecordMutationResult struct {
	Record   RefinementRecord
	Replayed bool
	ETag     string
}

type RefinementRecordPage struct {
	Records       []RefinementRecord `json:"records"`
	NextPageToken string             `json:"next_page_token"`
	ScannedCount  int                `json:"scanned_count"`
	ScanTruncated bool               `json:"scan_truncated"`
}

// RefinementRecordsAPI is the append-only typed /v1/refinement-records namespace.
type RefinementRecordsAPI struct{ client *Client }

func (r *RefinementRecordsAPI) Create(ctx context.Context, input RefinementRecordCreateInput, opts RefinementRecordCreateOptions) (*RefinementRecordMutationResult, error) {
	payload, err := promptNotePayload(input, r.client.resolveWorkspace(opts.WorkspaceID))
	if err != nil {
		return nil, err
	}
	var out struct {
		Record   RefinementRecord `json:"record"`
		Replayed bool             `json:"replayed"`
	}
	resp, err := r.client.doJSONResponse(ctx, requestSpec{
		method: http.MethodPost, path: "/refinement-records", body: payload, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, "If-None-Match", "*"), authority: opts.Authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Record.ETag
	}
	return &RefinementRecordMutationResult{Record: out.Record, Replayed: out.Replayed, ETag: etag}, nil
}

func (r *RefinementRecordsAPI) Get(ctx context.Context, recordID string, opts RefinementRecordGetOptions) (*RefinementRecord, error) {
	q := url.Values{}
	if ws := r.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out struct {
		Record RefinementRecord `json:"record"`
	}
	resp, err := r.client.doJSONResponse(ctx, requestSpec{method: http.MethodGet, path: "/refinement-records/" + url.PathEscape(recordID), query: q, authority: opts.Authority}, &out)
	if err != nil {
		return nil, err
	}
	if out.Record.ETag == "" {
		out.Record.ETag = resp.Header.Get("ETag")
	}
	return &out.Record, nil
}

func (r *RefinementRecordsAPI) List(ctx context.Context, opts RefinementRecordListOptions) (*RefinementRecordPage, error) {
	q := promptNotePageQuery(r.client.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	for _, value := range opts.Phases {
		q.Add("phase", value)
	}
	for _, value := range opts.Outcomes {
		q.Add("outcome", value)
	}
	for _, value := range opts.Scopes {
		q.Add("scope", value)
	}
	for _, value := range opts.ResourceKinds {
		q.Add("resource_kind", value)
	}
	if opts.RefinementID != "" {
		q.Set("refinement_id", opts.RefinementID)
	}
	if opts.Search != "" {
		q.Set("search", opts.Search)
	}
	var out RefinementRecordPage
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/refinement-records", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
