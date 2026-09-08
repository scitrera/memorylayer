package memorylayer

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// WorkspaceView is one durable checkout, worktree, snapshot, overlay, or
// directory identity within a logical MemoryLayer workspace.
type WorkspaceView struct {
	ID              string         `json:"id"`
	TenantID        string         `json:"tenant_id"`
	WorkspaceID     string         `json:"workspace_id"`
	ViewID          string         `json:"view_id"`
	Kind            string         `json:"kind"`
	DisplayName     *string        `json:"display_name,omitempty"`
	MemoryContextID *string        `json:"memory_context_id,omitempty"`
	Capabilities    []string       `json:"capabilities"`
	Metadata        map[string]any `json:"metadata"`
	SchemaVersion   int            `json:"schema_version"`
	Revision        int            `json:"revision"`
	Sequence        int            `json:"sequence"`
	ETag            string         `json:"etag"`
	CreatedBy       *string        `json:"created_by,omitempty"`
	UpdatedBy       *string        `json:"updated_by,omitempty"`
	CreatedAt       time.Time      `json:"created_at"`
	UpdatedAt       time.Time      `json:"updated_at"`
	DeletedAt       *time.Time     `json:"deleted_at,omitempty"`
}

// WorkspaceViewCreateInput creates a stable logical view identity.
type WorkspaceViewCreateInput struct {
	ViewID          string         `json:"view_id"`
	Kind            string         `json:"kind"`
	DisplayName     *string        `json:"display_name,omitempty"`
	MemoryContextID *string        `json:"memory_context_id,omitempty"`
	Capabilities    []string       `json:"capabilities,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
	SchemaVersion   int            `json:"schema_version,omitempty"`
}

// WorkspaceViewReplaceInput replaces a view's mutable description. ViewID is
// path-addressed and intentionally immutable.
type WorkspaceViewReplaceInput struct {
	Kind            string         `json:"kind"`
	DisplayName     *string        `json:"display_name,omitempty"`
	MemoryContextID *string        `json:"memory_context_id,omitempty"`
	Capabilities    []string       `json:"capabilities,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
	SchemaVersion   int            `json:"schema_version,omitempty"`
}

// WorkspaceVCSObservation is portable VCS state. It deliberately contains no
// host-local absolute path.
type WorkspaceVCSObservation struct {
	Kind         string  `json:"kind,omitempty"`
	RepositoryID *string `json:"repository_id,omitempty"`
	WorktreeID   *string `json:"worktree_id,omitempty"`
	HeadRevision *string `json:"head_revision,omitempty"`
	Branch       *string `json:"branch,omitempty"`
	Dirty        *bool   `json:"dirty,omitempty"`
	Detached     *bool   `json:"detached,omitempty"`
}

// WorkspaceViewObservation is the authoritative latest state published by one
// observer for one view.
type WorkspaceViewObservation struct {
	ID               string                   `json:"id"`
	TenantID         string                   `json:"tenant_id"`
	WorkspaceID      string                   `json:"workspace_id"`
	ViewID           string                   `json:"view_id"`
	ObserverID       string                   `json:"observer_id"`
	Generation       string                   `json:"generation"`
	Sequence         int64                    `json:"sequence"`
	ToolHostID       *string                  `json:"tool_host_id,omitempty"`
	ExecutionSite    *string                  `json:"execution_site,omitempty"`
	RootRef          *string                  `json:"root_ref,omitempty"`
	Capabilities     []string                 `json:"capabilities"`
	VCS              *WorkspaceVCSObservation `json:"vcs,omitempty"`
	ObservedAt       time.Time                `json:"observed_at"`
	ExpiresAt        *time.Time               `json:"expires_at,omitempty"`
	Metadata         map[string]any           `json:"metadata"`
	SchemaVersion    int                      `json:"schema_version"`
	ResourceRevision int                      `json:"resource_revision"`
	ResourceSequence int64                    `json:"resource_sequence"`
	ETag             string                   `json:"etag"`
	CreatedBy        *string                  `json:"created_by,omitempty"`
	UpdatedBy        *string                  `json:"updated_by,omitempty"`
	CreatedAt        time.Time                `json:"created_at"`
	UpdatedAt        time.Time                `json:"updated_at"`
	DeletedAt        *time.Time               `json:"deleted_at,omitempty"`
}

// WorkspaceViewObservationInput publishes one observer cursor and portable
// view state. ObserverID is supplied as a method argument.
type WorkspaceViewObservationInput struct {
	Generation    string                   `json:"generation"`
	Sequence      int64                    `json:"sequence"`
	ToolHostID    *string                  `json:"tool_host_id,omitempty"`
	ExecutionSite *string                  `json:"execution_site,omitempty"`
	RootRef       *string                  `json:"root_ref,omitempty"`
	Capabilities  []string                 `json:"capabilities,omitempty"`
	VCS           *WorkspaceVCSObservation `json:"vcs,omitempty"`
	ObservedAt    *time.Time               `json:"observed_at,omitempty"`
	ExpiresAt     *time.Time               `json:"expires_at,omitempty"`
	Metadata      map[string]any           `json:"metadata,omitempty"`
	SchemaVersion int                      `json:"schema_version,omitempty"`
}

// WorkspaceViewMutationOptions configures a conditional view mutation.
type WorkspaceViewMutationOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	ETag           string
	Authority      *AuthorityContext
}

// WorkspaceViewListOptions configures current-head pagination.
type WorkspaceViewListOptions struct {
	WorkspaceID    string
	Limit          int
	PageToken      string
	IncludeDeleted bool
	Authority      *AuthorityContext
}

// WorkspaceViewHistoryOptions configures immutable revision pagination.
type WorkspaceViewHistoryOptions struct {
	WorkspaceID string
	Limit       int
	PageToken   string
	Authority   *AuthorityContext
}

// WorkspaceObservationMutationOptions configures a conditional observer
// publication. Create sends If-None-Match; Replace sends If-Match.
type WorkspaceObservationMutationOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	ETag           string
	Authority      *AuthorityContext
}

// WorkspaceObservationListOptions configures current observation pagination.
type WorkspaceObservationListOptions struct {
	WorkspaceID string
	Limit       int
	PageToken   string
	Authority   *AuthorityContext
}

// WorkspaceViewMutationResult includes replay status and response ETag.
type WorkspaceViewMutationResult struct {
	View     WorkspaceView
	Replayed bool
	ETag     string
}

// WorkspaceObservationMutationResult includes replay status and response ETag.
type WorkspaceObservationMutationResult struct {
	Observation WorkspaceViewObservation
	Replayed    bool
	ETag        string
}

type WorkspaceViewPage struct {
	Views         []WorkspaceView `json:"views"`
	NextPageToken string          `json:"next_page_token"`
}

type WorkspaceViewRevision struct {
	View        WorkspaceView `json:"view"`
	Action      string        `json:"action"`
	OperationID string        `json:"operation_id"`
}

type WorkspaceViewRevisionPage struct {
	Revisions     []WorkspaceViewRevision `json:"revisions"`
	NextPageToken string                  `json:"next_page_token"`
}

type WorkspaceViewObservationPage struct {
	Observations  []WorkspaceViewObservation `json:"observations"`
	NextPageToken string                     `json:"next_page_token"`
}

type WorkspaceViewObservationRevision struct {
	Observation WorkspaceViewObservation `json:"observation"`
	Action      string                   `json:"action"`
	OperationID string                   `json:"operation_id"`
}

type WorkspaceViewObservationRevisionPage struct {
	Revisions     []WorkspaceViewObservationRevision `json:"revisions"`
	NextPageToken string                             `json:"next_page_token"`
}

// WorkspaceViewsAPI is the typed /v1/workspaces/{workspace}/views namespace.
type WorkspaceViewsAPI struct{ client *Client }

func (w *WorkspaceViewsAPI) Create(ctx context.Context, input WorkspaceViewCreateInput, opts WorkspaceViewMutationOptions) (*WorkspaceViewMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return w.mutateView(ctx, requestSpec{
		method: http.MethodPost, path: workspaceViewsPath(workspaceID), body: body,
		contentType: "application/json", headers: mutationHeaders(opts.IdempotencyKey, "If-None-Match", "*"), authority: opts.Authority,
	})
}

func (w *WorkspaceViewsAPI) Replace(ctx context.Context, viewID string, input WorkspaceViewReplaceInput, opts WorkspaceViewMutationOptions) (*WorkspaceViewMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return w.mutateView(ctx, requestSpec{
		method: http.MethodPut, path: workspaceViewPath(workspaceID, viewID), body: body,
		contentType: "application/json", headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// Delete writes a durable view tombstone when opts.ETag matches the head.
func (w *WorkspaceViewsAPI) Delete(ctx context.Context, viewID string, opts WorkspaceViewMutationOptions) (*WorkspaceViewMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	return w.mutateView(ctx, requestSpec{
		method: http.MethodDelete, path: workspaceViewPath(workspaceID, viewID),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// Restore reactivates a tombstoned view without changing its stable identity.
func (w *WorkspaceViewsAPI) Restore(ctx context.Context, viewID string, opts WorkspaceViewMutationOptions) (*WorkspaceViewMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	return w.mutateView(ctx, requestSpec{
		method: http.MethodPost, path: workspaceViewPath(workspaceID, viewID) + "/restore",
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

func (w *WorkspaceViewsAPI) Get(ctx context.Context, workspaceID, viewID string, authority *AuthorityContext) (*WorkspaceView, error) {
	workspaceID, err := w.workspaceID(workspaceID)
	if err != nil {
		return nil, err
	}
	var out struct {
		View WorkspaceView `json:"view"`
	}
	resp, err := w.client.doJSONResponse(ctx, requestSpec{
		method: http.MethodGet, path: workspaceViewPath(workspaceID, viewID), authority: authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	if out.View.ETag == "" {
		out.View.ETag = resp.Header.Get("ETag")
	}
	return &out.View, nil
}

func (w *WorkspaceViewsAPI) List(ctx context.Context, opts WorkspaceViewListOptions) (*WorkspaceViewPage, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	q := workspaceExecutionPageQuery(opts.Limit, opts.PageToken)
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out WorkspaceViewPage
	if err := w.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: workspaceViewsPath(workspaceID), query: q, authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (w *WorkspaceViewsAPI) History(ctx context.Context, viewID string, opts WorkspaceViewHistoryOptions) (*WorkspaceViewRevisionPage, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	var out WorkspaceViewRevisionPage
	if err := w.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: workspaceViewPath(workspaceID, viewID) + "/revisions",
		query: workspaceExecutionPageQuery(opts.Limit, opts.PageToken), authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (w *WorkspaceViewsAPI) CreateObservation(
	ctx context.Context,
	viewID, observerID string,
	input WorkspaceViewObservationInput,
	opts WorkspaceObservationMutationOptions,
) (*WorkspaceObservationMutationResult, error) {
	return w.mutateObservation(ctx, viewID, observerID, input, opts, "If-None-Match", "*")
}

func (w *WorkspaceViewsAPI) ReplaceObservation(
	ctx context.Context,
	viewID, observerID string,
	input WorkspaceViewObservationInput,
	opts WorkspaceObservationMutationOptions,
) (*WorkspaceObservationMutationResult, error) {
	return w.mutateObservation(ctx, viewID, observerID, input, opts, "If-Match", opts.ETag)
}

// DeleteObservation writes a durable observer tombstone.
func (w *WorkspaceViewsAPI) DeleteObservation(
	ctx context.Context,
	viewID, observerID string,
	opts WorkspaceObservationMutationOptions,
) (*WorkspaceObservationMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	return w.mutateObservationSpec(ctx, requestSpec{
		method: http.MethodDelete, path: workspaceObservationPath(workspaceID, viewID, observerID),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// RestoreObservation reactivates a tombstoned observer head.
func (w *WorkspaceViewsAPI) RestoreObservation(
	ctx context.Context,
	viewID, observerID string,
	opts WorkspaceObservationMutationOptions,
) (*WorkspaceObservationMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	return w.mutateObservationSpec(ctx, requestSpec{
		method: http.MethodPost, path: workspaceObservationPath(workspaceID, viewID, observerID) + "/restore",
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

func (w *WorkspaceViewsAPI) GetObservation(
	ctx context.Context,
	workspaceID, viewID, observerID string,
	authority *AuthorityContext,
) (*WorkspaceViewObservation, error) {
	workspaceID, err := w.workspaceID(workspaceID)
	if err != nil {
		return nil, err
	}
	var out struct {
		Observation WorkspaceViewObservation `json:"observation"`
	}
	resp, err := w.client.doJSONResponse(ctx, requestSpec{
		method: http.MethodGet, path: workspaceObservationPath(workspaceID, viewID, observerID), authority: authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	if out.Observation.ETag == "" {
		out.Observation.ETag = resp.Header.Get("ETag")
	}
	return &out.Observation, nil
}

func (w *WorkspaceViewsAPI) ListObservations(
	ctx context.Context,
	viewID string,
	opts WorkspaceObservationListOptions,
) (*WorkspaceViewObservationPage, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	var out WorkspaceViewObservationPage
	if err := w.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: workspaceViewPath(workspaceID, viewID) + "/observations",
		query: workspaceExecutionPageQuery(opts.Limit, opts.PageToken), authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (w *WorkspaceViewsAPI) ObservationHistory(
	ctx context.Context,
	viewID, observerID string,
	opts WorkspaceObservationListOptions,
) (*WorkspaceViewObservationRevisionPage, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	var out WorkspaceViewObservationRevisionPage
	if err := w.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: workspaceObservationPath(workspaceID, viewID, observerID) + "/revisions",
		query: workspaceExecutionPageQuery(opts.Limit, opts.PageToken), authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (w *WorkspaceViewsAPI) mutateView(ctx context.Context, spec requestSpec) (*WorkspaceViewMutationResult, error) {
	var out struct {
		View     WorkspaceView `json:"view"`
		Replayed bool          `json:"replayed"`
	}
	resp, err := w.client.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.View.ETag
	}
	return &WorkspaceViewMutationResult{View: out.View, Replayed: out.Replayed, ETag: etag}, nil
}

func (w *WorkspaceViewsAPI) mutateObservation(
	ctx context.Context,
	viewID, observerID string,
	input WorkspaceViewObservationInput,
	opts WorkspaceObservationMutationOptions,
	conditionalHeader, expected string,
) (*WorkspaceObservationMutationResult, error) {
	workspaceID, err := w.workspaceID(opts.WorkspaceID)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return w.mutateObservationSpec(ctx, requestSpec{
		method: http.MethodPut, path: workspaceObservationPath(workspaceID, viewID, observerID), body: body,
		contentType: "application/json", headers: mutationHeaders(opts.IdempotencyKey, conditionalHeader, expected), authority: opts.Authority,
	})
}

func (w *WorkspaceViewsAPI) mutateObservationSpec(ctx context.Context, spec requestSpec) (*WorkspaceObservationMutationResult, error) {
	var out struct {
		Observation WorkspaceViewObservation `json:"observation"`
		Replayed    bool                     `json:"replayed"`
	}
	resp, err := w.client.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Observation.ETag
	}
	return &WorkspaceObservationMutationResult{Observation: out.Observation, Replayed: out.Replayed, ETag: etag}, nil
}

func (w *WorkspaceViewsAPI) workspaceID(value string) (string, error) {
	value = w.client.resolveWorkspace(value)
	if value == "" {
		return "", errors.New("memorylayer: workspace_id must be provided or set on the client")
	}
	return value, nil
}

func workspaceViewsPath(workspaceID string) string {
	return "/workspaces/" + url.PathEscape(workspaceID) + "/views"
}

func workspaceViewPath(workspaceID, viewID string) string {
	return workspaceViewsPath(workspaceID) + "/" + url.PathEscape(viewID)
}

func workspaceObservationPath(workspaceID, viewID, observerID string) string {
	return workspaceViewPath(workspaceID, viewID) + "/observations/" + url.PathEscape(observerID)
}

func workspaceExecutionPageQuery(limit int, pageToken string) url.Values {
	q := url.Values{}
	if limit > 0 {
		q.Set("limit", strconv.Itoa(limit))
	}
	if pageToken != "" {
		q.Set("page_token", pageToken)
	}
	return q
}
