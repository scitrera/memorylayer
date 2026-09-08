package memorylayer

import (
	"context"
	"net/http"
	"net/url"
)

// AgentSpecification is a typed, revisioned reusable subagent definition.
type AgentSpecification struct {
	ID                 string         `json:"id"`
	TenantID           string         `json:"tenant_id"`
	WorkspaceID        string         `json:"workspace_id"`
	Key                string         `json:"key"`
	Name               string         `json:"name"`
	Description        string         `json:"description"`
	Instructions       string         `json:"instructions"`
	InvocationGuidance string         `json:"invocation_guidance"`
	Model              string         `json:"model"`
	MaxTurns           int            `json:"max_turns"`
	AllowedTools       []string       `json:"allowed_tools"`
	DeniedTools        []string       `json:"denied_tools"`
	Skills             []string       `json:"skills"`
	MCPServers         []string       `json:"mcp_servers"`
	PermissionMode     string         `json:"permission_mode"`
	ExecPolicyHint     string         `json:"exec_policy_hint"`
	Background         bool           `json:"background"`
	Enabled            bool           `json:"enabled"`
	SchemaVersion      int            `json:"schema_version"`
	Metadata           map[string]any `json:"metadata"`
	Revision           int            `json:"revision"`
	ETag               string         `json:"etag"`
	CreatedBy          *string        `json:"created_by,omitempty"`
	UpdatedBy          *string        `json:"updated_by,omitempty"`
	CreatedAt          string         `json:"created_at"`
	UpdatedAt          string         `json:"updated_at"`
	DeletedAt          *string        `json:"deleted_at,omitempty"`
}

// AgentSpecificationCreateInput is the typed content accepted by AgentSpecificationsAPI.Create.
type AgentSpecificationCreateInput struct {
	Key                string         `json:"key"`
	Name               string         `json:"name"`
	Description        string         `json:"description"`
	Instructions       string         `json:"instructions"`
	InvocationGuidance string         `json:"invocation_guidance,omitempty"`
	Model              string         `json:"model,omitempty"`
	MaxTurns           int            `json:"max_turns,omitempty"`
	AllowedTools       []string       `json:"allowed_tools,omitempty"`
	DeniedTools        []string       `json:"denied_tools,omitempty"`
	Skills             []string       `json:"skills,omitempty"`
	MCPServers         []string       `json:"mcp_servers,omitempty"`
	PermissionMode     string         `json:"permission_mode,omitempty"`
	ExecPolicyHint     string         `json:"exec_policy_hint,omitempty"`
	Background         bool           `json:"background,omitempty"`
	Enabled            *bool          `json:"enabled,omitempty"`
	SchemaVersion      int            `json:"schema_version,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
}

// AgentSpecificationReplaceInput is a complete semantic replacement. Key is immutable.
type AgentSpecificationReplaceInput struct {
	Name               string         `json:"name"`
	Description        string         `json:"description"`
	Instructions       string         `json:"instructions"`
	InvocationGuidance string         `json:"invocation_guidance,omitempty"`
	Model              string         `json:"model,omitempty"`
	MaxTurns           int            `json:"max_turns,omitempty"`
	AllowedTools       []string       `json:"allowed_tools,omitempty"`
	DeniedTools        []string       `json:"denied_tools,omitempty"`
	Skills             []string       `json:"skills,omitempty"`
	MCPServers         []string       `json:"mcp_servers,omitempty"`
	PermissionMode     string         `json:"permission_mode,omitempty"`
	ExecPolicyHint     string         `json:"exec_policy_hint,omitempty"`
	Background         bool           `json:"background,omitempty"`
	Enabled            *bool          `json:"enabled,omitempty"`
	SchemaVersion      int            `json:"schema_version,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
}

type AgentSpecificationMutationOptions = PromptNoteMutationOptions
type AgentSpecificationGetOptions = PromptNoteGetOptions
type AgentSpecificationListOptions = PromptNoteListOptions
type AgentSpecificationHistoryOptions = PromptNoteHistoryOptions

type AgentSpecificationMutationResult struct {
	Specification AgentSpecification
	Replayed      bool
	ETag          string
}

type AgentSpecificationPage struct {
	Specifications []AgentSpecification `json:"specifications"`
	NextPageToken  string               `json:"next_page_token"`
}

type AgentSpecificationRevision struct {
	Specification AgentSpecification `json:"specification"`
	Action        string             `json:"action"`
	OperationID   string             `json:"operation_id"`
}

type AgentSpecificationRevisionPage struct {
	Revisions     []AgentSpecificationRevision `json:"revisions"`
	NextPageToken string                       `json:"next_page_token"`
}

// AgentSpecificationsAPI is the typed /v1/agent-specifications namespace.
type AgentSpecificationsAPI struct{ client *Client }

func (a *AgentSpecificationsAPI) Create(ctx context.Context, input AgentSpecificationCreateInput, opts AgentSpecificationMutationOptions) (*AgentSpecificationMutationResult, error) {
	payload, err := promptNotePayload(input, a.client.resolveWorkspace(opts.WorkspaceID))
	if err != nil {
		return nil, err
	}
	return a.mutate(ctx, http.MethodPost, "/agent-specifications", payload, opts, "*")
}

func (a *AgentSpecificationsAPI) Replace(ctx context.Context, specificationID string, input AgentSpecificationReplaceInput, opts AgentSpecificationMutationOptions) (*AgentSpecificationMutationResult, error) {
	payload, err := promptNotePayload(input, a.client.resolveWorkspace(opts.WorkspaceID))
	if err != nil {
		return nil, err
	}
	return a.mutate(ctx, http.MethodPut, "/agent-specifications/"+url.PathEscape(specificationID), payload, opts, opts.ETag)
}

func (a *AgentSpecificationsAPI) Delete(ctx context.Context, specificationID string, opts AgentSpecificationMutationOptions) (*AgentSpecificationMutationResult, error) {
	q := url.Values{}
	if ws := a.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	return a.mutateSpec(ctx, requestSpec{
		method: http.MethodDelete, path: "/agent-specifications/" + url.PathEscape(specificationID), query: q,
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// Restore reactivates a tombstoned specification without changing its stable
// identity or semantic content.
func (a *AgentSpecificationsAPI) Restore(ctx context.Context, specificationID string, opts AgentSpecificationMutationOptions) (*AgentSpecificationMutationResult, error) {
	q := url.Values{}
	if ws := a.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	return a.mutateSpec(ctx, requestSpec{
		method: http.MethodPost, path: "/agent-specifications/" + url.PathEscape(specificationID) + "/restore", query: q,
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

func (a *AgentSpecificationsAPI) Get(ctx context.Context, specificationID string, opts AgentSpecificationGetOptions) (*AgentSpecification, error) {
	q := url.Values{}
	if ws := a.client.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out struct {
		Specification AgentSpecification `json:"specification"`
	}
	resp, err := a.client.doJSONResponse(ctx, requestSpec{method: http.MethodGet, path: "/agent-specifications/" + url.PathEscape(specificationID), query: q, authority: opts.Authority}, &out)
	if err != nil {
		return nil, err
	}
	if out.Specification.ETag == "" {
		out.Specification.ETag = resp.Header.Get("ETag")
	}
	return &out.Specification, nil
}

func (a *AgentSpecificationsAPI) List(ctx context.Context, opts AgentSpecificationListOptions) (*AgentSpecificationPage, error) {
	q := promptNotePageQuery(a.client.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out AgentSpecificationPage
	if err := a.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/agent-specifications", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (a *AgentSpecificationsAPI) History(ctx context.Context, specificationID string, opts AgentSpecificationHistoryOptions) (*AgentSpecificationRevisionPage, error) {
	q := promptNotePageQuery(a.client.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	var out AgentSpecificationRevisionPage
	if err := a.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/agent-specifications/" + url.PathEscape(specificationID) + "/revisions", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (a *AgentSpecificationsAPI) mutate(ctx context.Context, method, path string, payload []byte, opts AgentSpecificationMutationOptions, expected string) (*AgentSpecificationMutationResult, error) {
	header := "If-Match"
	if expected == "*" {
		header = "If-None-Match"
	}
	return a.mutateSpec(ctx, requestSpec{method: method, path: path, body: payload, contentType: "application/json", headers: mutationHeaders(opts.IdempotencyKey, header, expected), authority: opts.Authority})
}

func (a *AgentSpecificationsAPI) mutateSpec(ctx context.Context, spec requestSpec) (*AgentSpecificationMutationResult, error) {
	var out struct {
		Specification AgentSpecification `json:"specification"`
		Replayed      bool               `json:"replayed"`
	}
	resp, err := a.client.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Specification.ETag
	}
	return &AgentSpecificationMutationResult{Specification: out.Specification, Replayed: out.Replayed, ETag: etag}, nil
}
