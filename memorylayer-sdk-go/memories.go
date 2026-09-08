package memorylayer

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// SemanticMemoryCreateInput is the complete keyed document accepted by a
// conditional memory create. Associations remain separate child resources.
type SemanticMemoryCreateInput struct {
	LogicalKey         string         `json:"logical_key"`
	Content            string         `json:"content"`
	Type               MemoryType     `json:"type"`
	Subtype            string         `json:"subtype,omitempty"`
	Tags               []string       `json:"tags,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
	RefinementMetadata map[string]any `json:"refinement_metadata,omitempty"`
	Pinned             bool           `json:"pinned"`
	UserID             string         `json:"user_id,omitempty"`
	Scope              string         `json:"scope,omitempty"`
}

// SemanticMemoryReplaceInput completely replaces the refinement-owned
// semantic document while preserving native identity and operational fields.
type SemanticMemoryReplaceInput struct {
	Content            string         `json:"content"`
	Type               MemoryType     `json:"type"`
	Subtype            *string        `json:"subtype"`
	Tags               []string       `json:"tags"`
	RefinementMetadata map[string]any `json:"refinement_metadata"`
	Pinned             bool           `json:"pinned"`
}

// MemoryMutationOptions carries required idempotency, CAS, workspace, and
// delegated-authority metadata.
type MemoryMutationOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	ETag           string
	Authority      *AuthorityContext
}

// MemoryHistoryOptions configures immutable semantic history traversal.
type MemoryHistoryOptions struct {
	WorkspaceID string
	Limit       int
	PageToken   string
	Authority   *AuthorityContext
}

type MemoryMutationResult struct {
	Memory   Memory
	Replayed bool
	ETag     string
}

type MemoryRevision struct {
	Memory      Memory `json:"memory"`
	Action      string `json:"action"`
	OperationID string `json:"operation_id"`
}

type MemoryRevisionPage struct {
	Revisions     []MemoryRevision `json:"revisions"`
	NextPageToken string           `json:"next_page_token"`
}

// RememberOptions configures [Client.Remember]. The zero value is valid: only
// non-empty/non-nil fields are sent, letting the server apply its defaults.
type RememberOptions struct {
	Type        MemoryType
	Subtype     string
	Importance  float64 // 0.0-1.0; default 0.5 when zero
	Tags        []string
	Metadata    map[string]any
	ContextID   string
	UserID      string
	WorkspaceID string
	Authority   *AuthorityContext
	Relations   []EntityRelationInput

	// importanceSet allows distinguishing an explicit 0 from the zero value.
	importanceSet bool
}

// WithImportance returns a copy of opts with Importance explicitly set (so a
// value of 0 is honored rather than defaulted to 0.5).
func (o RememberOptions) WithImportance(v float64) RememberOptions {
	o.Importance = v
	o.importanceSet = true
	return o
}

// Remember stores a new memory.
func (c *Client) Remember(ctx context.Context, content string, opts RememberOptions) (*Memory, error) {
	importance := opts.Importance
	if !opts.importanceSet && importance == 0 {
		importance = 0.5
	}
	payload := map[string]any{"content": content, "importance": importance}
	if opts.Type != "" {
		payload["type"] = opts.Type
	}
	if opts.Subtype != "" {
		payload["subtype"] = opts.Subtype
	}
	if len(opts.Tags) > 0 {
		payload["tags"] = opts.Tags
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = opts.Metadata
	}
	if len(opts.Relations) > 0 {
		payload["relations"] = opts.Relations
	}
	if opts.ContextID != "" {
		payload["context_id"] = opts.ContextID
	}
	if opts.UserID != "" {
		payload["user_id"] = opts.UserID
	}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}

	var out struct {
		Memory Memory `json:"memory"`
	}
	if err := c.requestJSON(ctx, http.MethodPost, "/memories", payload,
		requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out.Memory, nil
}

// CreateMemoryVersioned conditionally creates one keyed semantic memory.
func (c *Client) CreateMemoryVersioned(ctx context.Context, input SemanticMemoryCreateInput, opts MemoryMutationOptions) (*MemoryMutationResult, error) {
	payload := map[string]any{}
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(encoded, &payload); err != nil {
		return nil, err
	}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return c.mutateMemory(ctx, requestSpec{
		method: http.MethodPost, path: "/memories", body: body, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, "If-None-Match", "*"), authority: opts.Authority,
	})
}

// ReplaceSemanticMemory atomically replaces the refinement-owned document.
func (c *Client) ReplaceSemanticMemory(ctx context.Context, memoryID string, input SemanticMemoryReplaceInput, opts MemoryMutationOptions) (*MemoryMutationResult, error) {
	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return c.mutateMemory(ctx, requestSpec{
		method: http.MethodPut, path: "/memories/" + url.PathEscape(memoryID) + "/semantic",
		query: memoryWorkspaceQuery(c.resolveWorkspace(opts.WorkspaceID)), body: body, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// DeleteMemoryVersioned conditionally writes a durable semantic tombstone.
func (c *Client) DeleteMemoryVersioned(ctx context.Context, memoryID string, opts MemoryMutationOptions) (*MemoryMutationResult, error) {
	return c.mutateMemory(ctx, requestSpec{
		method: http.MethodPost, path: "/memories/" + url.PathEscape(memoryID) + "/delete",
		query:   memoryWorkspaceQuery(c.resolveWorkspace(opts.WorkspaceID)),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// RestoreMemoryVersioned conditionally reactivates the exact tombstoned head.
func (c *Client) RestoreMemoryVersioned(ctx context.Context, memoryID string, opts MemoryMutationOptions) (*MemoryMutationResult, error) {
	return c.mutateMemory(ctx, requestSpec{
		method: http.MethodPost, path: "/memories/" + url.PathEscape(memoryID) + "/restore",
		query:   memoryWorkspaceQuery(c.resolveWorkspace(opts.WorkspaceID)),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// MemoryHistory returns immutable semantic revisions newest first.
func (c *Client) MemoryHistory(ctx context.Context, memoryID string, opts MemoryHistoryOptions) (*MemoryRevisionPage, error) {
	q := promptNotePageQuery(c.resolveWorkspace(opts.WorkspaceID), opts.Limit, opts.PageToken)
	var out MemoryRevisionPage
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: "/memories/" + url.PathEscape(memoryID) + "/revisions",
		query: q, authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) mutateMemory(ctx context.Context, spec requestSpec) (*MemoryMutationResult, error) {
	var out struct {
		Memory   Memory `json:"memory"`
		Replayed bool   `json:"replayed"`
	}
	resp, err := c.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Memory.ETag
	}
	return &MemoryMutationResult{Memory: out.Memory, Replayed: out.Replayed, ETag: etag}, nil
}

func memoryWorkspaceQuery(workspaceID string) url.Values {
	q := url.Values{}
	if workspaceID != "" {
		q.Set("workspace_id", workspaceID)
	}
	return q
}

// RecallOptions configures [Client.Recall]. Pointer fields left nil are omitted
// so the server applies its own default.
type RecallOptions struct {
	Types               []MemoryType
	Subtypes            []string
	Tags                []string
	Mode                RecallMode
	Limit               int // default 10 when <= 0
	MinRelevance        *float64
	RecencyWeight       *float64
	Tolerance           SearchTolerance
	IncludeAssociations *bool
	TraverseDepth       *int
	MaxExpansion        *int
	CreatedAfter        string
	CreatedBefore       string
	Offset              *int
	EventAfter          string
	EventBefore         string
	TimeOrder           string
	IncludeGlobal       *bool
	IncludeGlobalUser   *bool
	UserID              string
	WorkspaceID         string
	Authority           *AuthorityContext
	BudgetTokens        *int
	IncludeConfidence   *bool
	IncludeRelations    *bool
}

// Recall searches memories by semantic query.
func (c *Client) Recall(ctx context.Context, query string, opts RecallOptions) (*RecallResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	payload := map[string]any{"query": query, "limit": limit}
	if opts.Mode != "" {
		payload["mode"] = opts.Mode
	}
	if opts.Tolerance != "" {
		payload["tolerance"] = opts.Tolerance
	}
	if opts.IncludeAssociations != nil {
		payload["include_associations"] = *opts.IncludeAssociations
	}
	if opts.TraverseDepth != nil {
		payload["traverse_depth"] = *opts.TraverseDepth
	}
	if opts.MinRelevance != nil {
		payload["min_relevance"] = *opts.MinRelevance
	}
	if opts.RecencyWeight != nil {
		payload["recency_weight"] = *opts.RecencyWeight
	}
	if opts.MaxExpansion != nil {
		payload["max_expansion"] = *opts.MaxExpansion
	}
	if len(opts.Types) > 0 {
		payload["types"] = opts.Types
	}
	if len(opts.Subtypes) > 0 {
		payload["subtypes"] = opts.Subtypes
	}
	if len(opts.Tags) > 0 {
		payload["tags"] = opts.Tags
	}
	if opts.CreatedAfter != "" {
		payload["created_after"] = opts.CreatedAfter
	}
	if opts.CreatedBefore != "" {
		payload["created_before"] = opts.CreatedBefore
	}
	if opts.Offset != nil {
		payload["offset"] = *opts.Offset
	}
	if opts.EventAfter != "" {
		payload["event_after"] = opts.EventAfter
	}
	if opts.EventBefore != "" {
		payload["event_before"] = opts.EventBefore
	}
	if opts.TimeOrder != "" {
		payload["time_order"] = opts.TimeOrder
	}
	if opts.IncludeGlobal != nil {
		payload["include_global"] = *opts.IncludeGlobal
	}
	if opts.IncludeGlobalUser != nil {
		payload["include_global_user"] = *opts.IncludeGlobalUser
	}
	if opts.BudgetTokens != nil {
		payload["budget_tokens"] = *opts.BudgetTokens
	}
	if opts.IncludeConfidence != nil {
		payload["include_confidence"] = *opts.IncludeConfidence
	}
	if opts.IncludeRelations != nil {
		payload["include_relations"] = *opts.IncludeRelations
	}
	if opts.UserID != "" {
		payload["user_id"] = opts.UserID
	}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}

	var out RecallResult
	if err := c.requestJSON(ctx, http.MethodPost, "/memories/recall", payload,
		requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	if out.TotalCount == 0 {
		out.TotalCount = len(out.Memories)
	}
	return &out, nil
}

// ReflectOptions configures [Client.Reflect].
type ReflectOptions struct {
	MaxTokens      int   // default 500 when <= 0
	IncludeSources *bool // default true
	Authority      *AuthorityContext
}

// Reflect synthesizes and summarizes memories.
func (c *Client) Reflect(ctx context.Context, query string, opts ReflectOptions) (*ReflectResult, error) {
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 500
	}
	includeSources := true
	if opts.IncludeSources != nil {
		includeSources = *opts.IncludeSources
	}
	payload := map[string]any{
		"query":           query,
		"max_tokens":      maxTokens,
		"include_sources": includeSources,
	}
	var out ReflectResult
	if err := c.requestJSON(ctx, http.MethodPost, "/memories/reflect", payload,
		requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Forget deletes (hard) or soft-deletes a memory. Returns nil on success.
func (c *Client) Forget(ctx context.Context, memoryID string, hard bool) error {
	q := url.Values{"hard": {boolStr(hard)}}
	return c.doJSON(ctx, requestSpec{
		method: http.MethodDelete,
		path:   "/memories/" + memoryID,
		query:  q,
	}, nil)
}

// GetMemoryOptions configures [Client.GetMemory].
type GetMemoryOptions struct {
	// Authority applies a per-call OBO grant, overriding the client default —
	// required when reading on behalf of a user through the Aether proxy, whose
	// ACL check authorizes the request against the subject's grant (mirrors
	// [RecallOptions.Authority]).
	Authority      *AuthorityContext
	WorkspaceID    string
	IncludeDeleted bool
}

// GetMemory fetches a single memory by ID. An optional [GetMemoryOptions] may
// carry a per-call OBO authority; without it the client's default authority (if
// any) applies.
func (c *Client) GetMemory(ctx context.Context, memoryID string, opts ...GetMemoryOptions) (*Memory, error) {
	var o GetMemoryOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	var out struct {
		Memory Memory `json:"memory"`
	}
	q := url.Values{}
	if ws := c.resolveWorkspace(o.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if o.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	resp, err := c.doJSONResponse(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/memories/" + memoryID,
		query:     q,
		authority: o.Authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	if out.Memory.ETag == "" {
		out.Memory.ETag = resp.Header.Get("ETag")
	}
	return &out.Memory, nil
}

// UpdateMemoryOptions configures [Client.UpdateMemory]. Nil fields are left
// unchanged.
type UpdateMemoryOptions struct {
	Content    *string
	Importance *float64
	Tags       []string
	Metadata   map[string]any
}

// UpdateMemory updates an existing memory.
func (c *Client) UpdateMemory(ctx context.Context, memoryID string, opts UpdateMemoryOptions) (*Memory, error) {
	payload := map[string]any{}
	if opts.Content != nil {
		payload["content"] = *opts.Content
	}
	if opts.Importance != nil {
		payload["importance"] = *opts.Importance
	}
	if opts.Tags != nil {
		payload["tags"] = opts.Tags
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	var out struct {
		Memory Memory `json:"memory"`
	}
	if err := c.requestJSON(ctx, http.MethodPut, "/memories/"+memoryID, payload,
		requestSpec{}, &out); err != nil {
		return nil, err
	}
	return &out.Memory, nil
}

// ListMemoriesOptions configures [Client.ListMemories].
type ListMemoriesOptions struct {
	Limit     int // default 50 when <= 0
	Offset    int
	Type      MemoryType
	Subtype   string
	Tag       string
	ContextID string
	Authority *AuthorityContext
}

// ListMemories lists/browses memories ordered by recency (no vector search).
//
// The returned TotalCount reflects the size of the returned page, not a grand
// total — use [Client.IterateMemories] to walk all pages.
func (c *Client) ListMemories(ctx context.Context, opts ListMemoriesOptions) (*RecallResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{
		"limit":  {strconv.Itoa(limit)},
		"offset": {strconv.Itoa(opts.Offset)},
	}
	if opts.Type != "" {
		q.Set("type", string(opts.Type))
	}
	if opts.Subtype != "" {
		q.Set("subtype", opts.Subtype)
	}
	if opts.Tag != "" {
		q.Set("tag", opts.Tag)
	}
	if opts.ContextID != "" {
		q.Set("context_id", opts.ContextID)
	}
	var out RecallResult
	if err := c.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/memories",
		query:     q,
		authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	if out.TotalCount == 0 {
		out.TotalCount = len(out.Memories)
	}
	return &out, nil
}

// IterateMemories auto-paginates [Client.ListMemories], invoking fn for every
// memory. Returning a non-nil error from fn stops iteration and propagates it.
// pageSize is clamped to [1, 200].
func (c *Client) IterateMemories(ctx context.Context, pageSize int, opts ListMemoriesOptions, fn func(Memory) error) error {
	if pageSize <= 0 {
		pageSize = 50
	}
	if pageSize > 200 {
		pageSize = 200
	}
	offset := 0
	for {
		opts.Limit = pageSize
		opts.Offset = offset
		page, err := c.ListMemories(ctx, opts)
		if err != nil {
			return err
		}
		for _, m := range page.Memories {
			if err := fn(m); err != nil {
				return err
			}
		}
		if len(page.Memories) < pageSize {
			return nil
		}
		offset += pageSize
	}
}

// BatchMemories performs multiple memory operations in a single request. Each
// operation is a map like {"type": "create", "data": {...}}. The raw result map
// (per-operation success/error) is returned.
func (c *Client) BatchMemories(ctx context.Context, operations []map[string]any) (map[string]any, error) {
	payload := map[string]any{"operations": operations}
	var out map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/memories/batch", payload,
		requestSpec{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Decay reduces a memory's importance by decayRate (0.0-1.0).
func (c *Client) Decay(ctx context.Context, memoryID string, decayRate float64) (*Memory, error) {
	payload := map[string]any{"decay_rate": decayRate}
	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPost,
		fmt.Sprintf("/memories/%s/decay", memoryID), payload, requestSpec{}, &raw); err != nil {
		return nil, err
	}
	return decodeMemoryEnvelope(raw)
}
