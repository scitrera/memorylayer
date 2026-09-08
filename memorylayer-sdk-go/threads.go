package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// CreateThreadOptions configures [Client.CreateThread].
type CreateThreadOptions struct {
	WorkspaceID string
	ThreadID    string
	UserID      string
	ContextID   string
	ObserverID  string
	SubjectID   string
	Title       string
	Metadata    map[string]any
	ExpiresAt   string // ISO 8601
	Scope       string
	// Ownership is "user" (default — user-owned, cross-workspace) or
	// "workspace" (legacy workspace-scoped storage).
	Ownership string
	// ParentThread, when set, records this thread as a child of another thread
	// (chat_threads.parent_thread). NULL/empty means a top-level thread. Parent is
	// create-time only — the server has no re-parent endpoint.
	ParentThread string
	// Authority applies an OBO grant to the create (e.g. writing a user-owned
	// thread into _user_chat requires the user's grant). nil uses the client's
	// default authority.
	Authority *AuthorityContext
}

// CreateThread creates a new chat thread. With ownership "user" (the default)
// the thread is stored under [UserChatHomeWorkspace] regardless of the supplied
// workspace.
func (c *Client) CreateThread(ctx context.Context, opts CreateThreadOptions) (*ChatThread, error) {
	ownership := opts.Ownership
	if ownership == "" {
		ownership = "user"
	}
	payload := map[string]any{}
	ws := c.resolveWorkspace(opts.WorkspaceID)
	if ownership == "user" {
		ws = UserChatHomeWorkspace
	}
	if ws != "" {
		payload["workspace_id"] = ws
	}
	if opts.ThreadID != "" {
		payload["id"] = opts.ThreadID
	}
	if opts.UserID != "" {
		payload["user_id"] = opts.UserID
	}
	if opts.ContextID != "" {
		payload["context_id"] = opts.ContextID
	}
	if opts.ObserverID != "" {
		payload["observer_id"] = opts.ObserverID
	}
	if opts.SubjectID != "" {
		payload["subject_id"] = opts.SubjectID
	}
	if opts.Title != "" {
		payload["title"] = opts.Title
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	if opts.ExpiresAt != "" {
		payload["expires_at"] = opts.ExpiresAt
	}
	if opts.Scope != "" {
		payload["scope"] = opts.Scope
	}
	if opts.ParentThread != "" {
		payload["parent_thread"] = opts.ParentThread
	}
	payload["ownership"] = ownership

	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/threads", payload, requestSpec{authority: opts.Authority}, &raw); err != nil {
		return nil, err
	}
	return decodeThreadEnvelope(raw)
}

// ListThreadsOptions configures [Client.ListThreads].
type ListThreadsOptions struct {
	WorkspaceID     string
	UserID          string
	Limit           int // default 50 when <= 0
	Offset          int
	ScopeFilter     string
	OwnershipFilter string
	// ParentThread filters to the CHILDREN of the given thread. Empty (the
	// default) returns top-level threads only (server treats a missing filter as
	// parent_thread IS NULL).
	ParentThread string
}

// ListThreads lists chat threads.
func (c *Client) ListThreads(ctx context.Context, opts ListThreadsOptions) ([]ChatThread, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.UserID != "" {
		q.Set("user_id", opts.UserID)
	}
	if opts.ScopeFilter != "" {
		q.Set("scope_filter", opts.ScopeFilter)
	}
	if opts.OwnershipFilter != "" {
		q.Set("ownership_filter", opts.OwnershipFilter)
	}
	if opts.ParentThread != "" {
		q.Set("parent_thread", opts.ParentThread)
	}
	return c.listThreads(ctx, "/threads", q)
}

// ListUserThreadsOptions configures [Client.ListUserThreads].
type ListUserThreadsOptions struct {
	Ownership   string // default "user"
	ScopeFilter string
	Limit       int // default 50 when <= 0
	Offset      int
}

// ListUserThreads lists threads owned by a user across all workspaces. Unlike
// ListThreads it is keyed on (tenant, user, ownership) and needs no workspace.
func (c *Client) ListUserThreads(ctx context.Context, userID string, opts ListUserThreadsOptions) ([]ChatThread, error) {
	ownership := opts.Ownership
	if ownership == "" {
		ownership = "user"
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{
		"limit":     {strconv.Itoa(limit)},
		"offset":    {strconv.Itoa(opts.Offset)},
		"ownership": {ownership},
	}
	if opts.ScopeFilter != "" {
		q.Set("scope_filter", opts.ScopeFilter)
	}
	return c.listThreads(ctx, "/threads/user/"+userID, q)
}

// listThreads decodes the {"threads": [...]} envelope (or a bare list).
func (c *Client) listThreads(ctx context.Context, path string, q url.Values) ([]ChatThread, error) {
	resp, err := c.doRaw(ctx, requestSpec{method: http.MethodGet, path: path, query: q})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}
	return decodeThreadList(resp.Body)
}

// GetThread fetches thread metadata.
func (c *Client) GetThread(ctx context.Context, threadID, workspaceID string) (*ChatThread, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var raw map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/threads/" + threadID,
		query:  q,
	}, &raw); err != nil {
		return nil, err
	}
	return decodeThreadEnvelope(raw)
}

// GetThreadFullOptions configures [Client.GetThreadFull].
type GetThreadFullOptions struct {
	WorkspaceID string
	Limit       int // default 100 when <= 0
	Offset      int
	Order       string // "asc" (default) or "desc"
}

// GetThreadFull fetches a thread with its messages inlined.
func (c *Client) GetThreadFull(ctx context.Context, threadID string, opts GetThreadFullOptions) (*ChatThreadWithMessages, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	order := opts.Order
	if order == "" {
		order = "asc"
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}, "order": {order}}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out ChatThreadWithMessages
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   fmt.Sprintf("/threads/%s/full", threadID),
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// UpdateThreadOptions configures [Client.UpdateThread]. Nil fields are unchanged.
type UpdateThreadOptions struct {
	WorkspaceID string
	Title       *string
	Metadata    map[string]any
}

// UpdateThread updates a thread's metadata (e.g. renames it).
func (c *Client) UpdateThread(ctx context.Context, threadID string, opts UpdateThreadOptions) (*ChatThread, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	payload := map[string]any{}
	if opts.Title != nil {
		payload["title"] = *opts.Title
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPut, "/threads/"+threadID, payload, requestSpec{query: q}, &raw); err != nil {
		return nil, err
	}
	return decodeThreadEnvelope(raw)
}

// DeleteThread deletes a thread and its messages.
func (c *Client) DeleteThread(ctx context.Context, threadID, workspaceID string) error {
	q := url.Values{}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	return c.doJSON(ctx, requestSpec{
		method: http.MethodDelete,
		path:   "/threads/" + threadID,
		query:  q,
	}, nil)
}

// DeleteMessageOptions configures [Client.DeleteMessage].
type DeleteMessageOptions struct {
	WorkspaceID string
	Authority   *AuthorityContext
}

// DeleteMessage deletes a single message from a thread. Returns false if the
// message was not found.
func (c *Client) DeleteMessage(ctx context.Context, threadID, messageID string, opts DeleteMessageOptions) (bool, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	err := c.doJSON(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      fmt.Sprintf("/threads/%s/messages/%s", threadID, messageID),
		query:     q,
		authority: opts.Authority,
	}, nil)
	if err != nil {
		if IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// AppendMessagesOptions configures [Client.AppendMessages].
type AppendMessagesOptions struct {
	WorkspaceID string
	// Ownership is "user" (default) or "workspace". With "user", the supplied
	// workspace is folded into each message's metadata under
	// [MessageMetaAppWorkspaceKey] and the storage workspace becomes
	// [UserChatHomeWorkspace].
	Ownership string
	Authority *AuthorityContext
}

// AppendMessages appends messages to a thread. Each message is a map with at
// least "role" and "content" (and optional "metadata").
func (c *Client) AppendMessages(ctx context.Context, threadID string, messages []map[string]any, opts AppendMessagesOptions) ([]ChatMessage, error) {
	ownership := opts.Ownership
	if ownership == "" {
		ownership = "user"
	}
	ws := c.resolveWorkspace(opts.WorkspaceID)
	storageWS := ws
	if ownership == "user" {
		if ws != "" && ws != UserChatHomeWorkspace {
			messages = withAppWorkspace(messages, ws)
		}
		storageWS = UserChatHomeWorkspace
	}

	q := url.Values{}
	if storageWS != "" {
		q.Set("workspace_id", storageWS)
	}
	payload := map[string]any{"messages": messages}
	resp, err := c.doRaw(ctx, requestSpec{
		method:      http.MethodPost,
		path:        fmt.Sprintf("/threads/%s/messages", threadID),
		query:       q,
		body:        mustMarshal(payload),
		contentType: "application/json",
		authority:   opts.Authority,
	})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}
	return decodeMessageList(resp.Body)
}

// GetMessagesOptions configures [Client.GetMessages].
type GetMessagesOptions struct {
	WorkspaceID string
	Limit       int // default 100 when <= 0
	Offset      int
	AfterIndex  *int
	Order       string // "asc" (default) or "desc"
	// Authority applies an OBO authority to this call (overriding the client
	// default). Mirrors RecallOptions/AppendMessagesOptions so thread-history
	// reads can carry the per-request user grant.
	Authority *AuthorityContext
}

// GetMessages fetches messages from a thread.
func (c *Client) GetMessages(ctx context.Context, threadID string, opts GetMessagesOptions) ([]ChatMessage, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	order := opts.Order
	if order == "" {
		order = "asc"
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}, "order": {order}}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.AfterIndex != nil {
		q.Set("after_index", strconv.Itoa(*opts.AfterIndex))
	}
	resp, err := c.doRaw(ctx, requestSpec{
		method:    http.MethodGet,
		path:      fmt.Sprintf("/threads/%s/messages", threadID),
		query:     q,
		authority: opts.Authority,
	})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}
	return decodeMessageList(resp.Body)
}

// DecomposeThread triggers memory decomposition for unprocessed messages.
func (c *Client) DecomposeThread(ctx context.Context, threadID, workspaceID string) (*DecompositionResult, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out DecompositionResult
	if err := c.requestJSON(ctx, http.MethodPost, fmt.Sprintf("/threads/%s/decompose", threadID), nil, requestSpec{query: q}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// withAppWorkspace stamps the caller's originating app workspace onto each
// message's metadata under MessageMetaAppWorkspaceKey, without overwriting an
// existing value.
func withAppWorkspace(messages []map[string]any, workspaceID string) []map[string]any {
	out := make([]map[string]any, len(messages))
	for i, m := range messages {
		nm := make(map[string]any, len(m)+1)
		for k, v := range m {
			nm[k] = v
		}
		md := map[string]any{}
		if existing, ok := nm["metadata"].(map[string]any); ok {
			for k, v := range existing {
				md[k] = v
			}
		}
		if _, ok := md[MessageMetaAppWorkspaceKey]; !ok {
			md[MessageMetaAppWorkspaceKey] = workspaceID
		}
		nm["metadata"] = md
		out[i] = nm
	}
	return out
}
