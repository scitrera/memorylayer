package memorylayer

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// CreateCheckpointOptions configures a durable raw transcript capture.
type CreateCheckpointOptions struct {
	ContentHash    string
	IdempotencyKey string
	SourceKind     string
	SourceSequence *int
	SourceBoundary *int
}

// CreateCheckpoint stores a raw pre-compaction transcript segment before any
// derived indexing or enrichment work.
func (c *Client) CreateCheckpoint(ctx context.Context, sessionID, transcriptSegment string, opts CreateCheckpointOptions) (*SessionCheckpoint, error) {
	payload := map[string]any{
		"transcript_segment": transcriptSegment,
		"content_hash":       opts.ContentHash,
		"idempotency_key":    opts.IdempotencyKey,
	}
	if opts.SourceKind != "" {
		payload["source_kind"] = opts.SourceKind
	}
	if opts.SourceSequence != nil {
		payload["source_sequence"] = *opts.SourceSequence
	}
	if opts.SourceBoundary != nil {
		payload["source_boundary"] = *opts.SourceBoundary
	}
	var out SessionCheckpoint
	if err := c.requestJSON(ctx, http.MethodPost, "/sessions/"+url.PathEscape(sessionID)+"/checkpoints", payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetCheckpoint fetches a durable session checkpoint by ID.
func (c *Client) GetCheckpoint(ctx context.Context, sessionID, checkpointID string) (*SessionCheckpoint, error) {
	var out SessionCheckpoint
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/sessions/" + url.PathEscape(sessionID) + "/checkpoints/" + url.PathEscape(checkpointID),
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ContextPackOptions configures deterministic session context assembly.
type ContextPackOptions struct {
	Topic                     string
	EntityIDs                 []string
	EntityNames               []string
	BudgetTokens              int
	SectionLimits             map[string]int
	IncludeDirectives         *bool
	IncludeWorkingMemory      *bool
	IncludeRecentActivity     *bool
	IncludeContradictions     *bool
	IncludeSandboxSummary     *bool
	IncludeCheckpointRecovery *bool
}

// GetContextPack returns a deterministic, hard-budgeted session resume pack.
func (c *Client) GetContextPack(ctx context.Context, sessionID string, opts ContextPackOptions) (*ContextPack, error) {
	budget := opts.BudgetTokens
	if budget <= 0 {
		budget = 2048
	}
	payload := map[string]any{
		"entity_ids":    opts.EntityIDs,
		"entity_names":  opts.EntityNames,
		"budget_tokens": budget,
	}
	if opts.Topic != "" {
		payload["topic"] = opts.Topic
	}
	if opts.SectionLimits != nil {
		payload["section_limits"] = opts.SectionLimits
	}
	for key, value := range map[string]*bool{
		"include_directives":          opts.IncludeDirectives,
		"include_working_memory":      opts.IncludeWorkingMemory,
		"include_recent_activity":     opts.IncludeRecentActivity,
		"include_contradictions":      opts.IncludeContradictions,
		"include_sandbox_summary":     opts.IncludeSandboxSummary,
		"include_checkpoint_recovery": opts.IncludeCheckpointRecovery,
	} {
		if value != nil {
			payload[key] = *value
		}
	}
	var out ContextPack
	if err := c.requestJSON(ctx, http.MethodPost, "/sessions/"+url.PathEscape(sessionID)+"/context-pack", payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetContextDelta returns collapsed context changes since a pack cursor.
func (c *Client) GetContextDelta(ctx context.Context, sessionID, cursor string, budgetTokens int) (*ContextDelta, error) {
	if budgetTokens <= 0 {
		budgetTokens = 2048
	}
	var out ContextDelta
	if err := c.requestJSON(
		ctx,
		http.MethodPost,
		"/sessions/"+url.PathEscape(sessionID)+"/context-delta",
		map[string]any{"cursor": cursor, "budget_tokens": budgetTokens},
		requestSpec{},
		&out,
	); err != nil {
		return nil, err
	}
	return &out, nil
}

// CreateSessionOptions configures [Client.CreateSession].
type CreateSessionOptions struct {
	TTLSeconds  int    // default 3600 when <= 0
	WorkspaceID string // auto-created if it does not exist
	ContextID   string // defaults to _default server-side
	// AutoSetSession, when true (the default), sets the created session as the
	// active session for subsequent requests. Use NoAutoSetSession to disable.
	AutoSetSession bool
	noAuto         bool
}

// NoAutoSetSession returns a copy of opts that will NOT auto-activate the
// created session.
func (o CreateSessionOptions) NoAutoSetSession() CreateSessionOptions {
	o.AutoSetSession = false
	o.noAuto = true
	return o
}

// CreateSession creates a new working-memory session. Workspaces and contexts
// are auto-created if they do not exist. By default the new session is set as
// the active session.
func (c *Client) CreateSession(ctx context.Context, opts CreateSessionOptions) (*Session, error) {
	ttl := opts.TTLSeconds
	if ttl <= 0 {
		ttl = 3600
	}
	payload := map[string]any{"ttl_seconds": ttl}
	if opts.WorkspaceID != "" {
		payload["workspace_id"] = opts.WorkspaceID
	}
	if opts.ContextID != "" {
		payload["context_id"] = opts.ContextID
	}
	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/sessions", payload, requestSpec{}, &raw); err != nil {
		return nil, err
	}
	session, err := remapJSON[Session](envelopeOr(raw, "session"))
	if err != nil {
		return nil, err
	}
	autoSet := opts.AutoSetSession || !opts.noAuto
	if autoSet && session.ID != "" {
		c.SetSession(session.ID)
	}
	return session, nil
}

// GetSession fetches a session by ID.
func (c *Client) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	var out Session
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/sessions/" + sessionID,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// SetContext sets a context value in a session. ttlSeconds <= 0 omits the TTL.
func (c *Client) SetContext(ctx context.Context, sessionID, key string, value any, ttlSeconds int) error {
	payload := map[string]any{"key": key, "value": value}
	if ttlSeconds > 0 {
		payload["ttl_seconds"] = ttlSeconds
	}
	return c.requestJSON(ctx, http.MethodPost, "/sessions/"+sessionID+"/memory", payload, requestSpec{}, nil)
}

// GetContext gets context values from a session.
func (c *Client) GetContext(ctx context.Context, sessionID string, keys []string) (map[string]any, error) {
	q := url.Values{"key": {strings.Join(keys, ",")}}
	var out map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/sessions/" + sessionID + "/memory",
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// GetBriefingOptions configures [Client.GetBriefing].
type GetBriefingOptions struct {
	// LookbackHours, when > 0, overrides LookbackMinutes (deprecated knob kept
	// for parity).
	LookbackHours         int
	LookbackMinutes       int    // default 60 when <= 0
	DetailLevel           string // "abstract" (default), "overview", or "full"
	Limit                 int    // default 10 when <= 0
	IncludeMemories       *bool  // default true
	IncludeContradictions *bool  // default true
}

// GetBriefing returns a session briefing with recent activity and context.
func (c *Client) GetBriefing(ctx context.Context, opts GetBriefingOptions) (*SessionBriefing, error) {
	lookbackMinutes := opts.LookbackMinutes
	if lookbackMinutes <= 0 {
		lookbackMinutes = 60
	}
	if opts.LookbackHours > 0 {
		lookbackMinutes = opts.LookbackHours * 60
	}
	detailLevel := opts.DetailLevel
	if detailLevel == "" {
		detailLevel = "abstract"
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	includeMemories := true
	if opts.IncludeMemories != nil {
		includeMemories = *opts.IncludeMemories
	}
	includeContradictions := true
	if opts.IncludeContradictions != nil {
		includeContradictions = *opts.IncludeContradictions
	}
	q := url.Values{
		"lookback_minutes":       {strconv.Itoa(lookbackMinutes)},
		"detail_level":           {detailLevel},
		"limit":                  {strconv.Itoa(limit)},
		"include_memories":       {boolStr(includeMemories)},
		"include_contradictions": {boolStr(includeContradictions)},
	}
	var out SessionBriefing
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/sessions/briefing",
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListSessionsOptions configures [Client.ListSessions].
type ListSessionsOptions struct {
	WorkspaceID    string
	ContextID      string
	IncludeExpired bool
}

// ListSessions lists sessions in a workspace. Each entry is returned as a raw
// map (the server's session shape varies by deployment).
func (c *Client) ListSessions(ctx context.Context, opts ListSessionsOptions) ([]map[string]any, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.ContextID != "" {
		q.Set("context_id", opts.ContextID)
	}
	if opts.IncludeExpired {
		q.Set("include_expired", "true")
	}
	var out struct {
		Sessions []map[string]any `json:"sessions"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/sessions",
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return out.Sessions, nil
}

// DeleteSession deletes a session and all its context data.
func (c *Client) DeleteSession(ctx context.Context, sessionID string) error {
	return c.doJSON(ctx, requestSpec{
		method: http.MethodDelete,
		path:   "/sessions/" + sessionID,
	}, nil)
}

// CommitSessionOptions configures [Client.CommitSession].
type CommitSessionOptions struct {
	MinImportance float64 // default 0.5 when zero (and not set)
	Deduplicate   *bool   // default true
	Categories    []string
	MaxMemories   int // default 50 when <= 0

	minImportanceSet bool
}

// WithMinImportance returns a copy with MinImportance explicitly set.
func (o CommitSessionOptions) WithMinImportance(v float64) CommitSessionOptions {
	o.MinImportance = v
	o.minImportanceSet = true
	return o
}

// CommitSession commits session working memory to long-term memory. The raw
// commit result (extraction statistics) is returned.
func (c *Client) CommitSession(ctx context.Context, sessionID string, opts CommitSessionOptions) (map[string]any, error) {
	minImportance := opts.MinImportance
	if !opts.minImportanceSet && minImportance == 0 {
		minImportance = 0.5
	}
	dedup := true
	if opts.Deduplicate != nil {
		dedup = *opts.Deduplicate
	}
	maxMemories := opts.MaxMemories
	if maxMemories <= 0 {
		maxMemories = 50
	}
	payload := map[string]any{
		"min_importance": minImportance,
		"deduplicate":    dedup,
		"max_memories":   maxMemories,
	}
	if opts.Categories != nil {
		payload["categories"] = opts.Categories
	}
	var out map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/sessions/"+sessionID+"/commit", payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// TouchSession extends a session's TTL. The raw result (updated expiration) is
// returned.
func (c *Client) TouchSession(ctx context.Context, sessionID string) (map[string]any, error) {
	var out map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodPost,
		path:   "/sessions/" + sessionID + "/touch",
	}, &out); err != nil {
		return nil, err
	}
	return out, nil
}
