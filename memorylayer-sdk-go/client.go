package memorylayer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// DefaultBaseURL is the base URL used when none is supplied via [WithBaseURL].
const DefaultBaseURL = "http://localhost:61001"

// Client is the Go client for the MemoryLayer.ai API. It is safe for concurrent
// use. Construct one with [NewClient] and release it with [Client.Close].
//
// The sub-namespaces Skills, McpServers, PromptNotes, AgentSpecifications,
// RefinementRecords, Kb, and Rpg expose the corresponding
// server surfaces (client.Skills.List(...), client.Kb.Generate(...), etc.).
type Client struct {
	transport        Transport
	baseURL          string
	apiKey           string
	workspaceID      string
	defaultAuthority *AuthorityContext

	mu        sync.RWMutex
	sessionID string

	// Skills exposes the /v1/skills surface.
	Skills *SkillsAPI
	// McpServers exposes the /v1/mcp-servers surface.
	McpServers *McpServersAPI
	// Kb exposes the /v1/knowledgebase surface.
	Kb *KnowledgebaseAPI
	// Rpg exposes the /v1/rpg Repository Planning Graph surface.
	Rpg *RpgAPI
	// PromptNotes exposes the typed /v1/prompt-notes resource surface.
	PromptNotes *PromptNotesAPI
	// AgentSpecifications exposes revisioned reusable subagent definitions.
	AgentSpecifications *AgentSpecificationsAPI
	// RefinementRecords exposes append-only continual-refinement audit records.
	RefinementRecords *RefinementRecordsAPI
	// WorkspaceViews exposes durable workspace views and observer-published state.
	WorkspaceViews *WorkspaceViewsAPI
}

type clientConfig struct {
	baseURL          string
	apiKey           string
	workspaceID      string
	sessionID        string
	timeout          time.Duration
	maxRetries       int
	httpClient       *http.Client
	transport        Transport
	defaultAuthority *AuthorityContext
}

// Option configures a [Client] in [NewClient].
type Option func(*clientConfig)

// WithBaseURL sets the API base URL (default [DefaultBaseURL]). Used only by the
// default HTTP transport; ignored when a custom transport is supplied.
func WithBaseURL(baseURL string) Option {
	return func(c *clientConfig) { c.baseURL = baseURL }
}

// WithAPIKey sets the bearer token sent as the Authorization header.
func WithAPIKey(apiKey string) Option {
	return func(c *clientConfig) { c.apiKey = apiKey }
}

// WithWorkspaceID sets the default workspace ID applied to operations that take
// one.
func WithWorkspaceID(workspaceID string) Option {
	return func(c *clientConfig) { c.workspaceID = workspaceID }
}

// WithSessionID sets the initial session ID (sent as the X-Session-ID header).
func WithSessionID(sessionID string) Option {
	return func(c *clientConfig) { c.sessionID = sessionID }
}

// WithTimeout sets the per-request timeout for the default HTTP transport
// (default 30s). Ignored when [WithHTTPClient] or [WithTransport] is supplied.
func WithTimeout(timeout time.Duration) Option {
	return func(c *clientConfig) { c.timeout = timeout }
}

// WithMaxRetries sets the maximum retry attempts for idempotent requests on
// transient failures (default 3; 0 disables). HTTP transport only.
func WithMaxRetries(maxRetries int) Option {
	return func(c *clientConfig) { c.maxRetries = maxRetries }
}

// WithHTTPClient supplies a custom *http.Client for the default HTTP transport
// (e.g. to configure TLS, proxies, or a shared connection pool).
func WithHTTPClient(hc *http.Client) Option {
	return func(c *clientConfig) { c.httpClient = hc }
}

// WithTransport supplies a custom [Transport], replacing the default HTTP
// transport. This is how the Aether transport (companion module) is installed.
func WithTransport(t Transport) Option {
	return func(c *clientConfig) { c.transport = t }
}

// WithDefaultAuthority sets the OBO authority applied to every request that does
// not override it.
func WithDefaultAuthority(a *AuthorityContext) Option {
	return func(c *clientConfig) { c.defaultAuthority = a }
}

// NewClient constructs a Client. Unless [WithTransport] is supplied, it builds
// the default HTTP transport against the configured base URL.
func NewClient(opts ...Option) (*Client, error) {
	cfg := clientConfig{
		baseURL:    DefaultBaseURL,
		timeout:    30 * time.Second,
		maxRetries: 3,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	transport := cfg.transport
	if transport == nil {
		base := strings.TrimRight(cfg.baseURL, "/") + "/v1"
		transport = newHTTPTransport(base, cfg.httpClient, cfg.timeout, cfg.maxRetries)
	}

	c := &Client{
		transport:        transport,
		baseURL:          strings.TrimRight(cfg.baseURL, "/"),
		apiKey:           cfg.apiKey,
		workspaceID:      cfg.workspaceID,
		sessionID:        cfg.sessionID,
		defaultAuthority: cfg.defaultAuthority,
	}
	c.Skills = &SkillsAPI{client: c}
	c.McpServers = &McpServersAPI{client: c}
	c.Kb = &KnowledgebaseAPI{client: c}
	c.Rpg = &RpgAPI{client: c}
	c.PromptNotes = &PromptNotesAPI{client: c}
	c.AgentSpecifications = &AgentSpecificationsAPI{client: c}
	c.RefinementRecords = &RefinementRecordsAPI{client: c}
	c.WorkspaceViews = &WorkspaceViewsAPI{client: c}
	return c, nil
}

// Close releases resources held by the client's transport.
func (c *Client) Close() error {
	return c.transport.Close()
}

// WorkspaceID returns the client's default workspace ID.
func (c *Client) WorkspaceID() string { return c.workspaceID }

// SetSession sets the active session ID. Subsequent requests include it in the
// X-Session-ID header, enabling session-based workspace resolution.
func (c *Client) SetSession(sessionID string) {
	c.mu.Lock()
	c.sessionID = sessionID
	c.mu.Unlock()
}

// ClearSession clears the active session ID.
func (c *Client) ClearSession() {
	c.mu.Lock()
	c.sessionID = ""
	c.mu.Unlock()
}

// SessionID returns the current session ID, if any.
func (c *Client) SessionID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.sessionID
}

// ----------------------------------------------------------------------------
// Request engine
// ----------------------------------------------------------------------------

// requestSpec describes a single API request assembled by a Client method.
type requestSpec struct {
	method            string
	path              string
	query             url.Values
	body              []byte
	contentType       string
	headers           map[string]string
	authority         *AuthorityContext
	enterpriseFeature string
}

// buildHeaders assembles the per-request headers: auth, session, OBO, and any
// caller-supplied extras (e.g. content-type for uploads).
func (c *Client) buildHeaders(spec requestSpec) http.Header {
	h := http.Header{}
	if c.apiKey != "" {
		h.Set("Authorization", "Bearer "+c.apiKey)
	}
	if sid := c.SessionID(); sid != "" {
		h.Set("X-Session-ID", sid)
	}
	if spec.contentType != "" {
		h.Set("Content-Type", spec.contentType)
	}
	for k, v := range c.oboHeaders(spec.authority) {
		h.Set(k, v)
	}
	for k, v := range spec.headers {
		h.Set(k, v)
	}
	return h
}

// doRaw issues a request and returns the buffered response without mapping HTTP
// status codes to typed errors. Callers that need the raw bytes (NDJSON export,
// binary downloads) use this and inspect StatusCode themselves.
func (c *Client) doRaw(ctx context.Context, spec requestSpec) (*Response, error) {
	req := &Request{
		Method: spec.method,
		Path:   spec.path,
		Query:  spec.query,
		Header: c.buildHeaders(spec),
		Body:   spec.body,
	}
	return c.transport.RoundTrip(ctx, req)
}

// doJSON issues a request, maps error statuses to typed errors, and decodes a
// successful body into out (when out is non-nil).
func (c *Client) doJSON(ctx context.Context, spec requestSpec, out any) error {
	_, err := c.doJSONResponse(ctx, spec, out)
	return err
}

// doJSONResponse is doJSON with the successful response metadata preserved.
// Resource APIs use it to expose concurrency headers such as ETag.
func (c *Client) doJSONResponse(ctx context.Context, spec requestSpec, out any) (*Response, error) {
	if spec.body != nil && spec.contentType == "" {
		spec.contentType = "application/json"
	}
	resp, err := c.doRaw(ctx, spec)
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, spec.enterpriseFeature); err != nil {
		return resp, err
	}
	if resp.StatusCode == http.StatusNoContent || len(resp.Body) == 0 || out == nil {
		return resp, nil
	}
	if err := json.Unmarshal(resp.Body, out); err != nil {
		return resp, err
	}
	return resp, nil
}

// postJSON marshals body, issues a POST/PUT/PATCH, and decodes into out.
func (c *Client) requestJSON(ctx context.Context, method, path string, body any, spec requestSpec, out any) error {
	spec.method = method
	spec.path = path
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		spec.body = b
	}
	return c.doJSON(ctx, spec, out)
}

// mapError mirrors the Python SDK's status-to-exception mapping.
func mapError(resp *Response, enterpriseFeature string) error {
	sc := resp.StatusCode
	if sc < 400 {
		return nil
	}
	switch sc {
	case http.StatusUnauthorized: // 401
		return &AuthenticationError{newError(detail(resp.Body, "Authentication failed"), sc)}
	case http.StatusForbidden: // 403
		return &AuthorizationError{newError(detail(resp.Body, "Authorization denied"), sc)}
	case http.StatusNotFound: // 404
		return &NotFoundError{newError(detail(resp.Body, "Resource not found"), sc)}
	case http.StatusConflict: // 409
		return &ConflictError{newError(detail(resp.Body, "Resource conflict"), sc)}
	case http.StatusPreconditionFailed: // 412
		return &PreconditionFailedError{newError(detail(resp.Body, "Precondition failed"), sc)}
	case http.StatusPreconditionRequired: // 428
		return &PreconditionRequiredError{newError(detail(resp.Body, "Precondition required"), sc)}
	case http.StatusNotImplemented: // 501
		if enterpriseFeature != "" {
			msg := enterpriseFeature + " requires MemoryLayer Enterprise. See https://memorylayer.ai for upgrade options."
			return &EnterpriseRequiredError{APIError: newError(msg, sc), Feature: enterpriseFeature}
		}
		return &NotFoundError{newError(detail(resp.Body, "Not implemented"), sc)}
	case http.StatusUnprocessableEntity: // 422
		return &ValidationError{newError(detail(resp.Body, "Validation error"), sc)}
	case http.StatusTooManyRequests: // 429
		return &RateLimitError{newError(detail(resp.Body, "Rate limit exceeded"), sc)}
	}
	if sc >= 500 {
		return &ServerError{newError(detail(resp.Body, "Server error"), sc)}
	}
	return newError(detail(resp.Body, "Request failed"), sc)
}

// detail extracts the FastAPI {"detail": ...} message from a response body, or
// returns fallback. The detail field may be a string or a structured object.
func detail(body []byte, fallback string) string {
	if len(body) == 0 {
		return fallback
	}
	var envelope struct {
		Detail json.RawMessage `json:"detail"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || len(envelope.Detail) == 0 {
		return fallback
	}
	var s string
	if err := json.Unmarshal(envelope.Detail, &s); err == nil && s != "" {
		return s
	}
	return string(envelope.Detail)
}

// ----------------------------------------------------------------------------
// Small helpers shared by method groups
// ----------------------------------------------------------------------------

// resolveWorkspace returns the explicit workspace ID, or the client default.
func (c *Client) resolveWorkspace(workspaceID string) string {
	if workspaceID != "" {
		return workspaceID
	}
	return c.workspaceID
}

// boolStr renders a bool as the "true"/"false" string the API expects in query
// params.
func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// Ptr returns a pointer to v. It is a convenience for setting the pointer-typed
// optional fields on option structs, e.g.
// RecallOptions{IncludeAssociations: memorylayer.Ptr(true)}.
func Ptr[T any](v T) *T { return &v }
