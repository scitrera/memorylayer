package memorylayer

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
)

// All context-sandbox operations require an active session (call SetSession
// first). They map to the server's /v1/context/* endpoints.

// ContextExecOptions configures [Client.ContextExec].
type ContextExecOptions struct {
	ResultVar      string
	ReturnResult   *bool // default true
	MaxReturnChars int   // default 10000 when <= 0
}

// ContextExec executes Python code in the session's sandbox. The raw result
// (output, result, error, variables_changed) is returned.
func (c *Client) ContextExec(ctx context.Context, code string, opts ContextExecOptions) (map[string]any, error) {
	returnResult := true
	if opts.ReturnResult != nil {
		returnResult = *opts.ReturnResult
	}
	maxReturnChars := opts.MaxReturnChars
	if maxReturnChars <= 0 {
		maxReturnChars = 10000
	}
	payload := map[string]any{
		"code":             code,
		"return_result":    returnResult,
		"max_return_chars": maxReturnChars,
	}
	if opts.ResultVar != "" {
		payload["result_var"] = opts.ResultVar
	}
	return c.contextPost(ctx, "/context/execute", payload)
}

// ContextInspect inspects sandbox state or a specific variable (all when
// variable is empty). previewChars defaults to 200 when <= 0.
func (c *Client) ContextInspect(ctx context.Context, variable string, previewChars int) (map[string]any, error) {
	if previewChars <= 0 {
		previewChars = 200
	}
	q := url.Values{"preview_chars": {strconv.Itoa(previewChars)}}
	if variable != "" {
		q.Set("variable", variable)
	}
	var out map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodPost,
		path:   "/context/inspect",
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ContextLoadOptions configures [Client.ContextLoad].
type ContextLoadOptions struct {
	Limit             int // default 50 when <= 0
	Types             []string
	Tags              []string
	MinRelevance      *float64
	IncludeEmbeddings bool
}

// ContextLoad runs a recall query and stores the results into a sandbox
// variable. The raw result (count, variable info) is returned.
func (c *Client) ContextLoad(ctx context.Context, varName, query string, opts ContextLoadOptions) (map[string]any, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	payload := map[string]any{
		"var":                varName,
		"query":              query,
		"limit":              limit,
		"include_embeddings": opts.IncludeEmbeddings,
	}
	if opts.Types != nil {
		payload["types"] = opts.Types
	}
	if opts.Tags != nil {
		payload["tags"] = opts.Tags
	}
	if opts.MinRelevance != nil {
		payload["min_relevance"] = *opts.MinRelevance
	}
	return c.contextPost(ctx, "/context/load", payload)
}

// ContextInject injects a value into the sandbox state. When parseJSON is true,
// a string value is parsed as JSON server-side.
func (c *Client) ContextInject(ctx context.Context, key string, value any, parseJSON bool) (map[string]any, error) {
	payload := map[string]any{"key": key, "value": value, "parse_json": parseJSON}
	return c.contextPost(ctx, "/context/inject", payload)
}

// ContextQueryOptions configures [Client.ContextQuery].
type ContextQueryOptions struct {
	MaxContextChars *int
	ResultVar       string
}

// ContextQuery sends sandbox variables and a prompt to the LLM. The raw result
// (response text, token usage) is returned.
func (c *Client) ContextQuery(ctx context.Context, prompt string, variables []string, opts ContextQueryOptions) (map[string]any, error) {
	payload := map[string]any{"prompt": prompt, "variables": variables}
	if opts.MaxContextChars != nil {
		payload["max_context_chars"] = *opts.MaxContextChars
	}
	if opts.ResultVar != "" {
		payload["result_var"] = opts.ResultVar
	}
	return c.contextPost(ctx, "/context/query", payload)
}

// ContextRLMOptions configures [Client.ContextRLM].
type ContextRLMOptions struct {
	MemoryQuery   string
	MemoryLimit   int // default 100 when <= 0
	MaxIterations int // default 10 when <= 0
	Variables     []string
	ResultVar     string
	DetailLevel   string // "brief", "standard" (default), or "detailed"
}

// ContextRLM runs a Recursive Language Model loop toward a goal. The raw result
// (result, iterations, trace) is returned.
func (c *Client) ContextRLM(ctx context.Context, goal string, opts ContextRLMOptions) (map[string]any, error) {
	memoryLimit := opts.MemoryLimit
	if memoryLimit <= 0 {
		memoryLimit = 100
	}
	maxIterations := opts.MaxIterations
	if maxIterations <= 0 {
		maxIterations = 10
	}
	detailLevel := opts.DetailLevel
	if detailLevel == "" {
		detailLevel = "standard"
	}
	payload := map[string]any{
		"goal":           goal,
		"memory_limit":   memoryLimit,
		"max_iterations": maxIterations,
		"detail_level":   detailLevel,
	}
	if opts.MemoryQuery != "" {
		payload["memory_query"] = opts.MemoryQuery
	}
	if opts.Variables != nil {
		payload["variables"] = opts.Variables
	}
	if opts.ResultVar != "" {
		payload["result_var"] = opts.ResultVar
	}
	return c.contextPost(ctx, "/context/rlm", payload)
}

// ContextStatus returns the status of the session's sandbox environment.
func (c *Client) ContextStatus(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/context/status",
	}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ContextCheckpoint checkpoints the session's sandbox state for persistence.
func (c *Client) ContextCheckpoint(ctx context.Context) error {
	return c.doJSON(ctx, requestSpec{
		method: http.MethodPost,
		path:   "/context/checkpoint",
	}, nil)
}

// ContextCleanup removes the session's sandbox environment.
func (c *Client) ContextCleanup(ctx context.Context) error {
	return c.doJSON(ctx, requestSpec{
		method: http.MethodDelete,
		path:   "/context/cleanup",
	}, nil)
}

func (c *Client) contextPost(ctx context.Context, path string, payload map[string]any) (map[string]any, error) {
	var out map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, path, payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}
