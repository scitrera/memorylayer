package memorylayer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"
)

// McpServerModel mirrors the server McpServer record.
type McpServerModel struct {
	ID           string            `json:"id"`
	WorkspaceID  string            `json:"workspace_id"`
	TenantID     string            `json:"tenant_id"`
	UserID       *string           `json:"user_id,omitempty"`
	Name         string            `json:"name"`
	Description  *string           `json:"description,omitempty"`
	Transport    string            `json:"transport"`
	Command      *string           `json:"command,omitempty"`
	Args         []string          `json:"args"`
	Env          map[string]string `json:"env"`
	URL          *string           `json:"url,omitempty"`
	Headers      map[string]string `json:"headers"`
	Metadata     map[string]any    `json:"metadata"`
	SourceMode   string            `json:"source_mode"`
	ManifestHash string            `json:"manifest_hash"`
	Enabled      bool              `json:"enabled"`
	CreatedAt    *string           `json:"created_at,omitempty"`
	UpdatedAt    *string           `json:"updated_at,omitempty"`
}

// McpServersAPI is the /v1/mcp-servers namespace, reached via client.McpServers.
type McpServersAPI struct {
	client *Client
}

func (m *McpServersAPI) ws(workspaceID string) string {
	return m.client.resolveWorkspace(workspaceID)
}

// ListMcpServersOptions configures [McpServersAPI.List].
type ListMcpServersOptions struct {
	Transport   string
	Enabled     *bool
	Name        string
	UserID      string
	Limit       int // default 100 when <= 0
	Offset      int
	WorkspaceID string
	Authority   *AuthorityContext
}

// List lists MCP servers with optional filters.
func (m *McpServersAPI) List(ctx context.Context, opts ListMcpServersOptions) ([]McpServerModel, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}}
	if ws := m.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.Transport != "" {
		q.Set("transport", opts.Transport)
	}
	if opts.Enabled != nil {
		q.Set("enabled", boolStr(*opts.Enabled))
	}
	if opts.Name != "" {
		q.Set("name", opts.Name)
	}
	if opts.UserID != "" {
		q.Set("user_id", opts.UserID)
	}
	var out struct {
		McpServers []McpServerModel `json:"mcp_servers"`
	}
	if err := m.client.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/mcp-servers",
		query:     q,
		authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return out.McpServers, nil
}

// Get fetches an MCP server by ID.
func (m *McpServersAPI) Get(ctx context.Context, serverID string, authority *AuthorityContext) (*McpServerModel, error) {
	var out struct {
		McpServer McpServerModel `json:"mcp_server"`
	}
	if err := m.client.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/mcp-servers/" + serverID,
		authority: authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out.McpServer, nil
}

// CreateMcpServerOptions configures [McpServersAPI.Create].
type CreateMcpServerOptions struct {
	Command     string
	Args        []string
	Env         map[string]string
	URL         string
	Headers     map[string]string
	Description string
	Metadata    map[string]any
	SourceMode  string // default "server"
	Enabled     *bool  // default true
	WorkspaceID string
	UserID      string
	Authority   *AuthorityContext
}

// Create creates a new MCP server record. transport is one of "stdio", "http",
// "sse", or "streamable-http".
func (m *McpServersAPI) Create(ctx context.Context, name, transport string, opts CreateMcpServerOptions) (*McpServerModel, error) {
	enabled := true
	if opts.Enabled != nil {
		enabled = *opts.Enabled
	}
	payload := map[string]any{
		"name":        name,
		"transport":   transport,
		"source_mode": orDefault(opts.SourceMode, "server"),
		"enabled":     enabled,
	}
	if ws := m.ws(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	if opts.UserID != "" {
		payload["user_id"] = opts.UserID
	}
	if opts.Description != "" {
		payload["description"] = opts.Description
	}
	if opts.Command != "" {
		payload["command"] = opts.Command
	}
	if opts.Args != nil {
		payload["args"] = opts.Args
	}
	if opts.Env != nil {
		payload["env"] = opts.Env
	}
	if opts.URL != "" {
		payload["url"] = opts.URL
	}
	if opts.Headers != nil {
		payload["headers"] = opts.Headers
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	var out struct {
		McpServer McpServerModel `json:"mcp_server"`
	}
	if err := m.client.requestJSON(ctx, http.MethodPost, "/mcp-servers", payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out.McpServer, nil
}

// UpdateMcpServerOptions configures [McpServersAPI.Update]. Nil/zero fields are
// omitted (unchanged).
type UpdateMcpServerOptions struct {
	Description *string
	Command     *string
	Args        []string
	Env         map[string]string
	URL         *string
	Headers     map[string]string
	Metadata    map[string]any
	SourceMode  *string
	Enabled     *bool
	Authority   *AuthorityContext
}

// Update partially updates an MCP server record.
func (m *McpServersAPI) Update(ctx context.Context, serverID string, opts UpdateMcpServerOptions) (*McpServerModel, error) {
	payload := map[string]any{}
	if opts.Description != nil {
		payload["description"] = *opts.Description
	}
	if opts.Command != nil {
		payload["command"] = *opts.Command
	}
	if opts.Args != nil {
		payload["args"] = opts.Args
	}
	if opts.Env != nil {
		payload["env"] = opts.Env
	}
	if opts.URL != nil {
		payload["url"] = *opts.URL
	}
	if opts.Headers != nil {
		payload["headers"] = opts.Headers
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	if opts.SourceMode != nil {
		payload["source_mode"] = *opts.SourceMode
	}
	if opts.Enabled != nil {
		payload["enabled"] = *opts.Enabled
	}
	var out struct {
		McpServer McpServerModel `json:"mcp_server"`
	}
	if err := m.client.requestJSON(ctx, http.MethodPut, "/mcp-servers/"+serverID, payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out.McpServer, nil
}

// Delete deletes an MCP server record.
func (m *McpServersAPI) Delete(ctx context.Context, serverID string, authority *AuthorityContext) error {
	return m.client.doJSON(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      "/mcp-servers/" + serverID,
		authority: authority,
	}, nil)
}

// Resolve resolves an MCP server by name using 4-tier precedence; returns
// (nil, nil) when not found.
func (m *McpServersAPI) Resolve(ctx context.Context, name, workspaceID string, authority *AuthorityContext) (*McpServerModel, error) {
	payload := map[string]any{"name": name}
	if ws := m.ws(workspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	var out struct {
		McpServer *McpServerModel `json:"mcp_server"`
	}
	if err := m.client.requestJSON(ctx, http.MethodPost, "/mcp-servers/resolve", payload, requestSpec{authority: authority}, &out); err != nil {
		return nil, err
	}
	return out.McpServer, nil
}

// ------------------------------------------------------------------
// .mcp.json helpers
// ------------------------------------------------------------------

type mcpJSONEntry struct {
	Transport string            `json:"transport,omitempty"`
	Command   string            `json:"command,omitempty"`
	Args      []string          `json:"args,omitempty"`
	Env       map[string]string `json:"env,omitempty"`
	URL       string            `json:"url,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type mcpJSONDoc struct {
	McpServers map[string]mcpJSONEntry `json:"mcpServers"`
}

// PushJSON pushes a .mcp.json file, creating one record per server entry.
func (m *McpServersAPI) PushJSON(ctx context.Context, jsonPath, scope, sourceMode, workspaceID, userID string, authority *AuthorityContext) ([]McpServerModel, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}
	var doc mcpJSONDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	var results []McpServerModel
	for name, entry := range doc.McpServers {
		transport := entry.Transport
		if transport == "" {
			if entry.Command != "" {
				transport = "stdio"
			} else {
				transport = "http"
			}
		}
		server, err := m.Create(ctx, name, transport, CreateMcpServerOptions{
			Command:     entry.Command,
			Args:        entry.Args,
			Env:         entry.Env,
			URL:         entry.URL,
			Headers:     entry.Headers,
			SourceMode:  sourceMode,
			WorkspaceID: workspaceID,
			UserID:      userID,
			Authority:   authority,
		})
		if err != nil {
			return nil, err
		}
		results = append(results, *server)
	}
	return results, nil
}

// PullJSON materializes MCP servers into a .mcp.json file at outPath.
func (m *McpServersAPI) PullJSON(ctx context.Context, outPath, workspaceID, userID, transport string, enabled *bool, authority *AuthorityContext) error {
	servers, err := m.List(ctx, ListMcpServersOptions{
		Transport:   transport,
		Enabled:     enabled,
		UserID:      userID,
		WorkspaceID: workspaceID,
		Authority:   authority,
	})
	if err != nil {
		return err
	}
	doc := mcpJSONDoc{McpServers: map[string]mcpJSONEntry{}}
	for _, s := range servers {
		entry := mcpJSONEntry{Transport: s.Transport}
		if s.Transport == "stdio" {
			if s.Command != nil {
				entry.Command = *s.Command
			}
			entry.Args = s.Args
			entry.Env = s.Env
		} else {
			if s.URL != nil {
				entry.URL = *s.URL
			}
			entry.Headers = s.Headers
		}
		doc.McpServers[s.Name] = entry
	}
	body, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outPath, body, 0o644)
}
