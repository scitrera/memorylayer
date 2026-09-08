package memorylayer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// CreateWorkspace creates a new workspace. tags may be nil.
func (c *Client) CreateWorkspace(ctx context.Context, name string, tags []string) (*Workspace, error) {
	payload := map[string]any{"name": name}
	if tags != nil {
		payload["tags"] = tags
	}
	var out Workspace
	if err := c.requestJSON(ctx, http.MethodPost, "/workspaces", payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListWorkspaces lists workspaces, optionally filtered by tag. match is "all"
// (default) — require every tag — or "any".
func (c *Client) ListWorkspaces(ctx context.Context, tags []string, match string) ([]Workspace, error) {
	q := url.Values{}
	if len(tags) > 0 {
		for _, t := range tags {
			q.Add("tags", t)
		}
		if match == "" {
			match = "all"
		}
		q.Set("match", match)
	}
	var out struct {
		Workspaces []Workspace `json:"workspaces"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces",
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return out.Workspaces, nil
}

// GetWorkspace fetches workspace details (defaults to the client workspace when
// workspaceID is empty).
func (c *Client) GetWorkspace(ctx context.Context, workspaceID string) (*Workspace, error) {
	ws := c.resolveWorkspace(workspaceID)
	if ws == "" {
		return nil, errors.New("memorylayer: workspace_id must be provided or set on client")
	}
	var out Workspace
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces/" + ws,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// UpdateWorkspaceOptions configures [Client.UpdateWorkspace]. Nil fields are
// left unchanged.
type UpdateWorkspaceOptions struct {
	Name     *string
	Settings map[string]any
	Tags     []string
}

// UpdateWorkspace updates an existing workspace.
func (c *Client) UpdateWorkspace(ctx context.Context, workspaceID string, opts UpdateWorkspaceOptions) (*Workspace, error) {
	payload := map[string]any{}
	if opts.Name != nil {
		payload["name"] = *opts.Name
	}
	if opts.Settings != nil {
		payload["settings"] = opts.Settings
	}
	if opts.Tags != nil {
		payload["tags"] = opts.Tags
	}
	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPut, "/workspaces/"+workspaceID, payload, requestSpec{}, &raw); err != nil {
		return nil, err
	}
	return remapJSON[Workspace](envelopeOr(raw, "workspace"))
}

// CreateContext creates a context within a workspace. The raw created context is
// returned.
func (c *Client) CreateContext(ctx context.Context, workspaceID, name, description string, settings map[string]any) (map[string]any, error) {
	payload := map[string]any{"name": name}
	if description != "" {
		payload["description"] = description
	}
	if settings != nil {
		payload["settings"] = settings
	}
	var raw map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/workspaces/"+workspaceID+"/contexts", payload, requestSpec{}, &raw); err != nil {
		return nil, err
	}
	if v, ok := raw["context"].(map[string]any); ok {
		return v, nil
	}
	return raw, nil
}

// ListContexts lists all contexts in a workspace as raw maps.
func (c *Client) ListContexts(ctx context.Context, workspaceID string) ([]map[string]any, error) {
	var out struct {
		Contexts []map[string]any `json:"contexts"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces/" + workspaceID + "/contexts",
	}, &out); err != nil {
		return nil, err
	}
	return out.Contexts, nil
}

// DeleteContext deletes a context from a workspace (defaults to the client
// workspace when workspaceID is empty).
func (c *Client) DeleteContext(ctx context.Context, contextID, workspaceID string) error {
	ws := c.resolveWorkspace(workspaceID)
	if ws == "" {
		return errors.New("memorylayer: workspace_id must be provided or set on the client")
	}
	return c.doJSON(ctx, requestSpec{
		method: http.MethodDelete,
		path:   fmt.Sprintf("/workspaces/%s/contexts/%s", ws, contextID),
	}, nil)
}

// GetWorkspaceSchema returns a workspace's schema (relationship_types,
// memory_subtypes, can_customize).
func (c *Client) GetWorkspaceSchema(ctx context.Context, workspaceID string) (map[string]any, error) {
	var out map[string]any
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces/" + workspaceID + "/schema",
	}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ExportWorkspaceOptions configures [Client.ExportWorkspace] and the streaming
// variant.
type ExportWorkspaceOptions struct {
	WorkspaceID         string
	IncludeAssociations *bool // default true
	Offset              int
	Limit               int // 0 = all
}

func (o ExportWorkspaceOptions) query() url.Values {
	includeAssoc := true
	if o.IncludeAssociations != nil {
		includeAssoc = *o.IncludeAssociations
	}
	return url.Values{
		"include_associations": {boolStr(includeAssoc)},
		"offset":               {strconv.Itoa(o.Offset)},
		"limit":                {strconv.Itoa(o.Limit)},
	}
}

// ExportWorkspace exports a workspace's memories and associations, parsing the
// server's NDJSON stream into a single [WorkspaceExport].
func (c *Client) ExportWorkspace(ctx context.Context, opts ExportWorkspaceOptions) (*WorkspaceExport, error) {
	ws := c.resolveWorkspace(opts.WorkspaceID)
	if ws == "" {
		return nil, errors.New("memorylayer: workspace ID required")
	}
	resp, err := c.doRaw(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces/" + ws + "/export",
		query:  opts.query(),
	})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}

	export := &WorkspaceExport{Version: "1.0", WorkspaceID: ws}
	var header map[string]any
	for _, line := range splitNDJSON(resp.Body) {
		var parsed map[string]any
		if err := json.Unmarshal(line, &parsed); err != nil {
			return nil, err
		}
		switch parsed["type"] {
		case "header":
			header = parsed
		case "memory":
			if d, ok := parsed["data"].(map[string]any); ok {
				export.Memories = append(export.Memories, d)
			}
		case "association":
			if d, ok := parsed["data"].(map[string]any); ok {
				export.Associations = append(export.Associations, d)
			}
		}
	}
	if header != nil {
		if v, ok := header["version"].(string); ok {
			export.Version = v
		}
		if v, ok := header["workspace_id"].(string); ok {
			export.WorkspaceID = v
		}
		if v, ok := header["exported_at"].(string); ok {
			export.ExportedAt = v
		}
	}
	export.TotalMemories = len(export.Memories)
	export.TotalAssociations = len(export.Associations)
	return export, nil
}

// ExportWorkspaceStream exports a workspace as NDJSON, invoking fn for each
// parsed line (header, memory, association, footer). Returning a non-nil error
// from fn stops iteration.
func (c *Client) ExportWorkspaceStream(ctx context.Context, opts ExportWorkspaceOptions, fn func(map[string]any) error) error {
	ws := c.resolveWorkspace(opts.WorkspaceID)
	if ws == "" {
		return errors.New("memorylayer: workspace ID required")
	}
	resp, err := c.doRaw(ctx, requestSpec{
		method: http.MethodGet,
		path:   "/workspaces/" + ws + "/export",
		query:  opts.query(),
	})
	if err != nil {
		return err
	}
	if err := mapError(resp, ""); err != nil {
		return err
	}
	for _, line := range splitNDJSON(resp.Body) {
		var parsed map[string]any
		if err := json.Unmarshal(line, &parsed); err != nil {
			return err
		}
		if err := fn(parsed); err != nil {
			return err
		}
	}
	return nil
}

// ImportWorkspace imports memories and associations into a workspace from a
// previously exported document. The raw import result (counts) is returned.
func (c *Client) ImportWorkspace(ctx context.Context, workspaceID string, data any) (map[string]any, error) {
	payload := map[string]any{"data": data}
	var out map[string]any
	if err := c.requestJSON(ctx, http.MethodPost, "/workspaces/"+workspaceID+"/import", payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ImportWorkspaceStream imports a workspace from NDJSON lines (each serialized
// to one line). The raw import result (counts) is returned.
func (c *Client) ImportWorkspaceStream(ctx context.Context, workspaceID string, lines []map[string]any) (map[string]any, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, line := range lines {
		if err := enc.Encode(line); err != nil { // Encode appends '\n'
			return nil, err
		}
	}
	body := bytes.TrimRight(buf.Bytes(), "\n")

	resp, err := c.doRaw(ctx, requestSpec{
		method:      http.MethodPost,
		path:        "/workspaces/" + workspaceID + "/import",
		body:        body,
		contentType: "application/x-ndjson",
	})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}
	var out map[string]any
	if len(resp.Body) > 0 {
		if err := json.Unmarshal(resp.Body, &out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// splitNDJSON splits a buffered NDJSON body into non-empty trimmed lines.
func splitNDJSON(body []byte) [][]byte {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil
	}
	var lines [][]byte
	for _, raw := range bytes.Split(trimmed, []byte("\n")) {
		if l := bytes.TrimSpace(raw); len(l) > 0 {
			lines = append(lines, l)
		}
	}
	return lines
}
