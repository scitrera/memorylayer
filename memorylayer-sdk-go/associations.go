package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
)

// AssociateOptions configures [Client.Associate].
type AssociateOptions struct {
	Strength float64 // 0.0-1.0; default 0.5 when zero
	Metadata map[string]any

	strengthSet bool
}

// WithStrength returns a copy of opts with Strength explicitly set.
func (o AssociateOptions) WithStrength(v float64) AssociateOptions {
	o.Strength = v
	o.strengthSet = true
	return o
}

// Associate links two memories with a relationship.
func (c *Client) Associate(ctx context.Context, sourceID, targetID string, relationship RelationshipType, opts AssociateOptions) (*Association, error) {
	strength := opts.Strength
	if !opts.strengthSet && strength == 0 {
		strength = 0.5
	}
	payload := map[string]any{
		"target_id":    targetID,
		"relationship": relationship,
		"strength":     strength,
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = opts.Metadata
	}
	var out Association
	if err := c.requestJSON(ctx, http.MethodPost,
		fmt.Sprintf("/memories/%s/associate", sourceID), payload, requestSpec{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetAssociations returns associations for a memory. direction is "outgoing",
// "incoming", or "both" (default "both" when empty).
func (c *Client) GetAssociations(ctx context.Context, memoryID, direction string) ([]Association, error) {
	if direction == "" {
		direction = "both"
	}
	q := url.Values{"direction": {direction}}
	var out struct {
		Associations []Association `json:"associations"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method: http.MethodGet,
		path:   fmt.Sprintf("/memories/%s/associations", memoryID),
		query:  q,
	}, &out); err != nil {
		return nil, err
	}
	return out.Associations, nil
}

// UpdateAssociationOptions configures [Client.UpdateAssociation]. Nil fields are
// left unchanged.
type UpdateAssociationOptions struct {
	Strength  *float64
	Metadata  map[string]any
	Authority *AuthorityContext
}

// UpdateAssociation updates an association's strength and/or metadata. memoryID
// must be one of the edge's endpoints. Returns false if the association was not
// found.
func (c *Client) UpdateAssociation(ctx context.Context, memoryID, associationID string, opts UpdateAssociationOptions) (bool, error) {
	payload := map[string]any{}
	if opts.Strength != nil {
		payload["strength"] = *opts.Strength
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	err := c.requestJSON(ctx, http.MethodPatch,
		fmt.Sprintf("/memories/%s/associations/%s", memoryID, associationID), payload,
		requestSpec{authority: opts.Authority}, nil)
	if err != nil {
		if IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// DeleteAssociation deletes a graph edge by ID. memoryID must be one of the
// edge's endpoints. Returns false if the association was not found.
func (c *Client) DeleteAssociation(ctx context.Context, memoryID, associationID string, authority *AuthorityContext) (bool, error) {
	err := c.doJSON(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      fmt.Sprintf("/memories/%s/associations/%s", memoryID, associationID),
		authority: authority,
	}, nil)
	if err != nil {
		if IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// TraverseGraphOptions configures [Client.TraverseGraph].
type TraverseGraphOptions struct {
	MaxDepth          int // default 2 when <= 0
	RelationshipTypes []string
	Direction         string // default "both" when empty
	MinStrength       float64
	WorkspaceID       string
	Authority         *AuthorityContext
}

// TraverseGraph traverses the memory graph from a starting memory. The raw graph
// result (paths, total_paths, unique_nodes, query_latency_ms) is returned.
func (c *Client) TraverseGraph(ctx context.Context, memoryID string, opts TraverseGraphOptions) (map[string]any, error) {
	maxDepth := opts.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 2
	}
	direction := opts.Direction
	if direction == "" {
		direction = "both"
	}
	payload := map[string]any{
		"max_depth":    maxDepth,
		"direction":    direction,
		"min_strength": opts.MinStrength,
	}
	if opts.RelationshipTypes != nil {
		payload["relationship_types"] = opts.RelationshipTypes
	}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	var out map[string]any
	if err := c.requestJSON(ctx, http.MethodPost,
		fmt.Sprintf("/memories/%s/traverse", memoryID), payload,
		requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return out, nil
}
