package memorylayer

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
)

const entityRegistryFeature = "Entity registry"

// ListEntitiesOptions configures [Client.ListEntities].
type ListEntitiesOptions struct {
	Status      string // "active" (default) or "merged"
	Limit       int    // default 100 when <= 0
	WorkspaceID string
	Authority   *AuthorityContext
}

// ListEntities lists canonical entities in a workspace (ordered by id). Requires
// the server-side entity registry; otherwise returns an EnterpriseRequiredError.
func (c *Client) ListEntities(ctx context.Context, opts ListEntitiesOptions) ([]Entity, error) {
	status := opts.Status
	if status == "" {
		status = "active"
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	q := url.Values{"status": {status}, "limit": {strconv.Itoa(limit)}}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out struct {
		Entities []Entity `json:"entities"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/entities",
		query:             q,
		authority:         opts.Authority,
		enterpriseFeature: entityRegistryFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Entities, nil
}

// GetEntity fetches a single canonical entity by ID.
func (c *Client) GetEntity(ctx context.Context, entityID, workspaceID string, authority *AuthorityContext) (*Entity, error) {
	q := url.Values{}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out struct {
		Entity Entity `json:"entity"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/entities/" + entityID,
		query:             q,
		authority:         authority,
		enterpriseFeature: entityRegistryFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out.Entity, nil
}

// ResolveEntity resolves a surface name/alias to an existing canonical entity.
// It never creates an entity, and returns (nil, nil) when no entity matched.
// entityType defaults to "person" when empty.
func (c *Client) ResolveEntity(ctx context.Context, name, entityType, workspaceID string, authority *AuthorityContext) (*EntityResolution, error) {
	if entityType == "" {
		entityType = "person"
	}
	q := url.Values{"name": {name}, "entity_type": {entityType}}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out struct {
		Resolution EntityResolution `json:"resolution"`
	}
	err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/entities/resolve",
		query:             q,
		authority:         authority,
		enterpriseFeature: entityRegistryFeature,
	}, &out)
	if err != nil {
		if IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return &out.Resolution, nil
}

// MergeEntities merges sourceID into targetID and returns the surviving entity.
func (c *Client) MergeEntities(ctx context.Context, sourceID, targetID, reason, workspaceID string, authority *AuthorityContext) (*Entity, error) {
	payload := map[string]any{
		"source_id": sourceID,
		"target_id": targetID,
		"reason":    reason,
	}
	q := url.Values{}
	if ws := c.resolveWorkspace(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out struct {
		Entity Entity `json:"entity"`
	}
	if err := c.requestJSON(ctx, http.MethodPost, "/entities/merge", payload, requestSpec{
		query:             q,
		authority:         authority,
		enterpriseFeature: entityRegistryFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out.Entity, nil
}
