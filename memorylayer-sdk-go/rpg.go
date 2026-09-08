// SPDX-License-Identifier: Apache-2.0
package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// RpgNodeInput is a repository node accepted by [RpgAPI.Sync].
type RpgNodeInput struct {
	NodeID      string         `json:"node_id"`
	NodeType    string         `json:"node_type"`
	Path        string         `json:"path"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Language    *string        `json:"language,omitempty"`
	ParentID    *string        `json:"parent_id,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// RpgEdgeInput is a repository relationship accepted by [RpgAPI.Sync].
type RpgEdgeInput struct {
	SourceID     string `json:"source_id"`
	TargetID     string `json:"target_id"`
	Relationship string `json:"relationship"`
	// Strength is optional so nil uses the server default (1.0) while Ptr(0.0)
	// preserves an explicit zero-strength edge.
	Strength *float64       `json:"strength,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// RpgNode is a node returned by the RPG API.
type RpgNode struct {
	NodeID      string         `json:"node_id"`
	NodeType    string         `json:"node_type"`
	Path        string         `json:"path"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Language    *string        `json:"language,omitempty"`
	ParentID    *string        `json:"parent_id,omitempty"`
	Depth       int            `json:"depth"`
	Metadata    map[string]any `json:"metadata"`
	MemoryID    *string        `json:"memory_id,omitempty"`
	ContextID   *string        `json:"context_id,omitempty"`
}

// RpgEdge is an edge returned by the RPG API.
type RpgEdge struct {
	SourceID      string         `json:"source_id"`
	TargetID      string         `json:"target_id"`
	Relationship  string         `json:"relationship"`
	Strength      float64        `json:"strength"`
	Metadata      map[string]any `json:"metadata"`
	AssociationID *string        `json:"association_id,omitempty"`
}

// RpgSyncOptions configures [RpgAPI.Sync].
type RpgSyncOptions struct {
	FullSync     bool
	SourceCommit string
	ContextID    string
	TaskID       string
	Authority    *AuthorityContext
}

// RpgSyncResult reports changes from an RPG sync.
type RpgSyncResult struct {
	NodesCreated int     `json:"nodes_created"`
	NodesUpdated int     `json:"nodes_updated"`
	NodesDeleted int     `json:"nodes_deleted"`
	EdgesCreated int     `json:"edges_created"`
	EdgesUpdated int     `json:"edges_updated"`
	EdgesDeleted int     `json:"edges_deleted"`
	SyncTimeMs   int     `json:"sync_time_ms"`
	SourceCommit *string `json:"source_commit,omitempty"`
}

// RpgSubgraph is a repository graph rooted at a path or node.
type RpgSubgraph struct {
	Nodes      []RpgNode `json:"nodes"`
	Edges      []RpgEdge `json:"edges"`
	RootID     *string   `json:"root_id,omitempty"`
	Depth      int       `json:"depth"`
	TotalNodes int       `json:"total_nodes"`
	TotalEdges int       `json:"total_edges"`
	Truncated  bool      `json:"truncated,omitempty"`
}

// RpgSearchResult is a typed RPG search response.
type RpgSearchResult struct {
	Nodes      []RpgNode `json:"nodes"`
	TotalCount int       `json:"total_count"`
	Query      string    `json:"query"`
}

// RpgStatus reports repository graph coverage for a workspace.
type RpgStatus struct {
	WorkspaceID    string  `json:"workspace_id"`
	HasRPG         bool    `json:"has_rpg"`
	NodeCount      int     `json:"node_count"`
	EdgeCount      int     `json:"edge_count"`
	LastSyncCommit *string `json:"last_sync_commit,omitempty"`
	LastSyncAt     *string `json:"last_sync_at,omitempty"`
}

// RpgOverlay describes an overlay context.
type RpgOverlay struct {
	ContextID string `json:"context_id"`
	NodeCount int    `json:"node_count"`
}

// RpgOverlayList lists active overlays.
type RpgOverlayList struct {
	Overlays []RpgOverlay `json:"overlays"`
	Total    int          `json:"total"`
}

// RpgDeleteResult reports bulk deletion counts.
type RpgDeleteResult struct {
	Deleted  int `json:"deleted"`
	NotFound int `json:"not_found"`
}

// RpgOverlayDeleteResult reports an overlay deletion.
type RpgOverlayDeleteResult struct {
	ContextID string `json:"context_id"`
	Deleted   int    `json:"deleted"`
}

// RpgNodeList is a typed node-list response.
type RpgNodeList struct {
	Nodes      []RpgNode `json:"nodes"`
	TotalCount int       `json:"total_count"`
}

// RpgConflict describes one file-level conflict.
type RpgConflict struct {
	FilePath string   `json:"file_path"`
	TaskIDs  []string `json:"task_ids"`
	Severity string   `json:"severity"`
}

// RpgConflictResult contains conflicts for a task.
type RpgConflictResult struct {
	TaskID         string        `json:"task_id"`
	Conflicts      []RpgConflict `json:"conflicts"`
	TotalConflicts int           `json:"total_conflicts"`
}

// RpgSymbolConflict describes a symbol-level conflict.
type RpgSymbolConflict struct {
	SymbolID    string   `json:"symbol_id"`
	FilePath    string   `json:"file_path"`
	TaskIDs     []string `json:"task_ids"`
	Severity    string   `json:"severity"`
	Description string   `json:"description"`
}

// RpgSymbolConflictResult contains conflicts between two task overlays.
type RpgSymbolConflictResult struct {
	TaskIDs        []string            `json:"task_ids"`
	Conflicts      []RpgSymbolConflict `json:"conflicts"`
	TotalConflicts int                 `json:"total_conflicts"`
}

// RpgMaintenanceResult is returned by a graph maintenance operation.
type RpgMaintenanceResult struct {
	Operation   string         `json:"operation"`
	WorkspaceID string         `json:"workspace_id"`
	Result      map[string]any `json:"result"`
}

// RpgMaintenanceOperation names a supported graph maintenance operation.
type RpgMaintenanceOperation string

const (
	RpgMaintenanceValidate        RpgMaintenanceOperation = "validate"
	RpgMaintenanceCleanup         RpgMaintenanceOperation = "cleanup"
	RpgMaintenanceStatistics      RpgMaintenanceOperation = "statistics"
	RpgMaintenanceCleanupIntents  RpgMaintenanceOperation = "cleanup_intents"
	RpgMaintenanceRecomputeCounts RpgMaintenanceOperation = "recompute_counts"
)

// RpgEnrichmentResult reports a synchronous or scheduled enrichment.
type RpgEnrichmentResult struct {
	WorkspaceID string         `json:"workspace_id"`
	ContextID   string         `json:"context_id"`
	AsyncMode   bool           `json:"async_mode"`
	TaskID      *string        `json:"task_id,omitempty"`
	Stats       map[string]any `json:"stats,omitempty"`
}

// RpgEnrichOptions configures [RpgAPI.Enrich]. Async defaults to true when nil.
type RpgEnrichOptions struct {
	ContextID string
	Phases    []string
	Async     *bool
	Authority *AuthorityContext
}

// RpgAPI is the /v1/rpg namespace, reached through client.Rpg.
type RpgAPI struct{ client *Client }

// Sync upserts a repository graph snapshot.
func (r *RpgAPI) Sync(ctx context.Context, nodes []RpgNodeInput, edges []RpgEdgeInput, opts RpgSyncOptions) (*RpgSyncResult, error) {
	contextID := opts.ContextID
	if contextID == "" {
		contextID = "rpg"
	}
	payload := map[string]any{
		"nodes": nodes, "edges": edges, "full_sync": opts.FullSync,
		"context_id": contextID,
	}
	if opts.SourceCommit != "" {
		payload["source_commit"] = opts.SourceCommit
	}
	if opts.TaskID != "" {
		payload["task_id"] = opts.TaskID
	}
	var out RpgSyncResult
	if err := r.client.requestJSON(ctx, http.MethodPost, "/rpg/sync", payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// RpgSubgraphOptions configures [RpgAPI.GetSubgraph].
type RpgSubgraphOptions struct {
	Path              string
	NodeID            string
	Depth             int
	NodeTypes         []string
	RelationshipTypes []string
	ContextID         string
	MaxNodes          int
	Authority         *AuthorityContext
}

// GetSubgraph returns nodes and edges rooted at a path or node ID.
func (r *RpgAPI) GetSubgraph(ctx context.Context, opts RpgSubgraphOptions) (*RpgSubgraph, error) {
	if opts.Path == "" && opts.NodeID == "" {
		return nil, fmt.Errorf("path or node ID is required")
	}
	depth := opts.Depth
	if depth <= 0 {
		depth = 3
	}
	maxNodes := opts.MaxNodes
	if maxNodes <= 0 {
		maxNodes = 2000
	}
	contextID := opts.ContextID
	if contextID == "" {
		contextID = "rpg"
	}
	q := url.Values{"depth": {strconv.Itoa(depth)}, "context_id": {contextID}, "max_nodes": {strconv.Itoa(maxNodes)}}
	if opts.Path != "" {
		q.Set("path", opts.Path)
	}
	if opts.NodeID != "" {
		q.Set("node_id", opts.NodeID)
	}
	if len(opts.NodeTypes) > 0 {
		q.Set("node_types", strings.Join(opts.NodeTypes, ","))
	}
	if len(opts.RelationshipTypes) > 0 {
		q.Set("relationship_types", strings.Join(opts.RelationshipTypes, ","))
	}
	var out RpgSubgraph
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/subgraph", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// RpgSearchOptions configures [RpgAPI.Search].
type RpgSearchOptions struct {
	NodeTypes []string
	Limit     int
	ContextID string
	Authority *AuthorityContext
}

// Search finds RPG nodes by text.
func (r *RpgAPI) Search(ctx context.Context, query string, opts RpgSearchOptions) (*RpgSearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	contextID := opts.ContextID
	if contextID == "" {
		contextID = "rpg"
	}
	q := url.Values{"query": {query}, "limit": {strconv.Itoa(limit)}, "context_id": {contextID}}
	if len(opts.NodeTypes) > 0 {
		q.Set("node_types", strings.Join(opts.NodeTypes, ","))
	}
	var out RpgSearchResult
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/search", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetStatus returns RPG encoding status. Empty contextID aggregates contexts.
func (r *RpgAPI) GetStatus(ctx context.Context, contextID string, authority *AuthorityContext) (*RpgStatus, error) {
	q := url.Values{}
	if contextID != "" {
		q.Set("context_id", contextID)
	}
	var out RpgStatus
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/status", query: q, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListOverlays lists active RPG overlays.
func (r *RpgAPI) ListOverlays(ctx context.Context, authority *AuthorityContext) (*RpgOverlayList, error) {
	var out RpgOverlayList
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/overlays", authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteOverlay deletes all nodes in an overlay.
func (r *RpgAPI) DeleteOverlay(ctx context.Context, contextID string, authority *AuthorityContext) (*RpgOverlayDeleteResult, error) {
	var out RpgOverlayDeleteResult
	path := "/rpg/overlays/" + url.PathEscape(contextID)
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodDelete, path: path, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteNodes deletes nodes by RPG node ID.
func (r *RpgAPI) DeleteNodes(ctx context.Context, nodeIDs []string, contextID string, authority *AuthorityContext) (*RpgDeleteResult, error) {
	if contextID == "" {
		contextID = "rpg"
	}
	var out RpgDeleteResult
	err := r.client.requestJSON(ctx, http.MethodDelete, "/rpg/nodes", map[string]any{"node_ids": nodeIDs, "context_id": contextID}, requestSpec{authority: authority}, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteEdges deletes associations by ID.
func (r *RpgAPI) DeleteEdges(ctx context.Context, associationIDs []string, authority *AuthorityContext) (*RpgDeleteResult, error) {
	var out RpgDeleteResult
	err := r.client.requestJSON(ctx, http.MethodDelete, "/rpg/edges", map[string]any{"association_ids": associationIDs}, requestSpec{authority: authority}, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// RpgListNodesOptions configures [RpgAPI.ListNodes].
type RpgListNodesOptions struct {
	NodeType  string
	ContextID string
	Limit     int
	Authority *AuthorityContext
}

// ListNodes enumerates RPG nodes, optionally by type.
func (r *RpgAPI) ListNodes(ctx context.Context, opts RpgListNodesOptions) (*RpgNodeList, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 5000
	}
	contextID := opts.ContextID
	if contextID == "" {
		contextID = "rpg"
	}
	q := url.Values{"context_id": {contextID}, "limit": {strconv.Itoa(limit)}}
	if opts.NodeType != "" {
		q.Set("node_type", opts.NodeType)
	}
	var out RpgNodeList
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/nodes", query: q, authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetMergedSubgraph combines the canonical graph with one overlay.
func (r *RpgAPI) GetMergedSubgraph(ctx context.Context, overlayContext, baseContext, path string, depth int, authority *AuthorityContext) (*RpgSubgraph, error) {
	if baseContext == "" {
		baseContext = "rpg"
	}
	if depth <= 0 {
		depth = 3
	}
	q := url.Values{"base_context": {baseContext}, "overlay_context": {overlayContext}, "depth": {strconv.Itoa(depth)}}
	if path != "" {
		q.Set("path", path)
	}
	var out RpgSubgraph
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/merged-subgraph", query: q, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetConflicts detects file-level conflicts for a task.
func (r *RpgAPI) GetConflicts(ctx context.Context, taskID string, authority *AuthorityContext) (*RpgConflictResult, error) {
	var out RpgConflictResult
	q := url.Values{"task_id": {taskID}}
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/conflicts", query: q, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetSymbolConflicts detects symbol-level conflicts between two tasks.
func (r *RpgAPI) GetSymbolConflicts(ctx context.Context, taskIDA, taskIDB string, authority *AuthorityContext) (*RpgSymbolConflictResult, error) {
	var out RpgSymbolConflictResult
	q := url.Values{"task_id_a": {taskIDA}, "task_id_b": {taskIDB}}
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodGet, path: "/rpg/conflicts/symbols", query: q, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Maintenance runs validate, cleanup, statistics, cleanup_intents, or recompute_counts.
func (r *RpgAPI) Maintenance(ctx context.Context, operation RpgMaintenanceOperation, authority *AuthorityContext) (*RpgMaintenanceResult, error) {
	var out RpgMaintenanceResult
	path := fmt.Sprintf("/rpg/maintenance/%s", url.PathEscape(string(operation)))
	if err := r.client.doJSON(ctx, requestSpec{method: http.MethodPost, path: path, authority: authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Enrich runs or schedules RPG semantic enrichment.
func (r *RpgAPI) Enrich(ctx context.Context, opts RpgEnrichOptions) (*RpgEnrichmentResult, error) {
	contextID := opts.ContextID
	if contextID == "" {
		contextID = "rpg"
	}
	asyncMode := true
	if opts.Async != nil {
		asyncMode = *opts.Async
	}
	var out RpgEnrichmentResult
	payload := map[string]any{"context_id": contextID, "phases": opts.Phases, "async": asyncMode}
	if err := r.client.requestJSON(ctx, http.MethodPost, "/rpg/enrich", payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
