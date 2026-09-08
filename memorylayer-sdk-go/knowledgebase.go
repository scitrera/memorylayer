package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const kbFeature = "Knowledgebase"

// KbArticle is a single knowledgebase article (index, community, or entity).
type KbArticle struct {
	ID          string         `json:"id"`
	ArticleType string         `json:"article_type"`
	Title       string         `json:"title"`
	ContentMD   string         `json:"content_md"`
	Metadata    map[string]any `json:"metadata"`
	GeneratedAt time.Time      `json:"generated_at"`
}

// KbGraphStats holds aggregate statistics about a workspace association graph.
type KbGraphStats struct {
	NodeCount      int     `json:"node_count"`
	EdgeCount      int     `json:"edge_count"`
	CommunityCount int     `json:"community_count"`
	Density        float64 `json:"density"`
	AvgDegree      float64 `json:"avg_degree"`
	MaxDegree      int     `json:"max_degree"`
	GodNodeCount   int     `json:"god_node_count"`
}

// KbMetadata describes a generated knowledgebase for a workspace.
type KbMetadata struct {
	WorkspaceID    string        `json:"workspace_id"`
	ArticleCount   int           `json:"article_count"`
	CommunityCount int           `json:"community_count"`
	GeneratedAt    time.Time     `json:"generated_at"`
	Stats          *KbGraphStats `json:"stats,omitempty"`
}

// KbCommunity is a detected community (cluster) within the association graph.
type KbCommunity struct {
	ID             int      `json:"id"`
	MemoryIDs      []string `json:"memory_ids"`
	Size           int      `json:"size"`
	CohesionScore  float64  `json:"cohesion_score"`
	CentralNodeIDs []string `json:"central_node_ids"`
	Label          *string  `json:"label,omitempty"`
}

// KbCentralNode is a high-centrality node ("god node") in the graph.
type KbCentralNode struct {
	MemoryID    string  `json:"memory_id"`
	Degree      int     `json:"degree"`
	Betweenness float64 `json:"betweenness"`
	CommunityID int     `json:"community_id"`
}

// KbBridge is an edge connecting two different communities.
type KbBridge struct {
	SourceCommunityID int     `json:"source_community_id"`
	TargetCommunityID int     `json:"target_community_id"`
	MemoryIDSource    string  `json:"memory_id_source"`
	MemoryIDTarget    string  `json:"memory_id_target"`
	RelationshipType  string  `json:"relationship_type"`
	Strength          float64 `json:"strength"`
}

// KbGraphSnapshot is snapshot metadata for the association graph.
type KbGraphSnapshot struct {
	WorkspaceID string  `json:"workspace_id"`
	ContextID   *string `json:"context_id,omitempty"`
	NodeCount   int     `json:"node_count"`
	EdgeCount   int     `json:"edge_count"`
	IncludesRPG bool    `json:"includes_rpg"`
}

// KbGraphAnalysis is a complete graph analysis result for a workspace.
type KbGraphAnalysis struct {
	Snapshot     KbGraphSnapshot `json:"snapshot"`
	Communities  []KbCommunity   `json:"communities"`
	CentralNodes []KbCentralNode `json:"central_nodes"`
	Bridges      []KbBridge      `json:"bridges"`
	Stats        KbGraphStats    `json:"stats"`
}

// KnowledgebaseAPI is the /v1/knowledgebase namespace, reached via client.Kb.
type KnowledgebaseAPI struct {
	client *Client
}

func (k *KnowledgebaseAPI) ws(workspaceID string) string {
	return k.client.resolveWorkspace(workspaceID)
}

// KbGetOptions configures [KnowledgebaseAPI.Get].
type KbGetOptions struct {
	WorkspaceID string
	ContextID   string
	Authority   *AuthorityContext
}

// Get returns the latest knowledgebase metadata for a workspace.
func (k *KnowledgebaseAPI) Get(ctx context.Context, opts KbGetOptions) (*KbMetadata, error) {
	q := url.Values{}
	if ws := k.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.ContextID != "" {
		q.Set("context_id", opts.ContextID)
	}
	var out KbMetadata
	if err := k.client.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/knowledgebase",
		query:             q,
		authority:         opts.Authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// KbListArticlesOptions configures [KnowledgebaseAPI.ListArticles].
type KbListArticlesOptions struct {
	WorkspaceID string
	ArticleType string
	Limit       int // default 100 when <= 0
	Offset      int
	Authority   *AuthorityContext
}

// ListArticles lists knowledgebase articles, optionally filtered by type.
func (k *KnowledgebaseAPI) ListArticles(ctx context.Context, opts KbListArticlesOptions) ([]KbArticle, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}}
	if ws := k.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.ArticleType != "" {
		q.Set("article_type", opts.ArticleType)
	}
	var out struct {
		Articles []KbArticle `json:"articles"`
	}
	if err := k.client.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/knowledgebase/articles",
		query:             q,
		authority:         opts.Authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Articles, nil
}

// GetArticle fetches a single article by ID (e.g. "index", "community-3").
func (k *KnowledgebaseAPI) GetArticle(ctx context.Context, articleID, workspaceID string, authority *AuthorityContext) (*KbArticle, error) {
	q := url.Values{}
	if ws := k.ws(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out KbArticle
	if err := k.client.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/knowledgebase/articles/" + articleID,
		query:             q,
		authority:         authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// KbGenerateOptions configures [KnowledgebaseAPI.Generate].
type KbGenerateOptions struct {
	WorkspaceID    string
	Regenerate     bool
	MaxCommunities *int
	MaxGodNodes    *int
	IncludeRPG     bool
	ContextID      string
	Authority      *AuthorityContext
}

// Generate triggers (re)generation of the workspace knowledgebase.
func (k *KnowledgebaseAPI) Generate(ctx context.Context, opts KbGenerateOptions) (*KbMetadata, error) {
	payload := map[string]any{"regenerate": opts.Regenerate, "include_rpg": opts.IncludeRPG}
	if ws := k.ws(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	if opts.ContextID != "" {
		payload["context_id"] = opts.ContextID
	}
	if opts.MaxCommunities != nil {
		payload["max_communities"] = *opts.MaxCommunities
	}
	if opts.MaxGodNodes != nil {
		payload["max_god_nodes"] = *opts.MaxGodNodes
	}
	var out KbMetadata
	if err := k.client.requestJSON(ctx, http.MethodPost, "/knowledgebase/generate", payload, requestSpec{
		authority:         opts.Authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ExportVault downloads the knowledgebase as an Obsidian-compatible vault zip.
func (k *KnowledgebaseAPI) ExportVault(ctx context.Context, workspaceID string, authority *AuthorityContext) ([]byte, error) {
	q := url.Values{}
	if ws := k.ws(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	resp, err := k.client.doRaw(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/knowledgebase/export",
		query:     q,
		authority: authority,
	})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, kbFeature); err != nil {
		return nil, err
	}
	return resp.Body, nil
}

// GetGraphAnalysis runs a fresh graph analysis on the workspace association
// graph. Returns (nil, nil) when the server reports no analysis available.
func (k *KnowledgebaseAPI) GetGraphAnalysis(ctx context.Context, workspaceID, contextID string, authority *AuthorityContext) (*KbGraphAnalysis, error) {
	q := url.Values{}
	if ws := k.ws(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if contextID != "" {
		q.Set("context_id", contextID)
	}
	var out struct {
		Analysis *KbGraphAnalysis `json:"analysis"`
	}
	if err := k.client.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/knowledgebase/graph",
		query:             q,
		authority:         authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Analysis, nil
}

// GetCommunity fetches a single community by ID with its cached label and live
// members.
func (k *KnowledgebaseAPI) GetCommunity(ctx context.Context, communityID int, workspaceID string, authority *AuthorityContext) (*KbCommunity, error) {
	q := url.Values{}
	if ws := k.ws(workspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	var out KbCommunity
	if err := k.client.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              fmt.Sprintf("/knowledgebase/graph/communities/%d", communityID),
		query:             q,
		authority:         authority,
		enterpriseFeature: kbFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
