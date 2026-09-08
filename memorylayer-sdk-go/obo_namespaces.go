package memorylayer

import "context"

// This file adds the bound-namespace proxies reachable from an [OBOProxy]:
// p.Skills(), p.McpServers(), and p.Kb() return namespace views that inject the
// proxy's authority + default workspace into every call, mirroring the Python
// SDK's _SkillsOBOProxy / _McpServersOBOProxy / _KbOBOProxy. Each method still
// accepts an explicit authority/workspace to override the bound value.

// Skills returns the Skills namespace bound to this proxy's authority + workspace.
func (p *OBOProxy) Skills() *OBOSkillsAPI {
	return &OBOSkillsAPI{parent: p.parent.Skills, authority: p.authority, workspaceID: p.workspaceID}
}

// McpServers returns the McpServers namespace bound to this proxy's authority +
// workspace.
func (p *OBOProxy) McpServers() *OBOMcpServersAPI {
	return &OBOMcpServersAPI{parent: p.parent.McpServers, authority: p.authority, workspaceID: p.workspaceID}
}

// Kb returns the Knowledgebase namespace bound to this proxy's authority +
// workspace.
func (p *OBOProxy) Kb() *OBOKbAPI {
	return &OBOKbAPI{parent: p.parent.Kb, authority: p.authority, workspaceID: p.workspaceID}
}

// ----------------------------------------------------------------------------
// Skills
// ----------------------------------------------------------------------------

// OBOSkillsAPI is a [SkillsAPI] view bound to an OBO authority + workspace.
type OBOSkillsAPI struct {
	parent      *SkillsAPI
	authority   *AuthorityContext
	workspaceID string
}

func (s *OBOSkillsAPI) ws(w string) string {
	if w != "" {
		return w
	}
	return s.workspaceID
}

// List lists visible skills under the bound authority + workspace.
func (s *OBOSkillsAPI) List(ctx context.Context, opts ListSkillsOptions) ([]SkillModel, error) {
	opts.WorkspaceID = s.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, s.authority)
	return s.parent.List(ctx, opts)
}

// Get fetches a skill by ID under the bound authority.
func (s *OBOSkillsAPI) Get(ctx context.Context, skillID string, authority *AuthorityContext) (*SkillModel, error) {
	return s.parent.Get(ctx, skillID, resolveAuthority(authority, s.authority))
}

// ListFiles lists a skill's bundle files under the bound authority.
func (s *OBOSkillsAPI) ListFiles(ctx context.Context, skillID string, authority *AuthorityContext) ([]map[string]any, error) {
	return s.parent.ListFiles(ctx, skillID, resolveAuthority(authority, s.authority))
}

// GetManifest returns the rendered SKILL.md text under the bound authority.
func (s *OBOSkillsAPI) GetManifest(ctx context.Context, skillID string) (string, error) {
	return s.GetManifestWithOptions(ctx, skillID, GetSkillManifestOptions{})
}

// GetManifestWithOptions returns the rendered SKILL.md text under the bound authority + workspace.
func (s *OBOSkillsAPI) GetManifestWithOptions(ctx context.Context, skillID string, opts GetSkillManifestOptions) (string, error) {
	opts.WorkspaceID = s.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, s.authority)
	return s.parent.GetManifestWithOptions(ctx, skillID, opts)
}

// Save creates or updates a skill under the bound authority + workspace.
func (s *OBOSkillsAPI) Save(ctx context.Context, name, description string, opts SaveSkillOptions) (*SkillModel, error) {
	opts.WorkspaceID = s.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, s.authority)
	return s.parent.Save(ctx, name, description, opts)
}

// Delete deletes a skill under the bound authority.
func (s *OBOSkillsAPI) Delete(ctx context.Context, skillID string, authority *AuthorityContext) error {
	return s.parent.Delete(ctx, skillID, resolveAuthority(authority, s.authority))
}

// Resolve resolves a skill by name or query under the bound authority +
// workspace.
func (s *OBOSkillsAPI) Resolve(ctx context.Context, name, query, workspaceID string, authority *AuthorityContext) (*SkillResolution, error) {
	return s.parent.Resolve(ctx, name, query, s.ws(workspaceID), resolveAuthority(authority, s.authority))
}

// ResolveWithOptions resolves a skill by name or query under the bound authority + workspace.
func (s *OBOSkillsAPI) ResolveWithOptions(ctx context.Context, opts ResolveSkillOptions) (*SkillResolution, error) {
	opts.WorkspaceID = s.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, s.authority)
	return s.parent.ResolveWithOptions(ctx, opts)
}

// ----------------------------------------------------------------------------
// McpServers
// ----------------------------------------------------------------------------

// OBOMcpServersAPI is a [McpServersAPI] view bound to an OBO authority +
// workspace.
type OBOMcpServersAPI struct {
	parent      *McpServersAPI
	authority   *AuthorityContext
	workspaceID string
}

func (m *OBOMcpServersAPI) ws(w string) string {
	if w != "" {
		return w
	}
	return m.workspaceID
}

// List lists MCP servers under the bound authority + workspace.
func (m *OBOMcpServersAPI) List(ctx context.Context, opts ListMcpServersOptions) ([]McpServerModel, error) {
	opts.WorkspaceID = m.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, m.authority)
	return m.parent.List(ctx, opts)
}

// Get fetches an MCP server by ID under the bound authority.
func (m *OBOMcpServersAPI) Get(ctx context.Context, serverID string, authority *AuthorityContext) (*McpServerModel, error) {
	return m.parent.Get(ctx, serverID, resolveAuthority(authority, m.authority))
}

// Create creates an MCP server record under the bound authority + workspace.
func (m *OBOMcpServersAPI) Create(ctx context.Context, name, transport string, opts CreateMcpServerOptions) (*McpServerModel, error) {
	opts.WorkspaceID = m.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, m.authority)
	return m.parent.Create(ctx, name, transport, opts)
}

// Update partially updates an MCP server record under the bound authority.
func (m *OBOMcpServersAPI) Update(ctx context.Context, serverID string, opts UpdateMcpServerOptions) (*McpServerModel, error) {
	opts.Authority = resolveAuthority(opts.Authority, m.authority)
	return m.parent.Update(ctx, serverID, opts)
}

// Delete deletes an MCP server record under the bound authority.
func (m *OBOMcpServersAPI) Delete(ctx context.Context, serverID string, authority *AuthorityContext) error {
	return m.parent.Delete(ctx, serverID, resolveAuthority(authority, m.authority))
}

// Resolve resolves an MCP server by name under the bound authority + workspace.
func (m *OBOMcpServersAPI) Resolve(ctx context.Context, name, workspaceID string, authority *AuthorityContext) (*McpServerModel, error) {
	return m.parent.Resolve(ctx, name, m.ws(workspaceID), resolveAuthority(authority, m.authority))
}

// ----------------------------------------------------------------------------
// Knowledgebase
// ----------------------------------------------------------------------------

// OBOKbAPI is a [KnowledgebaseAPI] view bound to an OBO authority + workspace.
type OBOKbAPI struct {
	parent      *KnowledgebaseAPI
	authority   *AuthorityContext
	workspaceID string
}

func (k *OBOKbAPI) ws(w string) string {
	if w != "" {
		return w
	}
	return k.workspaceID
}

// Get returns knowledgebase metadata under the bound authority + workspace.
func (k *OBOKbAPI) Get(ctx context.Context, opts KbGetOptions) (*KbMetadata, error) {
	opts.WorkspaceID = k.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, k.authority)
	return k.parent.Get(ctx, opts)
}

// ListArticles lists articles under the bound authority + workspace.
func (k *OBOKbAPI) ListArticles(ctx context.Context, opts KbListArticlesOptions) ([]KbArticle, error) {
	opts.WorkspaceID = k.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, k.authority)
	return k.parent.ListArticles(ctx, opts)
}

// GetArticle fetches a single article under the bound authority + workspace.
func (k *OBOKbAPI) GetArticle(ctx context.Context, articleID, workspaceID string, authority *AuthorityContext) (*KbArticle, error) {
	return k.parent.GetArticle(ctx, articleID, k.ws(workspaceID), resolveAuthority(authority, k.authority))
}

// Generate triggers knowledgebase (re)generation under the bound authority +
// workspace.
func (k *OBOKbAPI) Generate(ctx context.Context, opts KbGenerateOptions) (*KbMetadata, error) {
	opts.WorkspaceID = k.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, k.authority)
	return k.parent.Generate(ctx, opts)
}

// ExportVault downloads the vault zip under the bound authority + workspace.
func (k *OBOKbAPI) ExportVault(ctx context.Context, workspaceID string, authority *AuthorityContext) ([]byte, error) {
	return k.parent.ExportVault(ctx, k.ws(workspaceID), resolveAuthority(authority, k.authority))
}

// GetGraphAnalysis runs a graph analysis under the bound authority + workspace.
func (k *OBOKbAPI) GetGraphAnalysis(ctx context.Context, workspaceID, contextID string, authority *AuthorityContext) (*KbGraphAnalysis, error) {
	return k.parent.GetGraphAnalysis(ctx, k.ws(workspaceID), contextID, resolveAuthority(authority, k.authority))
}

// GetCommunity fetches a community under the bound authority + workspace.
func (k *OBOKbAPI) GetCommunity(ctx context.Context, communityID int, workspaceID string, authority *AuthorityContext) (*KbCommunity, error) {
	return k.parent.GetCommunity(ctx, communityID, k.ws(workspaceID), resolveAuthority(authority, k.authority))
}
