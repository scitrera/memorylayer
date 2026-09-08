package memorylayer

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

// SkillModel mirrors the server Skill record.
type SkillModel struct {
	ID            string         `json:"id"`
	WorkspaceID   string         `json:"workspace_id"`
	TenantID      string         `json:"tenant_id"`
	UserID        *string        `json:"user_id,omitempty"`
	Name          string         `json:"name"`
	Description   string         `json:"description"`
	Version       string         `json:"version"`
	License       *string        `json:"license,omitempty"`
	Compatibility *string        `json:"compatibility,omitempty"`
	AllowedTools  *string        `json:"allowed_tools,omitempty"`
	Body          string         `json:"body"`
	Metadata      map[string]any `json:"metadata"`
	SourceMode    string         `json:"source_mode"`
	ManifestHash  string         `json:"manifest_hash"`
	BundleHash    string         `json:"bundle_hash"`
	Enabled       bool           `json:"enabled"`
	Revision      int            `json:"revision"`
	ETag          string         `json:"etag"`
	CreatedAt     *string        `json:"created_at,omitempty"`
	UpdatedAt     *string        `json:"updated_at,omitempty"`
	DeletedAt     *string        `json:"deleted_at,omitempty"`
}

// SkillManifestCreateInput is the semantic manifest accepted by a
// conditional skill create. Bundle files remain separate child resources.
type SkillManifestCreateInput struct {
	Name          string         `json:"name"`
	Description   string         `json:"description"`
	Version       string         `json:"version,omitempty"`
	License       *string        `json:"license,omitempty"`
	Compatibility *string        `json:"compatibility,omitempty"`
	AllowedTools  *string        `json:"allowed_tools,omitempty"`
	Body          string         `json:"body,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
	SourceMode    string         `json:"source_mode,omitempty"`
}

// SkillManifestReplaceInput completely replaces the semantic manifest while
// preserving the stable skill ID, name, ownership, and bundle children.
type SkillManifestReplaceInput struct {
	Description   string         `json:"description"`
	Version       string         `json:"version,omitempty"`
	License       *string        `json:"license"`
	Compatibility *string        `json:"compatibility"`
	AllowedTools  *string        `json:"allowed_tools"`
	Body          string         `json:"body"`
	Metadata      map[string]any `json:"metadata"`
	SourceMode    string         `json:"source_mode,omitempty"`
	Enabled       bool           `json:"enabled"`
}

// SkillManifestMutationOptions carries required idempotency, CAS, workspace,
// and delegated-authority metadata.
type SkillManifestMutationOptions struct {
	WorkspaceID    string
	IdempotencyKey string
	ETag           string
	Authority      *AuthorityContext
}

// SkillGetOptions configures retrieval of an active manifest or tombstone.
type SkillGetOptions struct {
	WorkspaceID    string
	IncludeDeleted bool
	Authority      *AuthorityContext
}

// SkillHistoryOptions configures immutable manifest history traversal.
type SkillHistoryOptions struct {
	WorkspaceID string
	Limit       int
	PageToken   string
	Authority   *AuthorityContext
}

type SkillMutationResult struct {
	Skill    SkillModel
	Replayed bool
	ETag     string
}

type SkillRevision struct {
	Skill       SkillModel `json:"skill"`
	Action      string     `json:"action"`
	OperationID string     `json:"operation_id"`
}

type SkillRevisionPage struct {
	Revisions     []SkillRevision `json:"revisions"`
	NextPageToken string          `json:"next_page_token"`
}

// SkillFile is a single file in a skill bundle (path relative to the skill
// root).
type SkillFile struct {
	Path    string
	Content []byte
}

// SkillResolution is the result of [SkillsAPI.Resolve]. Exactly one field is
// populated: Skill for a name lookup (precedence winner), or Skills for a query
// (vector recall) — Skills may be empty when nothing matched.
type SkillResolution struct {
	Skill  *SkillModel
	Skills []SkillModel
}

// SkillsAPI is the /v1/skills namespace, reached via client.Skills.
type SkillsAPI struct {
	client *Client
}

func (s *SkillsAPI) ws(workspaceID string) string {
	return s.client.resolveWorkspace(workspaceID)
}

// ListSkillsOptions configures [SkillsAPI.List].
type ListSkillsOptions struct {
	Scope           string
	Name            string
	Tags            []string
	Enabled         *bool
	IncludeShadowed bool
	IncludeAddenda  bool
	WorkspaceID     string
	Authority       *AuthorityContext
}

// List lists visible skills.
func (s *SkillsAPI) List(ctx context.Context, opts ListSkillsOptions) ([]SkillModel, error) {
	q := url.Values{}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.Scope != "" {
		q.Set("scope", opts.Scope)
	}
	if opts.Name != "" {
		q.Set("name", opts.Name)
	}
	if len(opts.Tags) > 0 {
		q.Set("tags", strings.Join(opts.Tags, ","))
	}
	if opts.Enabled != nil {
		q.Set("enabled", boolStr(*opts.Enabled))
	}
	if opts.IncludeShadowed {
		q.Set("include_shadowed", "true")
	}
	if opts.IncludeAddenda {
		q.Set("include_addenda", "true")
	}
	var out struct {
		Skills []SkillModel `json:"skills"`
	}
	if err := s.client.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/skills",
		query:     q,
		authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return out.Skills, nil
}

// Get fetches a skill by ID.
func (s *SkillsAPI) Get(ctx context.Context, skillID string, authority *AuthorityContext) (*SkillModel, error) {
	return s.GetWithOptions(ctx, skillID, SkillGetOptions{Authority: authority})
}

// GetWithOptions fetches one active skill or, when requested, its tombstone.
func (s *SkillsAPI) GetWithOptions(ctx context.Context, skillID string, opts SkillGetOptions) (*SkillModel, error) {
	q := url.Values{}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.IncludeDeleted {
		q.Set("include_deleted", "true")
	}
	var out struct {
		Skill SkillModel `json:"skill"`
	}
	resp, err := s.client.doJSONResponse(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/skills/" + url.PathEscape(skillID),
		query:     q,
		authority: opts.Authority,
	}, &out)
	if err != nil {
		return nil, err
	}
	if out.Skill.ETag == "" {
		out.Skill.ETag = resp.Header.Get("ETag")
	}
	return &out.Skill, nil
}

// CreateVersioned conditionally creates one native skill manifest.
func (s *SkillsAPI) CreateVersioned(ctx context.Context, input SkillManifestCreateInput, opts SkillManifestMutationOptions) (*SkillMutationResult, error) {
	payload := map[string]any{}
	encoded, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(encoded, &payload); err != nil {
		return nil, err
	}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return s.mutateManifest(ctx, requestSpec{
		method: http.MethodPost, path: "/skills", body: body, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, "If-None-Match", "*"), authority: opts.Authority,
	})
}

// ReplaceManifest atomically replaces the complete semantic manifest.
func (s *SkillsAPI) ReplaceManifest(ctx context.Context, skillID string, input SkillManifestReplaceInput, opts SkillManifestMutationOptions) (*SkillMutationResult, error) {
	body, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return s.mutateManifest(ctx, requestSpec{
		method: http.MethodPut, path: "/skills/" + url.PathEscape(skillID) + "/manifest",
		query: skillWorkspaceQuery(s.ws(opts.WorkspaceID)), body: body, contentType: "application/json",
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// DeleteVersioned conditionally writes a durable manifest tombstone.
func (s *SkillsAPI) DeleteVersioned(ctx context.Context, skillID string, opts SkillManifestMutationOptions) (*SkillMutationResult, error) {
	return s.mutateManifest(ctx, requestSpec{
		method: http.MethodPost, path: "/skills/" + url.PathEscape(skillID) + "/delete",
		query:   skillWorkspaceQuery(s.ws(opts.WorkspaceID)),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// Restore conditionally reactivates the exact tombstoned manifest.
func (s *SkillsAPI) Restore(ctx context.Context, skillID string, opts SkillManifestMutationOptions) (*SkillMutationResult, error) {
	return s.mutateManifest(ctx, requestSpec{
		method: http.MethodPost, path: "/skills/" + url.PathEscape(skillID) + "/restore",
		query:   skillWorkspaceQuery(s.ws(opts.WorkspaceID)),
		headers: mutationHeaders(opts.IdempotencyKey, "If-Match", opts.ETag), authority: opts.Authority,
	})
}

// History returns immutable skill-manifest revisions newest first.
func (s *SkillsAPI) History(ctx context.Context, skillID string, opts SkillHistoryOptions) (*SkillRevisionPage, error) {
	q := promptNotePageQuery(s.ws(opts.WorkspaceID), opts.Limit, opts.PageToken)
	var out SkillRevisionPage
	if err := s.client.doJSON(ctx, requestSpec{
		method: http.MethodGet, path: "/skills/" + url.PathEscape(skillID) + "/revisions",
		query: q, authority: opts.Authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (s *SkillsAPI) mutateManifest(ctx context.Context, spec requestSpec) (*SkillMutationResult, error) {
	var out struct {
		Skill    SkillModel `json:"skill"`
		Replayed bool       `json:"replayed"`
	}
	resp, err := s.client.doJSONResponse(ctx, spec, &out)
	if err != nil {
		return nil, err
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		etag = out.Skill.ETag
	}
	return &SkillMutationResult{Skill: out.Skill, Replayed: out.Replayed, ETag: etag}, nil
}

func skillWorkspaceQuery(workspaceID string) url.Values {
	q := url.Values{}
	if workspaceID != "" {
		q.Set("workspace_id", workspaceID)
	}
	return q
}

// GetSkillManifestOptions configures [SkillsAPI.GetManifestWithOptions].
type GetSkillManifestOptions struct {
	WorkspaceID    string
	IncludeAddenda bool
	Authority      *AuthorityContext
}

// GetManifest returns the rendered SKILL.md text for a skill.
func (s *SkillsAPI) GetManifest(ctx context.Context, skillID string) (string, error) {
	return s.GetManifestWithOptions(ctx, skillID, GetSkillManifestOptions{})
}

// GetManifestWithOptions returns the rendered SKILL.md text for a skill.
func (s *SkillsAPI) GetManifestWithOptions(ctx context.Context, skillID string, opts GetSkillManifestOptions) (string, error) {
	q := url.Values{}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.IncludeAddenda {
		q.Set("include_addenda", "true")
	}
	resp, err := s.client.doRaw(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/skills/" + skillID + "/manifest",
		query:     q,
		authority: opts.Authority,
	})
	if err != nil {
		return "", err
	}
	if err := mapError(resp, ""); err != nil {
		return "", err
	}
	return string(resp.Body), nil
}

// GetFile returns a single file from the skill bundle.
func (s *SkillsAPI) GetFile(ctx context.Context, skillID, path string) ([]byte, error) {
	resp, err := s.client.doRaw(ctx, requestSpec{method: http.MethodGet, path: fmt.Sprintf("/skills/%s/files/%s", skillID, path)})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, ""); err != nil {
		return nil, err
	}
	return resp.Body, nil
}

// ListFiles lists files in a skill bundle as raw maps (path, size, ...).
func (s *SkillsAPI) ListFiles(ctx context.Context, skillID string, authority *AuthorityContext) ([]map[string]any, error) {
	var out struct {
		Files []map[string]any `json:"files"`
	}
	if err := s.client.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/skills/" + skillID + "/files",
		authority: authority,
	}, &out); err != nil {
		return nil, err
	}
	return out.Files, nil
}

// SaveSkillOptions configures [SkillsAPI.Save].
type SaveSkillOptions struct {
	Body          string
	Files         []SkillFile
	Scope         string // default "workspace"
	SourceMode    string // default "server"
	WorkspaceID   string
	UserID        string
	Version       string
	License       string
	Compatibility string
	AllowedTools  string
	Metadata      map[string]any
	Authority     *AuthorityContext
}

// Save creates or updates a skill by name.
func (s *SkillsAPI) Save(ctx context.Context, name, description string, opts SaveSkillOptions) (*SkillModel, error) {
	sourceMode := orDefault(opts.SourceMode, "server")
	payload := map[string]any{
		"name":        name,
		"description": description,
		"body":        opts.Body,
		"source_mode": sourceMode,
	}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	if opts.UserID != "" {
		payload["user_id"] = opts.UserID
	}
	if opts.Version != "" {
		payload["version"] = opts.Version
	}
	if opts.License != "" {
		payload["license"] = opts.License
	}
	if opts.Compatibility != "" {
		payload["compatibility"] = opts.Compatibility
	}
	if opts.AllowedTools != "" {
		payload["allowed_tools"] = opts.AllowedTools
	}
	if opts.Metadata != nil {
		payload["metadata"] = opts.Metadata
	}
	if len(opts.Files) > 0 {
		files := make([]map[string]any, len(opts.Files))
		for i, f := range opts.Files {
			files[i] = map[string]any{"path": f.Path, "content": string(f.Content)}
		}
		payload["files"] = files
	}
	var out struct {
		Skill SkillModel `json:"skill"`
	}
	if err := s.client.requestJSON(ctx, http.MethodPost, "/skills", payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out.Skill, nil
}

// Delete deletes a skill and its files.
func (s *SkillsAPI) Delete(ctx context.Context, skillID string, authority *AuthorityContext) error {
	return s.client.doJSON(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      "/skills/" + skillID,
		authority: authority,
	}, nil)
}

// ResolveSkillOptions configures [SkillsAPI.ResolveWithOptions].
type ResolveSkillOptions struct {
	Name           string
	Query          string
	ScopeHint      string
	WorkspaceID    string
	IncludeAddenda bool
	Authority      *AuthorityContext
}

// Resolve resolves a skill by name (precedence winner) or query (vector recall).
func (s *SkillsAPI) Resolve(ctx context.Context, name, query, workspaceID string, authority *AuthorityContext) (*SkillResolution, error) {
	return s.ResolveWithOptions(ctx, ResolveSkillOptions{
		Name:        name,
		Query:       query,
		WorkspaceID: workspaceID,
		Authority:   authority,
	})
}

// ResolveWithOptions resolves a skill by name or query.
func (s *SkillsAPI) ResolveWithOptions(ctx context.Context, opts ResolveSkillOptions) (*SkillResolution, error) {
	payload := map[string]any{}
	if opts.Name != "" {
		payload["name"] = opts.Name
	}
	if opts.Query != "" {
		payload["query"] = opts.Query
	}
	if opts.ScopeHint != "" {
		payload["scope_hint"] = opts.ScopeHint
	}
	if opts.IncludeAddenda {
		payload["include_addenda"] = true
	}
	if ws := s.ws(opts.WorkspaceID); ws != "" {
		payload["workspace_id"] = ws
	}
	var out struct {
		Skill      *SkillModel  `json:"skill"`
		Candidates []SkillModel `json:"candidates"`
		Skills     []SkillModel `json:"skills"`
	}
	if err := s.client.requestJSON(ctx, http.MethodPost, "/skills/resolve", payload, requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	candidates := out.Candidates
	if len(candidates) == 0 && len(out.Skills) > 0 {
		candidates = out.Skills
	}
	return &SkillResolution{Skill: out.Skill, Skills: candidates}, nil
}

// ------------------------------------------------------------------
// Local filesystem helpers
// ------------------------------------------------------------------

// Pull materializes a named skill into outDir/<name>/ and returns its path.
func (s *SkillsAPI) Pull(ctx context.Context, name, outDir, workspaceID string) (string, error) {
	res, err := s.Resolve(ctx, name, "", workspaceID, nil)
	if err != nil {
		return "", err
	}
	skill := res.Skill
	if skill == nil {
		if len(res.Skills) == 0 {
			return "", fmt.Errorf("memorylayer: skill %q not found", name)
		}
		skill = &res.Skills[0]
	}
	skillDir := filepath.Join(outDir, name)
	if err := s.writeSkill(ctx, skill, skillDir); err != nil {
		return "", err
	}
	return skillDir, nil
}

// Push parses a skill directory (SKILL.md + scripts/references/assets) and
// uploads it.
func (s *SkillsAPI) Push(ctx context.Context, skillDir, scope, sourceMode, workspaceID string) (*SkillModel, error) {
	manifest, files, err := ParseSkillFolder(skillDir)
	if err != nil {
		return nil, err
	}
	opts := SaveSkillOptions{
		Body:          manifest.Body,
		Files:         files,
		Scope:         orDefault(scope, "workspace"),
		SourceMode:    orDefault(sourceMode, "server"),
		WorkspaceID:   workspaceID,
		Version:       manifest.Version,
		License:       manifest.License,
		Compatibility: manifest.Compatibility,
		AllowedTools:  manifest.AllowedTools,
		Metadata:      manifest.Metadata,
	}
	return s.Save(ctx, manifest.Name, manifest.Description, opts)
}

// Materialize bulk-pulls all visible skills into targetDir. It is idempotent via
// each skill's bundle_hash (skipped when unchanged). Returns the per-skill dirs.
func (s *SkillsAPI) Materialize(ctx context.Context, targetDir, scope, workspaceID string) ([]string, error) {
	skills, err := s.List(ctx, ListSkillsOptions{Scope: scope, WorkspaceID: workspaceID})
	if err != nil {
		return nil, err
	}
	var dirs []string
	for i := range skills {
		skill := &skills[i]
		skillDir := filepath.Join(targetDir, skill.Name)
		hashFile := filepath.Join(skillDir, ".bundle_hash")
		if skill.BundleHash != "" {
			if existing, err := os.ReadFile(hashFile); err == nil && strings.TrimSpace(string(existing)) == skill.BundleHash {
				dirs = append(dirs, skillDir)
				continue
			}
		}
		if err := s.writeSkill(ctx, skill, skillDir); err != nil {
			return nil, err
		}
		if skill.BundleHash != "" {
			_ = os.WriteFile(hashFile, []byte(skill.BundleHash), 0o644)
		}
		dirs = append(dirs, skillDir)
	}
	return dirs, nil
}

// writeSkill writes a skill's SKILL.md and bundle files into skillDir.
func (s *SkillsAPI) writeSkill(ctx context.Context, skill *SkillModel, skillDir string) error {
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		return err
	}
	manifest, err := s.GetManifest(ctx, skill.ID)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(manifest), 0o644); err != nil {
		return err
	}
	files, err := s.ListFiles(ctx, skill.ID, nil)
	if err != nil {
		return err
	}
	for _, info := range files {
		rel, _ := info["path"].(string)
		if rel == "" {
			continue
		}
		content, err := s.GetFile(ctx, skill.ID, rel)
		if err != nil {
			return err
		}
		dest := filepath.Join(skillDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dest, content, 0o644); err != nil {
			return err
		}
	}
	return nil
}
