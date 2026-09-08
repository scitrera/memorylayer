package memorylayer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// SkillManifest is the parsed SKILL.md frontmatter plus body.
type SkillManifest struct {
	Name          string
	Description   string
	Version       string
	License       string
	Compatibility string
	AllowedTools  string
	Body          string
	Metadata      map[string]any
}

// _kindDirs maps the Agentskills bundle directories that ParseSkillFolder walks.
var _kindDirs = []string{"scripts", "references", "assets"}

var frontmatterRe = regexp.MustCompile(`(?s)^---\s*\n(.*?)\n---\s*\n?(.*)`)

// ParseSkillFolder parses a skill directory into its manifest and bundle files.
// It reads SKILL.md for frontmatter (name, description, version, ...) and body,
// and walks scripts/, references/, and assets/ plus any root-level files.
func ParseSkillFolder(path string) (*SkillManifest, []SkillFile, error) {
	skillMD := filepath.Join(path, "SKILL.md")
	text, err := os.ReadFile(skillMD)
	if err != nil {
		return nil, nil, fmt.Errorf("memorylayer: SKILL.md not found in %s: %w", path, err)
	}
	fm, body := parseFrontmatter(string(text))

	manifest := &SkillManifest{
		Name:        stringOr(fm, "name", filepath.Base(path)),
		Description: stringOr(fm, "description", ""),
		Version:     stringOr(fm, "version", "0.1.0"),
		Body:        body,
	}
	manifest.License = stringOr(fm, "license", "")
	manifest.Compatibility = stringOr(fm, "compatibility", "")
	manifest.AllowedTools = stringOr(fm, "allowed_tools", "")

	known := map[string]bool{
		"name": true, "description": true, "version": true,
		"license": true, "compatibility": true, "allowed_tools": true,
	}
	extras := map[string]any{}
	for k, v := range fm {
		if !known[k] {
			extras[k] = v
		}
	}
	if len(extras) > 0 {
		manifest.Metadata = extras
	}

	var files []SkillFile
	for _, kind := range _kindDirs {
		dir := filepath.Join(path, kind)
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			continue
		}
		walked, err := collectFiles(path, dir)
		if err != nil {
			return nil, nil, err
		}
		files = append(files, walked...)
	}

	// Root-level files (excluding SKILL.md and the known subdirs).
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, nil, err
	}
	var rootNames []string
	for _, e := range entries {
		if !e.IsDir() && e.Name() != "SKILL.md" {
			rootNames = append(rootNames, e.Name())
		}
	}
	sort.Strings(rootNames)
	for _, name := range rootNames {
		content, err := os.ReadFile(filepath.Join(path, name))
		if err != nil {
			return nil, nil, err
		}
		files = append(files, SkillFile{Path: name, Content: content})
	}

	return manifest, files, nil
}

// collectFiles walks dir and returns its files with paths relative to root,
// using forward slashes (the skill bundle path convention).
func collectFiles(root, dir string) ([]SkillFile, error) {
	var out []SkillFile
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		content, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		out = append(out, SkillFile{Path: filepath.ToSlash(rel), Content: content})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// parseFrontmatter splits SKILL.md into a frontmatter map and body. It accepts
// JSON frontmatter, otherwise a simple key: value parser (no YAML dependency).
func parseFrontmatter(text string) (map[string]any, string) {
	m := frontmatterRe.FindStringSubmatch(text)
	if m == nil {
		return map[string]any{}, text
	}
	raw, body := m[1], m[2]

	var jsonFM map[string]any
	if err := json.Unmarshal([]byte(raw), &jsonFM); err == nil {
		return jsonFM, body
	}
	return parseSimpleYAML(raw), body
}

// parseSimpleYAML parses simple "key: value" lines without an external YAML
// dependency.
func parseSimpleYAML(text string) map[string]any {
	result := map[string]any{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if len(val) >= 2 {
			if (val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'') {
				val = val[1 : len(val)-1]
			}
		}
		result[key] = val
	}
	return result
}

func stringOr(m map[string]any, key, def string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return def
}
