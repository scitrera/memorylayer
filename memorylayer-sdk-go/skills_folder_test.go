package memorylayer

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseSkillFolder(t *testing.T) {
	dir := t.TempDir()
	skillMD := `---
name: my-skill
description: Does a thing
version: 1.2.3
license: MIT
custom_key: custom_value
---
# Body

The skill body.`
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(skillMD), 0o644); err != nil {
		t.Fatal(err)
	}
	scripts := filepath.Join(dir, "scripts")
	if err := os.MkdirAll(scripts, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scripts, "run.sh"), []byte("echo hi"), 0o644); err != nil {
		t.Fatal(err)
	}

	manifest, files, err := ParseSkillFolder(dir)
	if err != nil {
		t.Fatalf("ParseSkillFolder: %v", err)
	}
	if manifest.Name != "my-skill" || manifest.Description != "Does a thing" || manifest.Version != "1.2.3" {
		t.Errorf("manifest = %#v", manifest)
	}
	if manifest.License != "MIT" {
		t.Errorf("license = %q", manifest.License)
	}
	if manifest.Body == "" || manifest.Metadata["custom_key"] != "custom_value" {
		t.Errorf("body/metadata = %q / %#v", manifest.Body, manifest.Metadata)
	}
	var foundScript bool
	for _, f := range files {
		if f.Path == "scripts/run.sh" && string(f.Content) == "echo hi" {
			foundScript = true
		}
	}
	if !foundScript {
		t.Errorf("expected scripts/run.sh in files, got %#v", files)
	}
}

func TestParseSkillFolderMissingMD(t *testing.T) {
	if _, _, err := ParseSkillFolder(t.TempDir()); err == nil {
		t.Error("expected error for missing SKILL.md")
	}
}
