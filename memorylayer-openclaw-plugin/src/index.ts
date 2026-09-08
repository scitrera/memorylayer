/**
 * @scitrera/memorylayer-openclaw-plugin — OpenClaw plugin entry.
 *
 * Implements OpenClaw's first-class memory capability, backed by MemoryLayer.ai
 * via the per-sandbox Aether sidecar relay. The plugin holds no real
 * credentials — the sidecar terminates the relay locally and injects the
 * proper Aether identity upstream.
 *
 * Sibling references:
 *   - @openclaw/memory-core              — reference impl (file/lancedb backed)
 *   - @scitrera/memorylayer-cc-plugin    — sibling plugin for Claude Code
 *   - @scitrera/memorylayer-opencode-plugin — sibling plugin for OpenCode
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { createMemoryLayerClient, type ConfiguredMemoryLayer } from "./client.js";
import { resolveConfig } from "./config.js";
import { buildMemoryCapability } from "./memory-capability.js";
import { registerMessagePersistence } from "./persistence.js";
import { registerSkillsSync } from "./skills-sync.js";

// The return type is annotated explicitly rather than inferred: the inferred
// type names internal openclaw SDK types via a hashed private module path, which
// `tsc` cannot emit into a portable declaration file (TS2742). Naming it through
// `ReturnType<typeof definePluginEntry>` is the same indirection the openclaw SDK
// uses for its own entry types.
const entry: ReturnType<typeof definePluginEntry> = definePluginEntry({
  // Must match openclaw.plugin.json `id` / package name / host config — the
  // loader rejects a mismatch between the manifest/config id and the entry's
  // exported id (it would otherwise fail validation and not load).
  id: "@scitrera/memorylayer-openclaw-plugin",
  name: "MemoryLayer",
  description:
    "First-class memory backed by MemoryLayer.ai via per-sandbox Aether sidecar transport.",
  // `register` is synchronous in OpenClaw's plugin SDK; we kick off Aether
  // connection lazily on first capability/hook call so we don't block plugin load.
  register: (api) => {
    const config = resolveConfig();

    let connectionPromise: Promise<ConfiguredMemoryLayer> | null = null;
    const getConnection = (): Promise<ConfiguredMemoryLayer> => {
      if (connectionPromise === null) {
        connectionPromise = createMemoryLayerClient(config);
      }
      return connectionPromise;
    };

    api.registerMemoryCapability(buildMemoryCapability({ getConnection, config }));
    registerMessagePersistence(api, { getConnection, config });
    // Opt-in (MEMORYLAYER_SKILLS_SYNC=1): materialize MemoryLayer skills into
    // OpenClaw's watched workspace skills dir on agent bootstrap. No-op when off.
    registerSkillsSync(api, { getConnection, config });
  },
});

export default entry;
