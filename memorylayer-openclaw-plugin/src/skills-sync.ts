/**
 * Skills sync — RECONCILE MemoryLayer skills into a dedicated, exclusively
 * MemoryLayer-managed skills directory that OpenClaw discovers.
 *
 * OpenClaw has native, first-class skills: `SKILL.md` directories discovered
 * from several roots, including the configured `skills.load.extraDirs` (each
 * dir is scanned one level deep for skill subfolders). We make MemoryLayer the
 * source of truth and OpenClaw a local cache, but we must NOT write into the
 * shared native `<workspaceDir>/skills` dir: that root also holds
 * user-authored / plugin / bundled skills, and a reconcile there would prune
 * (clobber) skills we don't own. Instead we materialize into a SEPARATE,
 * exclusively-MemoryLayer dir — by convention `<workspaceDir>/.memorylayer-skills`
 * (a sibling, NOT under `skills/`, since OpenClaw scans one level) — and the
 * sandbox-provider registers that same dir in `skills.load.extraDirs` so
 * OpenClaw discovers it. Because the dir is exclusively ours, we can safely
 * reconcile it.
 *
 * On agent bootstrap we fetch the enabled skills and call the MemoryLayer SDK's
 * `client.skills.materialize(managedDir, { workspaceId, enabled: true,
 * reconcile: true })`. `reconcile: true` makes the sync a true reconcile: after
 * successfully listing AND writing the enabled set, the SDK removes any managed
 * skill dir (one containing `SKILL.md`) in `managedDir` that is NOT in the
 * desired set. This is how a skill disabled/deleted/renamed in MemoryLayer gets
 * its stale on-disk copy pruned. Pruning runs ONLY after a successful list, so a
 * transient server outage can never wipe the dir.
 *
 * Single source of truth for the dir: the plugin honours
 * `MEMORYLAYER_SKILLS_DIR` (absolute path) when set, otherwise defaults to
 * `<workspaceDir>/.memorylayer-skills`. The sandbox-provider stamps the same
 * value as both the plugin env var and the rendered `skills.load.extraDirs`
 * entry, so the plugin's write path and OpenClaw's discovery path always agree.
 *
 * Trigger: the `agent:bootstrap` internal hook (OpenClaw's per-agent
 * session-start lifecycle event). Its context carries `workspaceDir`, used to
 * build the default managed-dir path when the env override is absent. See
 * `AgentBootstrapHookContext` in
 * `openclaw/plugin-sdk/.../hooks/internal-hooks` (`{ workspaceDir, ... }`).
 *
 * This sync is opt-in (`MEMORYLAYER_SKILLS_SYNC=1`, default OFF) and best-effort:
 * any error is logged and swallowed so it never blocks the agent bootstrap or
 * the rest of the plugin. This is graceful degradation of an optional sync, not
 * a hidden fallback for the memory capability.
 *
 * Plan: Phase 4 — MemoryLayer-backed skills (4d/unify).
 */

import path from "node:path";

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";

import type { ConfiguredMemoryLayer } from "./client.js";
import type { PluginConfig } from "./config.js";

/** Internal hook key for OpenClaw's per-agent bootstrap lifecycle event. */
const AGENT_BOOTSTRAP_EVENT = "agent:bootstrap";

/** Unique hook registration name (required by OpenClaw's plugin hook loader). */
const SKILLS_SYNC_HOOK_NAME = "memorylayer-skills-sync";

/**
 * Dedicated, exclusively-MemoryLayer skills subdir of the agent workspace.
 *
 * NOT the shared native `skills/` dir — this is a sibling we own outright, so
 * the SDK's `reconcile` can safely prune stale entries here. The sandbox-provider
 * registers this same path in OpenClaw's `skills.load.extraDirs` for discovery.
 */
const WORKSPACE_MANAGED_SKILLS_SUBDIR = ".memorylayer-skills";

/**
 * Env var the sandbox-provider may stamp with the absolute managed-skills dir,
 * keeping the plugin's write path identical to the rendered `extraDirs` entry.
 * When unset, the plugin falls back to `<workspaceDir>/.memorylayer-skills`.
 */
const ENV_SKILLS_DIR = "MEMORYLAYER_SKILLS_DIR";

/**
 * Resolve the managed-skills dir: prefer the `MEMORYLAYER_SKILLS_DIR` env
 * override (single source of truth shared with the provider's rendered
 * `extraDirs`), else the `<workspaceDir>/.memorylayer-skills` convention.
 */
function resolveManagedDir(
  workspaceDir: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const override = env[ENV_SKILLS_DIR]?.trim();
  if (override) return override;
  return path.join(workspaceDir, WORKSPACE_MANAGED_SKILLS_SUBDIR);
}

export interface SkillsSyncParams {
  getConnection: () => Promise<ConfiguredMemoryLayer>;
  config: PluginConfig;
  /** Optional log sink — defaults to console.warn. */
  warn?: (msg: string, err?: unknown) => void;
  /** Optional info sink — defaults to console.info. */
  info?: (msg: string) => void;
}

/**
 * Reconcile the principal's enabled MemoryLayer skills into `targetDir`.
 *
 * `targetDir` MUST be the dedicated, exclusively-MemoryLayer managed dir (never
 * the shared native `skills/`): we pass `reconcile: true`, so the SDK removes
 * any managed skill dir not in the enabled set. Idempotent / safe to call
 * repeatedly: each run re-writes the enabled `SKILL.md` + files and prunes
 * stale ones, converging the dir to exactly the server's enabled set. Errors
 * propagate to the caller (the hook wrapper swallows them) so this stays
 * unit-testable in isolation.
 */
export async function materializeSkills(
  targetDir: string,
  params: SkillsSyncParams,
): Promise<void> {
  const { config } = params;
  const info = params.info ?? ((msg) => console.info(`[memorylayer-plugin] ${msg}`));
  const { client } = await params.getConnection();

  // Scope the skills namespace to the principal's OBO authority when configured;
  // `client.skills` itself carries no default authority (the client only applies
  // its defaultAuthority on memory calls), so we must attach it explicitly here.
  const skills = config.grantId
    ? client.skills.withAuthority(
        {
          grantId: config.grantId,
          subject: { type: config.principalType, id: config.principalId },
        },
        config.defaultWorkspaceId,
      )
    : client.skills;

  // Discover the enabled set first — gives us a count for logging. We do NOT
  // short-circuit on an empty set: with reconcile semantics an empty enabled
  // set means "prune everything", so the materialize call below must still run
  // to clear out any now-stale skills the managed dir is holding.
  const enabled = await skills.list({
    workspaceId: config.defaultWorkspaceId,
    enabled: true,
  });

  // Reuse the SDK's materialize (writes SKILL.md + files; Node-only). Pass
  // `enabled: true` so it re-lists only the enabled subset — disabled skills
  // must never land in the managed dir — and `reconcile: true` so any managed
  // skill dir not in that enabled set is pruned. `targetDir` MUST be the
  // dedicated, exclusively-MemoryLayer dir (the SDK only prunes there).
  await skills.materialize(targetDir, {
    workspaceId: config.defaultWorkspaceId,
    enabled: true,
    reconcile: true,
  });
  info(`skills sync: reconciled ${enabled.length} enabled skill(s) into managed dir ${targetDir}`);
}

/**
 * Register the skills-sync hook on OpenClaw's `agent:bootstrap` lifecycle event.
 *
 * No-op (returns without registering) when `config.skillsSyncEnabled` is false,
 * so the default-OFF flag is honoured at registration time.
 */
export function registerSkillsSync(
  api: OpenClawPluginApi,
  params: SkillsSyncParams,
): void {
  const warn =
    params.warn ?? ((msg, err) => console.warn(`[memorylayer-plugin] ${msg}`, err ?? ""));

  if (!params.config.skillsSyncEnabled) {
    return; // opt-in: default OFF
  }

  api.registerHook(
    AGENT_BOOTSTRAP_EVENT,
    async (event) => {
      // AgentBootstrapHookContext: { workspaceDir, bootstrapFiles, agentId, ... }
      if (event.type !== "agent" || event.action !== "bootstrap") return;
      const ctx = event.context as { workspaceDir?: string; agentId?: string };
      const workspaceDir = ctx.workspaceDir;
      if (!workspaceDir) {
        warn("skills sync: agent:bootstrap missing workspaceDir; skipping");
        return;
      }
      // Target the dedicated, exclusively-MemoryLayer managed dir (env override
      // or <workspaceDir>/.memorylayer-skills) — NOT the shared native skills/
      // dir. The sandbox-provider registers this same dir in OpenClaw's
      // skills.load.extraDirs so it gets discovered; reconcile is safe here
      // because the dir is ours alone.
      const targetDir = resolveManagedDir(workspaceDir);
      try {
        await materializeSkills(targetDir, params);
      } catch (err) {
        // Best-effort: skills sync is an optional enhancement. A failure here
        // (transport down, auth, fs perms) must not break agent bootstrap.
        warn(`skills sync failed for workspaceDir=${workspaceDir}`, err);
      }
    },
    { name: SKILLS_SYNC_HOOK_NAME },
  );
}
