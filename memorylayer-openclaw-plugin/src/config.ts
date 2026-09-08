/**
 * Plugin configuration sourced from environment variables.
 *
 * The sandbox-provider sets these when launching the OpenClaw container so
 * the plugin knows (a) where the Aether sidecar is, (b) which MemoryLayer
 * topic to target, and (c) what principal it's acting on behalf of.
 *
 * Per the workclaw shift plan, the plugin holds no real credentials —
 * the sidecar injects the proper Aether identity upstream.
 */

export interface PluginConfig {
  aetherAddress: string;
  aetherWorkspace: string;
  aetherImplementation: string;
  aetherSpecifier: string;

  memorylayerTargetTopic: string;
  memorylayerBaseUrl: string;
  memorylayerTimeoutMs: number;

  tenantId: string;
  principalType: "user" | "agent";
  principalId: string;

  defaultWorkspaceId?: string;
  grantId?: string;

  /**
   * Opt-in: materialize MemoryLayer skills into the OpenClaw-watched workspace
   * skills directory so OpenClaw discovers + hot-reloads them natively.
   * Default OFF. Enable with `MEMORYLAYER_SKILLS_SYNC=1` (or `true`).
   */
  skillsSyncEnabled: boolean;
}

const ENV = {
  AETHER_ADDRESS: "AETHER_GATEWAY_ADDRESS",
  AETHER_WORKSPACE: "AETHER_WORKSPACE",
  AETHER_IMPLEMENTATION: "AETHER_IMPLEMENTATION",
  AETHER_SPECIFIER: "AETHER_SPECIFIER",
  MEMORYLAYER_TARGET_TOPIC: "MEMORYLAYER_TARGET_TOPIC",
  MEMORYLAYER_BASE_URL: "MEMORYLAYER_BASE_URL",
  MEMORYLAYER_TIMEOUT_MS: "MEMORYLAYER_TIMEOUT_MS",
  WORKCLAW_TENANT_ID: "WORKCLAW_TENANT_ID",
  WORKCLAW_PRINCIPAL_TYPE: "WORKCLAW_PRINCIPAL_TYPE",
  WORKCLAW_PRINCIPAL_ID: "WORKCLAW_PRINCIPAL_ID",
  WORKCLAW_DEFAULT_WORKSPACE: "WORKCLAW_DEFAULT_WORKSPACE",
  WORKCLAW_GRANT_ID: "WORKCLAW_GRANT_ID",
  MEMORYLAYER_SKILLS_SYNC: "MEMORYLAYER_SKILLS_SYNC",
} as const;

export function resolveConfig(env: NodeJS.ProcessEnv = process.env): PluginConfig {
  const principalType = req(env, ENV.WORKCLAW_PRINCIPAL_TYPE);
  if (principalType !== "user" && principalType !== "agent") {
    throw new Error(
      `[memorylayer-openclaw-plugin] ${ENV.WORKCLAW_PRINCIPAL_TYPE} must be "user" or "agent", got: ${principalType}`,
    );
  }
  return {
    aetherAddress: req(env, ENV.AETHER_ADDRESS),
    aetherWorkspace: req(env, ENV.AETHER_WORKSPACE),
    aetherImplementation: req(env, ENV.AETHER_IMPLEMENTATION),
    aetherSpecifier: env[ENV.AETHER_SPECIFIER] ?? "default",
    memorylayerTargetTopic: req(env, ENV.MEMORYLAYER_TARGET_TOPIC),
    // Base URL is mostly nominal — AetherFetchTransport tunnels regardless —
    // but the SDK still wants a syntactically valid URL.
    memorylayerBaseUrl: env[ENV.MEMORYLAYER_BASE_URL] ?? "http://memorylayer.aether-relay",
    memorylayerTimeoutMs: parseIntOr(env[ENV.MEMORYLAYER_TIMEOUT_MS], 30_000),
    tenantId: req(env, ENV.WORKCLAW_TENANT_ID),
    principalType,
    principalId: req(env, ENV.WORKCLAW_PRINCIPAL_ID),
    defaultWorkspaceId: env[ENV.WORKCLAW_DEFAULT_WORKSPACE],
    grantId: env[ENV.WORKCLAW_GRANT_ID],
    skillsSyncEnabled: parseBool(env[ENV.MEMORYLAYER_SKILLS_SYNC]),
  };
}

/** Truthy iff the value is `1` or `true` (case-insensitive). Default false. */
function parseBool(value: string | undefined): boolean {
  if (!value) return false;
  const v = value.trim().toLowerCase();
  return v === "1" || v === "true";
}

function req(env: NodeJS.ProcessEnv, name: string): string {
  const v = env[name];
  if (!v) {
    throw new Error(`[memorylayer-openclaw-plugin] missing required env var: ${name}`);
  }
  return v;
}

function parseIntOr(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}
