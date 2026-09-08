import { describe, it, expect } from "vitest";

import { resolveConfig } from "../src/config.js";

describe("resolveConfig", () => {
  const baseEnv = {
    AETHER_GATEWAY_ADDRESS: "localhost:50099",
    AETHER_WORKSPACE: "ws-prod",
    AETHER_IMPLEMENTATION: "openclaw-plugin",
    MEMORYLAYER_TARGET_TOPIC: "sv::memorylayer::default",
    WORKCLAW_TENANT_ID: "acme",
    WORKCLAW_PRINCIPAL_TYPE: "user",
    WORKCLAW_PRINCIPAL_ID: "alice",
  } satisfies NodeJS.ProcessEnv;

  it("populates required fields", () => {
    const cfg = resolveConfig({ ...baseEnv });
    expect(cfg.aetherAddress).toBe("localhost:50099");
    expect(cfg.aetherWorkspace).toBe("ws-prod");
    expect(cfg.aetherImplementation).toBe("openclaw-plugin");
    expect(cfg.aetherSpecifier).toBe("default");
    expect(cfg.memorylayerTargetTopic).toBe("sv::memorylayer::default");
    expect(cfg.tenantId).toBe("acme");
    expect(cfg.principalType).toBe("user");
    expect(cfg.principalId).toBe("alice");
  });

  it("applies defaults for optional fields", () => {
    const cfg = resolveConfig({ ...baseEnv });
    expect(cfg.aetherSpecifier).toBe("default");
    expect(cfg.memorylayerBaseUrl).toBe("http://memorylayer.aether-relay");
    expect(cfg.memorylayerTimeoutMs).toBe(30_000);
    expect(cfg.defaultWorkspaceId).toBeUndefined();
    expect(cfg.grantId).toBeUndefined();
    expect(cfg.skillsSyncEnabled).toBe(false);
  });

  it("enables skills sync only for truthy MEMORYLAYER_SKILLS_SYNC values", () => {
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "1" }).skillsSyncEnabled).toBe(true);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "true" }).skillsSyncEnabled).toBe(true);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "TRUE" }).skillsSyncEnabled).toBe(true);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "0" }).skillsSyncEnabled).toBe(false);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "no" }).skillsSyncEnabled).toBe(false);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_SKILLS_SYNC: "" }).skillsSyncEnabled).toBe(false);
  });

  it("honours overrides", () => {
    const cfg = resolveConfig({
      ...baseEnv,
      AETHER_SPECIFIER: "tenant-acme",
      MEMORYLAYER_BASE_URL: "http://ml.local",
      MEMORYLAYER_TIMEOUT_MS: "10000",
      WORKCLAW_DEFAULT_WORKSPACE: "ws-1",
      WORKCLAW_GRANT_ID: "grant-abc",
    });
    expect(cfg.aetherSpecifier).toBe("tenant-acme");
    expect(cfg.memorylayerBaseUrl).toBe("http://ml.local");
    expect(cfg.memorylayerTimeoutMs).toBe(10_000);
    expect(cfg.defaultWorkspaceId).toBe("ws-1");
    expect(cfg.grantId).toBe("grant-abc");
  });

  it("rejects invalid principal_type", () => {
    expect(() =>
      resolveConfig({ ...baseEnv, WORKCLAW_PRINCIPAL_TYPE: "service" }),
    ).toThrow(/principal_type|must be "user"/i);
  });

  it("rejects missing required env", () => {
    const incomplete = { ...baseEnv } as Record<string, string | undefined>;
    delete incomplete["AETHER_GATEWAY_ADDRESS"];
    expect(() => resolveConfig(incomplete)).toThrow(/missing required env var: AETHER_GATEWAY_ADDRESS/);
  });

  it("falls back to default timeout when invalid number given", () => {
    const cfg = resolveConfig({ ...baseEnv, MEMORYLAYER_TIMEOUT_MS: "not-a-number" });
    expect(cfg.memorylayerTimeoutMs).toBe(30_000);
  });

  it("falls back to default timeout for zero / negative values", () => {
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_TIMEOUT_MS: "0" }).memorylayerTimeoutMs).toBe(30_000);
    expect(resolveConfig({ ...baseEnv, MEMORYLAYER_TIMEOUT_MS: "-1" }).memorylayerTimeoutMs).toBe(30_000);
  });

  it("accepts agent principal type", () => {
    const cfg = resolveConfig({ ...baseEnv, WORKCLAW_PRINCIPAL_TYPE: "agent", WORKCLAW_PRINCIPAL_ID: "bot-1" });
    expect(cfg.principalType).toBe("agent");
    expect(cfg.principalId).toBe("bot-1");
  });
});
