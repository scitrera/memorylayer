// @ts-ignore
import path from "node:path";

import { describe, it, expect, vi } from "vitest";

import { materializeSkills, registerSkillsSync } from "../src/skills-sync.js";
import type { ConfiguredMemoryLayer } from "../src/client.js";
import type { PluginConfig } from "../src/config.js";

const baseConfig: PluginConfig = {
  aetherAddress: "localhost:50099",
  aetherWorkspace: "ws-prod",
  aetherImplementation: "openclaw-plugin",
  aetherSpecifier: "default",
  memorylayerTargetTopic: "sv::memorylayer::default",
  memorylayerBaseUrl: "http://memorylayer.aether-relay",
  memorylayerTimeoutMs: 30_000,
  tenantId: "acme",
  principalType: "user",
  principalId: "alice",
  defaultWorkspaceId: "ws-1",
  skillsSyncEnabled: true,
};

interface SkillsStub {
  list?: ReturnType<typeof vi.fn>;
  materialize?: ReturnType<typeof vi.fn>;
  withAuthority?: ReturnType<typeof vi.fn>;
}

/**
 * Builds a fake connection whose `client.skills` exposes the supplied stubs.
 * `withAuthority` defaults to returning the same skills object so the OBO path
 * and the non-OBO path share the same stubbed list/materialize.
 */
function fakeConnection(stub: SkillsStub): {
  getConnection: () => Promise<ConfiguredMemoryLayer>;
  skills: Required<Pick<SkillsStub, "list" | "materialize" | "withAuthority">>;
} {
  const list = stub.list ?? vi.fn().mockResolvedValue([{ id: "s1", name: "alpha" }]);
  const materialize = stub.materialize ?? vi.fn().mockResolvedValue(undefined);
  const skills: Record<string, unknown> = { list, materialize };
  const withAuthority = stub.withAuthority ?? vi.fn(() => skills);
  skills.withAuthority = withAuthority;
  const client = { skills };
  const conn = { client, aether: {} } as unknown as ConfiguredMemoryLayer;
  return {
    getConnection: async () => conn,
    skills: { list, materialize, withAuthority },
  };
}

function fakeApi(): {
  hooks: Map<string, Function>;
  opts: Map<string, unknown>;
  api: any;
} {
  const hooks = new Map<string, Function>();
  const opts = new Map<string, unknown>();
  const api = {
    registerHook: (events: string | string[], handler: Function, o?: unknown) => {
      const key = Array.isArray(events) ? events.join(",") : events;
      hooks.set(key, handler);
      opts.set(key, o);
    },
  };
  return { hooks, opts, api };
}

describe("materializeSkills", () => {
  it("lists enabled skills and materializes into the target dir", async () => {
    const list = vi.fn().mockResolvedValue([
      { id: "s1", name: "alpha" },
      { id: "s2", name: "beta" },
    ]);
    const materialize = vi.fn().mockResolvedValue(undefined);
    const { getConnection } = fakeConnection({ list, materialize });

    await materializeSkills("/ws/.memorylayer-skills", { getConnection, config: baseConfig, info: vi.fn() });

    expect(list).toHaveBeenCalledTimes(1);
    expect(list).toHaveBeenCalledWith({ workspaceId: "ws-1", enabled: true });
    expect(materialize).toHaveBeenCalledTimes(1);
    // enabled:true so the SDK's materialize re-lists only enabled skills —
    // disabled skills must never be written into the managed dir — and
    // reconcile:true so stale (disabled/deleted) skill dirs are pruned. The
    // target MUST be the dedicated managed dir, never the shared skills/ dir.
    expect(materialize).toHaveBeenCalledWith("/ws/.memorylayer-skills", {
      workspaceId: "ws-1",
      enabled: true,
      reconcile: true,
    });
  });

  it("scopes via withAuthority when a grantId is configured", async () => {
    const list = vi.fn().mockResolvedValue([{ id: "s1", name: "alpha" }]);
    const materialize = vi.fn().mockResolvedValue(undefined);
    const scoped = { list, materialize };
    const withAuthority = vi.fn(() => scoped);
    // Base skills object should NOT be used directly when grant is present.
    const baseList = vi.fn();
    const { getConnection } = fakeConnection({
      list: baseList,
      materialize: vi.fn(),
      withAuthority,
    });

    await materializeSkills("/ws/.memorylayer-skills", {
      getConnection,
      config: { ...baseConfig, grantId: "grant-abc" },
      info: vi.fn(),
    });

    expect(withAuthority).toHaveBeenCalledWith(
      { grantId: "grant-abc", subject: { type: "user", id: "alice" } },
      "ws-1",
    );
    expect(list).toHaveBeenCalledTimes(1);
    expect(materialize).toHaveBeenCalledTimes(1);
    expect(baseList).not.toHaveBeenCalled();
  });

  it("still reconciles (prunes) when there are no enabled skills", async () => {
    // An empty enabled set means "prune everything": with reconcile we must
    // still call materialize so the SDK clears any now-stale managed skill
    // dirs. Short-circuiting here would leave disabled/deleted skills on disk.
    const list = vi.fn().mockResolvedValue([]);
    const materialize = vi.fn().mockResolvedValue(undefined);
    const { getConnection } = fakeConnection({ list, materialize });

    await materializeSkills("/ws/.memorylayer-skills", { getConnection, config: baseConfig, info: vi.fn() });

    expect(list).toHaveBeenCalledTimes(1);
    expect(materialize).toHaveBeenCalledTimes(1);
    expect(materialize).toHaveBeenCalledWith("/ws/.memorylayer-skills", {
      workspaceId: "ws-1",
      enabled: true,
      reconcile: true,
    });
  });
});

describe("registerSkillsSync", () => {
  it("registers the agent:bootstrap hook with a name when sync is enabled", () => {
    const { hooks, opts, api } = fakeApi();
    const { getConnection } = fakeConnection({});
    registerSkillsSync(api, { getConnection, config: baseConfig });
    expect(hooks.has("agent:bootstrap")).toBe(true);
    expect(opts.get("agent:bootstrap")).toMatchObject({ name: expect.any(String) });
  });

  it("is a no-op when skillsSyncEnabled is false", () => {
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({});
    registerSkillsSync(api, {
      getConnection,
      config: { ...baseConfig, skillsSyncEnabled: false },
    });
    expect(hooks.size).toBe(0);
  });

  it("materializes into the dedicated <workspaceDir>/.memorylayer-skills dir on bootstrap", async () => {
    const list = vi.fn().mockResolvedValue([{ id: "s1", name: "alpha" }]);
    const materialize = vi.fn().mockResolvedValue(undefined);
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({ list, materialize });
    registerSkillsSync(api, { getConnection, config: baseConfig, info: vi.fn() });

    const handler = hooks.get("agent:bootstrap")!;
    await handler({
      type: "agent",
      action: "bootstrap",
      sessionKey: "agent:main:thread-1",
      context: { workspaceDir: "/workspaces/main", agentId: "main", bootstrapFiles: [] },
      timestamp: new Date(),
      messages: [],
    });

    expect(materialize).toHaveBeenCalledTimes(1);
    const [targetDir, opts] = materialize.mock.calls[0];
    // Dedicated managed dir (NOT the shared skills/), reconcile enabled.
    expect(targetDir).toBe(path.join("/workspaces/main", ".memorylayer-skills"));
    expect(opts).toMatchObject({ enabled: true, reconcile: true });
  });

  it("honours MEMORYLAYER_SKILLS_DIR override for the managed dir", async () => {
    const list = vi.fn().mockResolvedValue([{ id: "s1", name: "alpha" }]);
    const materialize = vi.fn().mockResolvedValue(undefined);
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({ list, materialize });
    registerSkillsSync(api, { getConnection, config: baseConfig, info: vi.fn() });

    const prev = process.env.MEMORYLAYER_SKILLS_DIR;
    process.env.MEMORYLAYER_SKILLS_DIR = "/mnt/managed/ml-skills";
    try {
      const handler = hooks.get("agent:bootstrap")!;
      await handler({
        type: "agent",
        action: "bootstrap",
        sessionKey: "agent:main:thread-1",
        context: { workspaceDir: "/workspaces/main", agentId: "main", bootstrapFiles: [] },
        timestamp: new Date(),
        messages: [],
      });
    } finally {
      if (prev === undefined) delete process.env.MEMORYLAYER_SKILLS_DIR;
      else process.env.MEMORYLAYER_SKILLS_DIR = prev;
    }

    expect(materialize).toHaveBeenCalledTimes(1);
    const [targetDir] = materialize.mock.calls[0];
    expect(targetDir).toBe("/mnt/managed/ml-skills");
  });

  it("skips events that are not agent:bootstrap (defensive)", async () => {
    const materialize = vi.fn();
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({ materialize });
    registerSkillsSync(api, { getConnection, config: baseConfig, info: vi.fn() });

    const handler = hooks.get("agent:bootstrap")!;
    await handler({
      type: "message",
      action: "sent",
      sessionKey: "agent:main:thread-1",
      context: { workspaceDir: "/workspaces/main" },
      timestamp: new Date(),
      messages: [],
    });
    expect(materialize).not.toHaveBeenCalled();
  });

  it("warns and skips when workspaceDir is missing", async () => {
    const materialize = vi.fn();
    const warn = vi.fn();
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({ materialize });
    registerSkillsSync(api, { getConnection, config: baseConfig, warn, info: vi.fn() });

    const handler = hooks.get("agent:bootstrap")!;
    await handler({
      type: "agent",
      action: "bootstrap",
      sessionKey: "agent:main:thread-1",
      context: { agentId: "main", bootstrapFiles: [] },
      timestamp: new Date(),
      messages: [],
    });
    expect(materialize).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
  });

  it("swallows materialize errors and logs (best-effort)", async () => {
    const list = vi.fn().mockResolvedValue([{ id: "s1", name: "alpha" }]);
    const materialize = vi.fn().mockRejectedValue(new Error("fs perms"));
    const warn = vi.fn();
    const { hooks, api } = fakeApi();
    const { getConnection } = fakeConnection({ list, materialize });
    registerSkillsSync(api, { getConnection, config: baseConfig, warn, info: vi.fn() });

    const handler = hooks.get("agent:bootstrap")!;
    // Must not throw.
    await handler({
      type: "agent",
      action: "bootstrap",
      sessionKey: "agent:main:thread-1",
      context: { workspaceDir: "/workspaces/main", agentId: "main", bootstrapFiles: [] },
      timestamp: new Date(),
      messages: [],
    });
    expect(warn).toHaveBeenCalled();
  });
});
