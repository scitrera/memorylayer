import { describe, it, expect, vi } from "vitest";

import {
  registerMessagePersistence,
  threadIdFromSessionKey,
} from "../src/persistence.js";
import type { ConfiguredMemoryLayer } from "../src/client.js";
import type { PluginConfig } from "../src/config.js";

describe("threadIdFromSessionKey", () => {
  it("parses agent:<id>:<thread> correctly", () => {
    expect(threadIdFromSessionKey("agent:main:thread-abc")).toBe("thread-abc");
    expect(threadIdFromSessionKey("agent:ops:incident-42")).toBe("incident-42");
  });

  it("preserves colons inside thread_id", () => {
    // Defensive: thread_ids that contain ':' should be preserved verbatim
    // after the agent:<id>: prefix.
    expect(threadIdFromSessionKey("agent:main:tenant:user:thr")).toBe("tenant:user:thr");
  });

  it("returns null for missing or malformed input", () => {
    expect(threadIdFromSessionKey(undefined)).toBeNull();
    expect(threadIdFromSessionKey("")).toBeNull();
    expect(threadIdFromSessionKey("main:thread")).toBeNull(); // missing agent: prefix
    expect(threadIdFromSessionKey("agent:")).toBeNull();
    expect(threadIdFromSessionKey("agent:main:")).toBeNull(); // empty thread_id
    expect(threadIdFromSessionKey("agent:main")).toBeNull(); // no colon after id
  });
});

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
  skillsSyncEnabled: false,
};

function fakeApi(): {
  hooks: Map<string, Function>;
  opts: Map<string, { name?: string } | undefined>;
  api: any;
} {
  const hooks = new Map<string, Function>();
  const opts = new Map<string, { name?: string } | undefined>();
  const api = {
    registerHook: (event: string, handler: Function, hookOpts?: { name?: string }) => {
      hooks.set(event, handler);
      opts.set(event, hookOpts);
    },
  };
  return { hooks, opts, api };
}

function fakeConnection(stub: { appendMessages?: Function }): () => Promise<ConfiguredMemoryLayer> {
  const client = { appendMessages: stub.appendMessages ?? vi.fn() };
  const conn = { client, aether: {} } as unknown as ConfiguredMemoryLayer;
  return async () => conn;
}

describe("registerMessagePersistence", () => {
  it("subscribes to the message:sent hook with a stable registration name", () => {
    const { hooks, opts, api } = fakeApi();
    registerMessagePersistence(api, { getConnection: fakeConnection({}), config: baseConfig });
    // OpenClaw keys internal hooks as `${type}:${action}`; the underscore form
    // never matches the dispatcher, so the colon form is required.
    expect(hooks.has("message:sent")).toBe(true);
    expect(hooks.has("message_sent")).toBe(false);
    // OpenClaw's plugin hook loader throws ("hook registration missing name")
    // unless a name is supplied.
    expect(opts.get("message:sent")?.name).toBe("memorylayer-message-persistence");
  });

  it("appends successful assistant messages to MemoryLayer", async () => {
    const appendMessages = vi.fn().mockResolvedValue({});
    const { hooks, api } = fakeApi();
    registerMessagePersistence(api, {
      getConnection: fakeConnection({ appendMessages }),
      config: baseConfig,
    });
    const handler = hooks.get("message:sent")!;
    await handler({
      type: "message",
      action: "sent",
      sessionKey: "agent:main:thread-xyz",
      context: { to: "user-alice", content: "hi!", success: true, messageId: "m-1", channelId: "chat" },
      timestamp: new Date("2026-05-25T12:00:00Z"),
      messages: [],
    });
    expect(appendMessages).toHaveBeenCalledTimes(1);
    const [threadId, msgs] = appendMessages.mock.calls[0];
    expect(threadId).toBe("thread-xyz");
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("assistant");
    expect(msgs[0].content).toBe("hi!");
    expect(msgs[0].metadata.messageId).toBe("m-1");
    expect(msgs[0].metadata.sentAt).toBe("2026-05-25T12:00:00.000Z");
  });

  it("skips failed sends", async () => {
    const appendMessages = vi.fn();
    const { hooks, api } = fakeApi();
    registerMessagePersistence(api, {
      getConnection: fakeConnection({ appendMessages }),
      config: baseConfig,
    });
    const handler = hooks.get("message:sent")!;
    await handler({
      type: "message",
      action: "sent",
      sessionKey: "agent:main:t-1",
      context: { to: "u", content: "x", success: false, error: "boom", channelId: "chat" },
      timestamp: new Date(),
      messages: [],
    });
    expect(appendMessages).not.toHaveBeenCalled();
  });

  it("skips when sessionKey is missing or unparseable", async () => {
    const appendMessages = vi.fn();
    const warn = vi.fn();
    const { hooks, api } = fakeApi();
    registerMessagePersistence(api, {
      getConnection: fakeConnection({ appendMessages }),
      config: baseConfig,
      warn,
    });
    const handler = hooks.get("message:sent")!;
    await handler({
      type: "message",
      action: "sent",
      sessionKey: "garbage",
      context: { to: "u", content: "x", success: true, channelId: "chat" },
      timestamp: new Date(),
      messages: [],
    });
    expect(appendMessages).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
  });

  it("skips events from non-message hooks (defensive)", async () => {
    const appendMessages = vi.fn();
    const { hooks, api } = fakeApi();
    registerMessagePersistence(api, {
      getConnection: fakeConnection({ appendMessages }),
      config: baseConfig,
    });
    const handler = hooks.get("message:sent")!;
    await handler({
      type: "session",
      action: "sent",
      sessionKey: "agent:main:t-1",
      context: { content: "x" },
      timestamp: new Date(),
      messages: [],
    });
    expect(appendMessages).not.toHaveBeenCalled();
  });

  it("swallows appendMessages errors and logs (best-effort)", async () => {
    const appendMessages = vi.fn().mockRejectedValue(new Error("404 thread missing"));
    const warn = vi.fn();
    const { hooks, api } = fakeApi();
    registerMessagePersistence(api, {
      getConnection: fakeConnection({ appendMessages }),
      config: baseConfig,
      warn,
    });
    const handler = hooks.get("message:sent")!;
    // Should not throw.
    await handler({
      type: "message",
      action: "sent",
      sessionKey: "agent:main:t-missing",
      context: { to: "u", content: "x", success: true, channelId: "chat" },
      timestamp: new Date(),
      messages: [],
    });
    expect(warn).toHaveBeenCalled();
  });
});
