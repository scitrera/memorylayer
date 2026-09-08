/**
 * Message persistence hook — every outbound agent message → MemoryLayer thread.
 *
 * The Aether↔WS sidecar bridge addresses a sandbox by its `sessionKey` shaped
 * as `agent:<agent-id>:<thread_id>` (per OpenClaw's gateway protocol). We
 * extract `thread_id` here, then call `MemoryLayer.appendMessages(threadId, ...)`
 * so MemoryLayer becomes the authoritative record of every agent reply.
 *
 * Thread creation is the app-server's job (it owns first contact when a user
 * opens a thread from the frontend). This hook is defensive: on 404 we log
 * and continue — the agent's reply already streamed to the user; missing
 * persistence is an operational issue, not a turn-blocking one.
 *
 * Plan: Phase 4, step 2.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";

import type { ConfiguredMemoryLayer } from "./client.js";
import type { PluginConfig } from "./config.js";

/** sessionKey format produced by the sidecar bridge. */
const SESSION_KEY_PREFIX = "agent:";

/**
 * Internal hook key for OpenClaw's "an outbound message was sent" lifecycle
 * event. OpenClaw normalises internal hook keys as `${type}:${action}` (see
 * openclaw runtime `registerInternalHook` / `runInternalHooks`), so the
 * MessageSent event (`type:"message"`, `action:"sent"`) is keyed `message:sent`.
 */
const MESSAGE_SENT_EVENT = "message:sent";

/** Unique hook registration name (required by OpenClaw's plugin hook loader). */
const MESSAGE_PERSISTENCE_HOOK_NAME = "memorylayer-message-persistence";

/** Parses `agent:<agent-id>:<thread_id>` into `thread_id`. Returns null on mismatch. */
export function threadIdFromSessionKey(sessionKey: string | undefined): string | null {
  if (!sessionKey || !sessionKey.startsWith(SESSION_KEY_PREFIX)) return null;
  // Skip "agent:" prefix; first ":" splits agent-id from thread_id.
  const afterPrefix = sessionKey.slice(SESSION_KEY_PREFIX.length);
  const colonIdx = afterPrefix.indexOf(":");
  if (colonIdx <= 0) return null;
  const threadId = afterPrefix.slice(colonIdx + 1);
  return threadId.length > 0 ? threadId : null;
}

export interface RegisterMessagePersistenceParams {
  getConnection: () => Promise<ConfiguredMemoryLayer>;
  config: PluginConfig;
  /** Optional log sink — defaults to console.warn. */
  warn?: (msg: string, err?: unknown) => void;
}

export function registerMessagePersistence(
  api: OpenClawPluginApi,
  params: RegisterMessagePersistenceParams,
): void {
  const warn = params.warn ?? ((msg, err) => console.warn(`[memorylayer-plugin] ${msg}`, err ?? ""));

  api.registerHook(MESSAGE_SENT_EVENT, async (event) => {
    // OpenClaw normalises every hook into InternalHookEvent: {type, action,
    // sessionKey, context, ...}. For "message:sent" the context shape is
    // MessageSentHookContext (see openclaw/src/hooks/internal-hooks.ts).
    if (event.type !== "message" || event.action !== "sent") return;
    const ctx = event.context as {
      to?: string;
      content?: string;
      success?: boolean;
      messageId?: string;
      conversationId?: string;
      channelId?: string;
    };
    if (!ctx.success) return; // skip failed sends; user-side recovery is separate
    if (!ctx.content) return; // nothing to record
    const threadId = threadIdFromSessionKey(event.sessionKey);
    if (!threadId) {
      warn(`message_sent: could not parse thread_id from sessionKey=${event.sessionKey}; skipping persistence`);
      return;
    }
    try {
      const { client } = await params.getConnection();
      await client.appendMessages(threadId, [
        {
          role: "assistant",
          content: ctx.content,
          metadata: {
            messageId: ctx.messageId,
            conversationId: ctx.conversationId,
            channelId: ctx.channelId,
            // Timestamp is on the InternalHookEvent itself.
            sentAt: event.timestamp instanceof Date ? event.timestamp.toISOString() : undefined,
          },
        },
      ]);
    } catch (err) {
      // Defensive: agent's reply already streamed; persistence is best-effort.
      // App-server is responsible for thread creation before first append; a
      // 404 here means we raced or app-server skipped that.
      warn(`appendMessages failed for thread_id=${threadId}`, err);
    }
  }, { name: MESSAGE_PERSISTENCE_HOOK_NAME });
}
