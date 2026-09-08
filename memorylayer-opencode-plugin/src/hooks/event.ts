/**
 * Event hook for MemoryLayer OpenCode plugin.
 *
 * Handles context compaction by preserving critical memory state
 * and committing working memory before context is lost.
 */

import { createHash } from "crypto";
import { getClient } from "../shared/client.js";
import { acknowledgeCheckpointBoundary, getCheckpointBoundary } from "../shared/state.js";

interface CompactionInput {
  sessionID: string;
  transcript?: string;
  messages?: unknown[];
}

const CAPTURE_CHUNK_BYTES = 900_000;
const MAX_ATTEMPTS = 3;

async function retry<T>(operation: () => Promise<T>): Promise<T> {
  let lastError: unknown;
  const deadline = Date.now() + 5_000;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < MAX_ATTEMPTS && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, attempt * 100));
      } else {
        break;
      }
    }
  }
  throw lastError;
}

function transcriptPayload(input: CompactionInput): string | null {
  if (typeof input.transcript === "string" && input.transcript.length > 0) {
    return input.transcript;
  }
  if (Array.isArray(input.messages) && input.messages.length > 0) {
    return input.messages.map(message => JSON.stringify(message)).join("\n") + "\n";
  }
  return null;
}

async function captureTranscript(input: CompactionInput, serverSessionId: string): Promise<number> {
  const payload = transcriptPayload(input);
  if (!payload) return 0;
  const client = getClient();
  const allBytes = Buffer.from(payload, "utf-8");
  const transcriptKey = `opencode:${input.sessionID}`;
  let boundary = getCheckpointBoundary(input.sessionID, transcriptKey);
  if (boundary > allBytes.length) boundary = 0;
  let captured = 0;

  while (boundary < allBytes.length) {
    let end = Math.min(boundary + CAPTURE_CHUNK_BYTES, allBytes.length);
    while (end > boundary && end < allBytes.length && (allBytes[end] & 0xc0) === 0x80) end -= 1;
    if (end === boundary) throw new Error("unable to find a complete UTF-8 boundary");
    const segment = allBytes.subarray(boundary, end).toString("utf-8");
    const contentHash = createHash("sha256").update(Buffer.from(segment, "utf-8")).digest("hex");
    const idempotencyKey = `opencode:${input.sessionID}:${end}:${contentHash}`;
    await retry(() => client.createCheckpoint(serverSessionId, {
      transcript_segment: segment,
      content_hash: contentHash,
      idempotency_key: idempotencyKey,
      source_kind: "opencode_transcript",
      source_boundary: end,
    }));
    acknowledgeCheckpointBoundary(input.sessionID, transcriptKey, end);
    captured += end - boundary;
    boundary = end;
  }
  return captured;
}

/**
 * Handle session compacting — commit working memory and checkpoint sandbox.
 *
 * OpenCode's `experimental.session.compacting` hook is the equivalent
 * of Claude Code's PreCompact. We commit working memory and return
 * context strings that survive the compaction.
 */
export async function handleCompacting(input: CompactionInput): Promise<string[]> {
  const context: string[] = [];

  const client = getClient();
  const clientSessionId = client.getSessionId();

  if (clientSessionId) {
    try {
      const captured = await captureTranscript(input, clientSessionId);
      if (captured > 0) {
        context.push(`[MemoryLayer] Durably captured ${captured} new transcript bytes before compaction.`);
      }
    } catch (error) {
      console.error("[compacting] raw transcript capture failed; compaction will continue:", error instanceof Error ? error.message : error);
    }

    // Commit working memory to long-term storage before compaction
    try {
      await retry(() => client.commitSession(clientSessionId, { importance_threshold: 0.3 }));
      context.push(
        "[MemoryLayer] Working memory committed to long-term storage before compaction. " +
        "Use `memory_recall` to retrieve prior context. " +
        "Use `memory_context_inspect` to check server-side sandbox variables."
      );
    } catch {
      // Commit may fail, but we should still try to preserve context
    }

    // Checkpoint sandbox state if active
    try {
      const status = await client.contextStatus() as { exists?: boolean; variable_count?: number };
      if (status.exists && (status.variable_count ?? 0) > 0) {
        await retry(() => client.contextCheckpoint());
        context.push(
          "[MemoryLayer] Server-side sandbox state checkpointed. " +
          "Variables persist across compaction — use `memory_context_inspect` to re-orient."
        );
      }
    } catch {
      // Sandbox checkpoint is best-effort
    }
  }

  // Always include recovery instructions
  if (context.length === 0) {
    context.push(
      "[MemoryLayer] Context compaction occurred. Use `memory_recall` to retrieve " +
      "prior context and `memory_context_inspect` to check sandbox state."
    );
  }

  return context;
}
