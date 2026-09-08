/** Durable, deterministic pre-compaction capture for Claude Code transcripts. */

import { closeSync, existsSync, fstatSync, openSync, readSync } from "fs";
import { createHash } from "crypto";
import type { HookInput, HookOutput } from "../types.js";
import { getClient } from "../client.js";
import { acknowledgeCheckpointBoundary, getCheckpointBoundary } from "../state.js";

const CAPTURE_CHUNK_BYTES = 900_000;
const MAX_ATTEMPTS = 3;

function decodeUtf8Prefix(buffer: Buffer): { content: string; byteCount: number } {
  for (let trim = 0; trim <= Math.min(4, buffer.length); trim += 1) {
    const candidate = trim === 0 ? buffer : buffer.subarray(0, buffer.length - trim);
    try {
      return {
        content: new TextDecoder("utf-8", { fatal: true }).decode(candidate),
        byteCount: candidate.length,
      };
    } catch {
      // A chunk may end within one UTF-8 code point; try the preceding boundary.
    }
  }
  throw new Error("transcript contains malformed UTF-8");
}

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

async function captureTranscript(input: HookInput, sessionId: string): Promise<number> {
  const transcriptPath = input.transcript_path;
  if (!transcriptPath || !existsSync(transcriptPath)) {
    console.error("[pre-compact] transcript path unavailable; continuing with commit/checkpoint side effects");
    return 0;
  }

  const client = getClient();
  const descriptor = openSync(transcriptPath, "r");
  let captured = 0;
  try {
    const size = fstatSync(descriptor).size;
    let boundary = getCheckpointBoundary(sessionId, transcriptPath);
    if (boundary > size) boundary = 0;

    while (boundary < size) {
      const requested = Math.min(CAPTURE_CHUNK_BYTES, size - boundary);
      const buffer = Buffer.allocUnsafe(requested);
      const bytesRead = readSync(descriptor, buffer, 0, requested, boundary);
      if (bytesRead === 0) break;
      const decoded = decodeUtf8Prefix(buffer.subarray(0, bytesRead));
      if (decoded.byteCount === 0) throw new Error("unable to find a complete UTF-8 boundary");
      const nextBoundary = boundary + decoded.byteCount;
      const contentHash = createHash("sha256").update(Buffer.from(decoded.content, "utf-8")).digest("hex");
      const idempotencyKey = `claude-code:${sessionId}:${nextBoundary}:${contentHash}`;

      await retry(() => client.createCheckpoint(sessionId, {
        transcript_segment: decoded.content,
        content_hash: contentHash,
        idempotency_key: idempotencyKey,
        source_kind: "claude_code_transcript",
        source_boundary: nextBoundary,
      }));
      acknowledgeCheckpointBoundary(sessionId, transcriptPath, nextBoundary);
      captured += decoded.byteCount;
      boundary = nextBoundary;
    }
  } finally {
    closeSync(descriptor);
  }
  return captured;
}

/** Capture raw transcript first, then best-effort commit and sandbox checkpoint. */
export async function handlePreCompact(input: HookInput): Promise<HookOutput> {
  const client = getClient();
  const sessionId = client.getSessionId();
  if (!sessionId) {
    console.error("[pre-compact] no server session; MemoryLayer capture skipped without blocking compaction");
    return { success: true };
  }

  try {
    const captured = await captureTranscript(input, sessionId);
    console.error(`[pre-compact] durably captured ${captured} new transcript bytes`);
  } catch (error) {
    console.error("[pre-compact] raw capture failed; compaction will continue:", error instanceof Error ? error.message : error);
  }

  try {
    await retry(() => client.commitSession(sessionId, { importance_threshold: 0.3 }));
  } catch (error) {
    console.error("[pre-compact] working-memory commit failed:", error instanceof Error ? error.message : error);
  }

  try {
    const status = await client.contextStatus() as { exists?: boolean; variable_count?: number };
    if (status.exists && (status.variable_count ?? 0) > 0) {
      await retry(() => client.contextCheckpoint());
    }
  } catch (error) {
    console.error("[pre-compact] sandbox checkpoint failed:", error instanceof Error ? error.message : error);
  }

  return { success: true };
}
