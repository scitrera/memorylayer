/**
 * Hook state management - persists state between hook invocations
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import type { HookState } from "./types.js";

const STATE_DIR = join(homedir(), ".memorylayer");
const STATE_FILE = join(STATE_DIR, "hook-state.json");

/** Default empty state */
const DEFAULT_STATE: HookState = {
  recallDoneThisTurn: false,
};

/**
 * Read current hook state from disk
 */
export function readHookState(): HookState {
  try {
    if (!existsSync(STATE_FILE)) {
      return { ...DEFAULT_STATE };
    }
    const data = readFileSync(STATE_FILE, "utf-8");
    return JSON.parse(data) as HookState;
  } catch {
    return { ...DEFAULT_STATE };
  }
}

/**
 * Write hook state to disk
 */
export function writeHookState(state: HookState): void {
  try {
    if (!existsSync(STATE_DIR)) {
      mkdirSync(STATE_DIR, { recursive: true });
    }
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (error) {
    console.error("Failed to write hook state:", error);
  }
}

/**
 * Mark that recall has been done this turn for a specific query
 */
export function markRecallDone(query: string): void {
  const state = readHookState();
  state.recallDoneThisTurn = true;
  state.lastRecallAt = new Date().toISOString();
  state.lastRecallQuery = query;
  const queries = state.recallQueriesThisTurn || [];
  queries.push(query);
  state.recallQueriesThisTurn = queries;
  writeHookState(state);
}

/**
 * Check if recall was already done this turn (any query)
 */
export function wasRecallDoneThisTurn(): boolean {
  const state = readHookState();
  return state.recallDoneThisTurn;
}

/**
 * Check if a specific query (or similar) was already recalled this turn.
 * Allows different queries to proceed even if a recall was already done.
 */
export function wasQueryRecalledThisTurn(query: string): boolean {
  const state = readHookState();
  const queries = state.recallQueriesThisTurn || [];
  const normalized = query.toLowerCase().trim().replace(/\s+/g, " ");
  return queries.some(q => q.toLowerCase().trim().replace(/\s+/g, " ") === normalized);
}

/**
 * Reset recall status for new turn
 */
export function resetRecallStatus(): void {
  const state = readHookState();
  state.recallDoneThisTurn = false;
  state.recallQueriesThisTurn = [];
  writeHookState(state);
}

/**
 * Store the user's current topic for cross-hook context
 */
export function setCurrentTopic(topic: string): void {
  const state = readHookState();
  state.currentTopic = topic;
  writeHookState(state);
}

/**
 * Get the user's current topic (set by UserPromptSubmit)
 */
export function getCurrentTopic(): string | undefined {
  return readHookState().currentTopic;
}

/**
 * Update workspace/session info
 */
export function updateSessionInfo(workspaceId: string, sessionId?: string): void {
  const state = readHookState();
  state.workspaceId = workspaceId;
  state.sessionId = sessionId;
  writeHookState(state);
}

/**
 * Get current workspace ID from state
 */
export function getWorkspaceId(): string | undefined {
  return readHookState().workspaceId;
}

/**
 * Get current session ID from state
 */
export function getSessionId(): string | undefined {
  return readHookState().sessionId;
}
