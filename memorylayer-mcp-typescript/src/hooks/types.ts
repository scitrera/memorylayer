/**
 * Hook type definitions for MemoryLayer Claude Code hooks
 */

/** Hook event types that Claude Code can trigger */
export enum HookEvent {
  SessionStart = "SessionStart",
  Stop = "Stop",
  PreCompact = "PreCompact",
  PreToolUse = "PreToolUse",
  PostToolUse = "PostToolUse",
  UserPromptSubmit = "UserPromptSubmit"
}

/** Input provided by Claude Code to hook commands via stdin */
export interface HookInput {
  /** The hook event type */
  hook_type: HookEvent;
  /** Session ID if available */
  session_id?: string;
  /** Transcript context (recent messages) */
  transcript?: TranscriptMessage[];
  /** For tool hooks: the tool being used */
  tool_name?: string;
  /** For tool hooks: tool input arguments */
  tool_input?: Record<string, unknown>;
  /** For tool hooks: tool output (PostToolUse only) */
  tool_output?: string;
  /** For UserPromptSubmit: the user's prompt */
  user_prompt?: string;
}

export interface TranscriptMessage {
  role: "user" | "assistant";
  content: string;
}

/** Hook-specific output nested inside the response */
export interface HookSpecificOutput {
  /** Must match the hook event name */
  hookEventName: string;
  /** Content to inject into assistant context */
  additionalContext?: string;
}

/** Output from hook command, written to stdout as JSON */
export interface HookOutput {
  /** Decision: "approve", "block", or "skip" */
  decision?: "approve" | "block" | "skip";
  /** Reason for the decision */
  reason?: string;
  /** Hook-specific output containing additionalContext (proper format) */
  hookSpecificOutput?: HookSpecificOutput;
  /** Whether hook execution succeeded (for error reporting) */
  success?: boolean;
  /** Error message if failed */
  error?: string;
  /** Legacy: Content to inject (CLI converts to hookSpecificOutput) */
  additionalContext?: string;
  /** Legacy: Whether to block (CLI converts to decision) */
  block?: boolean;
  /** Legacy: Reason for blocking */
  blockReason?: string;
}

/** State tracked between hook invocations */
export interface HookState {
  /** Current session ID */
  sessionId?: string;
  /** Workspace ID being used */
  workspaceId?: string;
  /** Whether recall has been done this conversation turn */
  recallDoneThisTurn: boolean;
  /** Timestamp of last recall */
  lastRecallAt?: string;
  /** Query used for last recall */
  lastRecallQuery?: string;
  /** Queries already performed this turn (for query-aware dedup) */
  recallQueriesThisTurn?: string[];
  /** User's current topic from most recent prompt */
  currentTopic?: string;
}
