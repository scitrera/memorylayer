/**
 * Session state management for MCP server.
 *
 * Each MCP server instance (spawned per Claude Code window) maintains
 * a single session. This provides working memory that persists across
 * tool calls within a Claude Code session.
 */

export interface WorkingMemoryEntry {
  key: string;
  value: unknown;
  createdAt: Date;
  updatedAt: Date;
}

export interface SessionState {
  id: string;
  workspaceId: string;
  createdAt: Date;
  workingMemory: Map<string, WorkingMemoryEntry>;
  /** Whether this session has been committed to long-term storage */
  committed: boolean;
  /** Server-side session ID (from MemoryLayer API) */
  serverSessionId?: string;
}

export interface SessionConfig {
  /** Enable session tracking (default: true) */
  enabled: boolean;
  /** Auto-create session on first tool call (default: true) */
  autoCreate: boolean;
  /** Session TTL in seconds for server-side session (default: 3600) */
  ttlSeconds: number;
}

export class SessionManager {
  private session: SessionState | null = null;
  private config: SessionConfig;

  constructor(config: Partial<SessionConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      autoCreate: config.autoCreate ?? true,
      ttlSeconds: config.ttlSeconds ?? 3600,
    };
  }

  get isEnabled(): boolean {
    return this.config.enabled;
  }

  get hasActiveSession(): boolean {
    return this.session !== null;
  }

  get currentSession(): SessionState | null {
    return this.session;
  }

  /**
   * Start a new session. Replaces any existing session.
   */
  startSession(workspaceId: string, serverSessionId?: string): SessionState {
    const sessionId = `local_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    this.session = {
      id: sessionId,
      workspaceId,
      createdAt: new Date(),
      workingMemory: new Map(),
      committed: false,
      serverSessionId,
    };

    console.error(`Session started: ${sessionId} (workspace: ${workspaceId})`);
    return this.session;
  }

  /**
   * End the current session.
   */
  endSession(): SessionState | null {
    const session = this.session;
    if (session) {
      console.error(`Session ended: ${session.id}`);
    }
    this.session = null;
    return session;
  }

  /**
   * Mark session as committed.
   */
  markCommitted(): void {
    if (this.session) {
      this.session.committed = true;
    }
  }

  /**
   * Get all working memory entries.
   */
  getAllWorkingMemory(): WorkingMemoryEntry[] {
    if (!this.session) {
      return [];
    }
    return Array.from(this.session.workingMemory.values());
  }

  /**
   * Clear all working memory.
   */
  clearWorkingMemory(): void {
    this.session?.workingMemory.clear();
  }

  /**
   * Get session summary for briefing.
   */
  getSessionSummary(): Record<string, unknown> {
    if (!this.session) {
      return {
        active: false,
      };
    }

    return {
      active: true,
      id: this.session.id,
      workspaceId: this.session.workspaceId,
      createdAt: this.session.createdAt.toISOString(),
      workingMemoryCount: this.session.workingMemory.size,
      committed: this.session.committed,
      serverSessionId: this.session.serverSessionId,
    };
  }

  /**
   * Get TTL config for server-side session creation.
   */
  getTtlSeconds(): number {
    return this.config.ttlSeconds;
  }
}
