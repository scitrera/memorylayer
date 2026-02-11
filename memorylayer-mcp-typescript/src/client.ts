/**
 * MCP-compatible adapter for the MemoryLayer SDK
 *
 * This module provides a thin compatibility layer between the MCP server's expectations
 * and the SDK's API surface. All operations are delegated to the SDK client.
 */

import {
  MemoryLayerClient as SDKClient,
  MemoryType,
  MemorySubtype,
  RecallMode,
  SearchTolerance
} from "@scitrera/memorylayer-sdk";
import type {
  Memory,
  RecallResult,
  Association,
  Session
} from "@scitrera/memorylayer-sdk";
import type {
  RememberInput,
  RecallInput,
  ReflectInput,
  AssociateInput,
  GraphQueryInput,
  GraphQueryResult,
  ToolResponse,
  ReflectResult,
  ContextExecInput,
  ContextInspectInput,
  ContextLoadInput,
  ContextInjectInput,
  ContextQueryInput,
  ContextRlmInput,
} from "./types.js";

export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  workspaceId?: string;
  timeout?: number;
}

/**
 * MCP-compatible client that wraps the SDK client
 */
export class MemoryLayerClient {
  private sdk: SDKClient;
  private workspaceId: string;
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = options.baseUrl || process.env.MEMORYLAYER_URL || "http://localhost:61001";
    this.apiKey = options.apiKey || process.env.MEMORYLAYER_API_KEY;
    this.workspaceId = options.workspaceId || process.env.MEMORYLAYER_WORKSPACE_ID || "_default";
    this.timeout = options.timeout || 30000;

    this.sdk = new SDKClient({
      baseUrl: this.baseUrl,
      apiKey: this.apiKey,
      workspaceId: this.workspaceId,
      timeout: this.timeout
    });
  }

  async remember(input: RememberInput): Promise<Memory> {
    return this.sdk.remember(input.content, {
      type: input.type as MemoryType | undefined,
      subtype: input.subtype as MemorySubtype | undefined,
      importance: input.importance,
      tags: input.tags,
      metadata: input.metadata,
      associations: input.associations,
      contextId: input.context_id
    });
  }

  async recall(input: RecallInput): Promise<RecallResult> {
    return this.sdk.recall(input.query, {
      types: input.types as MemoryType[] | undefined,
      subtypes: input.subtypes as MemorySubtype[] | undefined,
      tags: input.tags,
      contextId: input.context_id,
      mode: input.mode as RecallMode | undefined,
      tolerance: input.tolerance as SearchTolerance | undefined,
      limit: input.limit,
      minRelevance: input.min_relevance,
      includeAssociations: input.include_associations,
      traverseDepth: input.traverse_depth,
      maxExpansion: input.max_expansion,
      createdAfter: input.created_after ? new Date(input.created_after) : undefined,
      createdBefore: input.created_before ? new Date(input.created_before) : undefined,
      conversationContext: input.context,
      ragThreshold: input.rag_threshold,
      detailLevel: input.detail_level
    });
  }

  async reflect(input: ReflectInput): Promise<ReflectResult> {
    const result = await this.sdk.reflect(input.query, {
      detailLevel: input.detail_level,
      includeSources: input.include_sources,
      depth: input.depth,
      types: input.types as MemoryType[] | undefined,
      subtypes: input.subtypes as MemorySubtype[] | undefined,
      tags: input.tags,
      contextId: input.context_id
    });

    return {
      reflection: result.reflection,
      source_memories: result.source_memories,
      confidence: result.confidence ?? 0.8,
      tokens_processed: result.tokens_processed
    };
  }

  async forget(memoryId: string, hard: boolean = false, reason?: string): Promise<ToolResponse> {
    await this.sdk.forget(memoryId, hard);
    return {
      success: true,
      memory_id: memoryId,
      hard_delete: hard,
      reason: reason || "No reason provided"
    };
  }

  async associate(input: AssociateInput): Promise<Association> {
    return this.sdk.associate(
      input.source_id,
      input.target_id,
      input.relationship,
      input.strength
    );
  }

  async getMemory(memoryId: string): Promise<Memory> {
    return this.sdk.getMemory(memoryId);
  }

  async getBriefing(includeContradictions: boolean = true): Promise<ToolResponse> {
    const briefing = await this.sdk.getBriefing(24, includeContradictions);

    return {
      success: true,
      total_memories: briefing.workspace_summary?.total_memories ?? 0,
      total_associations: briefing.workspace_summary?.total_associations ?? 0,
      total_categories: briefing.workspace_summary?.total_categories ?? 0,
      memory_types: briefing.workspace_summary?.memory_types ?? {},
      active_topics: briefing.workspace_summary?.active_topics ?? [],
      recent_activity: briefing.recent_activity ?? [],
      contradictions_found: briefing.contradictions_detected?.length ?? 0,
    };
  }

  async getStatistics(includeBreakdown: boolean = true): Promise<{
    success: boolean;
    total_memories: number;
    total_associations: number;
    breakdown?: {
      by_type: Record<string, number>;
      by_subtype: Record<string, number>;
    };
  }> {
    // SDK doesn't have a dedicated statistics endpoint yet
    // For now, we'll return a basic response
    // TODO: Implement when SDK adds statistics endpoint
    return {
      success: true,
      total_memories: 0,
      total_associations: 0,
      breakdown: includeBreakdown ? {
        by_type: {},
        by_subtype: {}
      } : undefined
    };
  }

  async graphQuery(input: GraphQueryInput): Promise<GraphQueryResult> {
    return this.sdk.traverseGraph(input.start_memory_id, {
      relationshipTypes: input.relationship_types as string[] | undefined,
      maxDepth: input.max_depth,
      direction: input.direction,
      minStrength: input.min_strength,
      maxPaths: input.max_paths,
      maxNodes: input.max_nodes
    });
  }

  async auditMemories(_memoryId?: string, _autoResolve: boolean = false): Promise<ToolResponse> {
    // SDK doesn't have audit endpoint yet
    // TODO: Implement when SDK adds audit endpoint
    return {
      success: true,
      message: "Audit functionality not yet available in SDK"
    };
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  /**
   * Get the workspace ID this client is configured for.
   */
  getWorkspaceId(): string {
    return this.workspaceId;
  }

  /**
   * Get the current session ID from the SDK client.
   */
  getSessionId(): string | undefined {
    return this.sdk.getSessionId();
  }

  /**
   * Set the active session ID on the SDK client.
   * This will include the X-Session-ID header in subsequent requests.
   */
  setSessionId(sessionId: string): void {
    this.sdk.setSession(sessionId);
  }

  /**
   * Clear the active session ID from the SDK client.
   */
  clearSessionId(): void {
    this.sdk.clearSession();
  }

  /**
   * Start a server-side session for working memory tracking.
   *
   * The workspace is auto-created if it doesn't exist (OSS "just works" pattern).
   * The SDK's createSession automatically sets the session ID for subsequent requests.
   */
  async startSession(options: {
    ttl_seconds?: number;
    context_id?: string;
    metadata?: Record<string, unknown>;
  } = {}): Promise<{ session_id: string; session: Session }> {
    const response = await this.sdk.createSession({
      workspaceId: this.workspaceId,
      ttlSeconds: options.ttl_seconds ?? 3600,
      contextId: options.context_id,
      metadata: options.metadata,
    }, true); // autoSetSession = true

    return {
      session_id: response.session.id,
      session: response.session
    };
  }

  /**
   * End a server-side session, optionally committing working memory.
   */
  async endSession(sessionId: string, options: {
    commit?: boolean;
    importance_threshold?: number;
    working_memory?: Array<{ key: string; value: unknown; category?: string }>;
  } = {}): Promise<{ memories_extracted?: number; memories_created?: number }> {
    let memoriesCreated = 0;

    // Commit session before deletion if requested
    if (options.commit) {
      try {
        const commitResult = await this.sdk.commitSession(sessionId, {
          minImportance: options.importance_threshold ?? 0.5,
        });
        memoriesCreated = commitResult.memories_created;
      } catch (error) {
        console.warn("Failed to commit session:", error);
      }
    }

    // Delete the session
    try {
      await this.sdk.deleteSession(sessionId);
    } catch (error) {
      // Session may have already expired, that's okay
      console.warn("Failed to delete session (may have expired):", error);
    }

    // Clear the session from the SDK
    this.sdk.clearSession();

    return { memories_extracted: memoriesCreated, memories_created: memoriesCreated };
  }

  /**
   * Commit working memory to long-term storage WITHOUT ending the session.
   * Use this for checkpoints during long sessions.
   */
  async commitSession(sessionId: string, options: {
    importance_threshold?: number;
    clear_after_commit?: boolean;
  } = {}): Promise<{ memories_extracted: number; memories_created: number }> {
    const commitResult = await this.sdk.commitSession(sessionId, {
      minImportance: options.importance_threshold ?? 0.5,
    });

    return {
      memories_extracted: commitResult.memories_extracted ?? 0,
      memories_created: commitResult.memories_created ?? 0
    };
  }

  /**
   * Set working memory on the server-side session.
   */
  async setWorkingMemory(sessionId: string, key: string, value: unknown): Promise<void> {
    await this.sdk.setWorkingMemory(sessionId, key, value);
  }

  /**
   * Get working memory from the server-side session.
   */
  async getWorkingMemory(sessionId: string, key?: string): Promise<Record<string, unknown>> {
    return this.sdk.getWorkingMemory(sessionId, key);
  }

  /**
   * Touch session to extend its TTL.
   */
  async touchSession(sessionId: string): Promise<{ expires_at: string }> {
    const result = await this.sdk.touchSession(sessionId);
    return { expires_at: result.expires_at };
  }

  // ============================================================================
  // Context Environment (delegated to SDK)
  // ============================================================================

  /**
   * Execute Python code in the server-side sandbox.
   */
  async contextExec(input: ContextExecInput): Promise<Record<string, unknown>> {
    return this.sdk.contextExec(input.code, {
      resultVar: input.result_var,
      returnResult: input.return_result,
      maxReturnChars: input.max_return_chars,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Inspect variables in the server-side sandbox.
   */
  async contextInspect(input: ContextInspectInput): Promise<Record<string, unknown>> {
    return this.sdk.contextInspect({
      variable: input.variable,
      previewChars: input.preview_chars,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Load memories into a sandbox variable via semantic search.
   */
  async contextLoad(input: ContextLoadInput): Promise<Record<string, unknown>> {
    return this.sdk.contextLoad(input.var, input.query, {
      limit: input.limit,
      types: input.types,
      tags: input.tags,
      minRelevance: input.min_relevance,
      includeEmbeddings: input.include_embeddings,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Inject a value directly into the sandbox as a named variable.
   */
  async contextInject(input: ContextInjectInput): Promise<Record<string, unknown>> {
    return this.sdk.contextInject(input.key, input.value, {
      parseJson: input.parse_json,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Query the server-side LLM using sandbox variables as context.
   */
  async contextQuery(input: ContextQueryInput): Promise<Record<string, unknown>> {
    return this.sdk.contextQuery(input.prompt, input.variables, {
      maxContextChars: input.max_context_chars,
      resultVar: input.result_var,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Run a Recursive Language Model (RLM) loop on the server.
   */
  async contextRlm(input: ContextRlmInput): Promise<Record<string, unknown>> {
    return this.sdk.contextRlm(input.goal, {
      memoryQuery: input.memory_query,
      memoryLimit: input.memory_limit,
      maxIterations: input.max_iterations,
      variables: input.variables,
      resultVar: input.result_var,
      detailLevel: input.detail_level,
    }) as unknown as Record<string, unknown>;
  }

  /**
   * Get context environment status.
   */
  async contextStatus(): Promise<Record<string, unknown>> {
    return this.sdk.contextStatus() as unknown as Record<string, unknown>;
  }

  /**
   * Checkpoint the sandbox state for persistence.
   */
  async contextCheckpoint(): Promise<void> {
    await this.sdk.contextCheckpoint();
  }
}
