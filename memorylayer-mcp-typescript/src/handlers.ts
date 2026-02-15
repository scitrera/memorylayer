/**
 * MCP tool handler implementations
 */

import {MemoryLayerClient} from "./client.js";
import {MemoryType, MemorySubtype} from "./types.js";
import type {
    ContextExecInput,
    ContextInspectInput,
    ContextLoadInput,
    ContextInjectInput,
    ContextQueryInput,
    ContextRlmInput,
} from "./types.js";
import {SessionManager} from "./session.js";

export class MCPToolHandlers {
    constructor(
        private client: MemoryLayerClient,
        private sessionManager: SessionManager
    ) {
    }

    async handleMemoryRemember(args: Record<string, unknown>): Promise<string> {
        const content = args.content as string;
        if (!content) {
            throw new Error("content is required");
        }

        const memory = await this.client.remember({
            content,
            type: args.type as MemoryType | undefined,
            subtype: args.subtype as MemorySubtype | undefined,
            importance: args.importance as number | undefined,
            tags: args.tags as string[] | undefined
        });

        return JSON.stringify({
            success: true,
            memory_id: memory.id,
            type: memory.type,
            importance: memory.importance,
            tags: memory.tags,
            message: `Stored memory ${memory.id}`
        }, null, 2);
    }

    async handleMemoryRecall(args: Record<string, unknown>): Promise<string> {
        const query = args.query as string;
        if (!query) {
            throw new Error("query is required");
        }

        const result = await this.client.recall({
            query,
            types: args.types as MemoryType[] | undefined,
            tags: args.tags as string[] | undefined,
            limit: args.limit as number | undefined,
            min_relevance: args.min_relevance as number | undefined
        });

        const memoriesData = result.memories.map(memory => ({
            id: memory.id,
            content: memory.content,
            type: memory.type,
            subtype: memory.subtype,
            importance: memory.importance,
            tags: memory.tags,
            created_at: memory.created_at,
            access_count: memory.access_count
        }));

        return JSON.stringify({
            success: true,
            memories: memoriesData,
            total_count: result.total_count,
            search_latency_ms: result.search_latency_ms,
            mode_used: result.mode_used
        }, null, 2);
    }

    async handleMemoryReflect(args: Record<string, unknown>): Promise<string> {
        const query = args.query as string;
        if (!query) {
            throw new Error("query is required");
        }

        const result = await this.client.reflect({
            query,
            detail_level: args.detail_level as 'abstract' | 'overview' | 'full' | undefined,
            include_sources: args.include_sources as boolean | undefined,
            depth: args.depth as number | undefined
        });

        return JSON.stringify({
            success: true,
            reflection: result.reflection,
            source_memories: result.source_memories,
            tokens_processed: result.tokens_processed
        }, null, 2);
    }

    async handleMemoryForget(args: Record<string, unknown>): Promise<string> {
        const memoryId = args.memory_id as string;
        if (!memoryId) {
            throw new Error("memory_id is required");
        }

        const hard = args.hard as boolean || false;
        const reason = args.reason as string | undefined;

        const result = await this.client.forget(memoryId, hard, reason);

        return JSON.stringify({
            success: result.success,
            memory_id: memoryId,
            hard_delete: hard,
            reason: reason || "No reason provided",
            message: result.success
                ? `Forgot memory ${memoryId}`
                : `Failed to forget memory ${memoryId}`
        }, null, 2);
    }

    async handleMemoryAssociate(args: Record<string, unknown>): Promise<string> {
        const sourceId = args.source_id as string;
        const targetId = args.target_id as string;
        const relationship = args.relationship as string;

        if (!sourceId || !targetId || !relationship) {
            throw new Error("source_id, target_id, and relationship are required");
        }

        const association = await this.client.associate({
            source_id: sourceId,
            target_id: targetId,
            relationship,
            strength: args.strength as number | undefined
        });

        return JSON.stringify({
            success: true,
            association_id: association.id,
            source_id: association.source_id,
            target_id: association.target_id,
            relationship: association.relationship,
            strength: association.strength,
            message: `Created association ${association.id}`
        }, null, 2);
    }

    async handleMemoryBriefing(args: Record<string, unknown>): Promise<string> {
        const result = await this.client.getBriefing({
            timeWindowMinutes: args.time_window_minutes as number | undefined,
            detailLevel: args.detail_level as string | undefined,
            limit: args.limit as number | undefined,
            includeMemories: args.include_memories as boolean | undefined,
            includeContradictions: args.include_contradictions as boolean | undefined,
        });

        return JSON.stringify(result, null, 2);
    }

    async handleMemoryStatistics(args: Record<string, unknown>): Promise<string> {
        const includeBreakdown = args.include_breakdown as boolean !== false;

        const result = await this.client.getStatistics(includeBreakdown);

        return JSON.stringify(result, null, 2);
    }

    async handleMemoryGraphQuery(args: Record<string, unknown>): Promise<string> {
        const startMemoryId = args.start_memory_id as string;
        if (!startMemoryId) {
            throw new Error("start_memory_id is required");
        }

        const result = await this.client.graphQuery({
            start_memory_id: startMemoryId,
            relationship_types: args.relationship_types as string[] | undefined,
            max_depth: args.max_depth as number | undefined,
            direction: args.direction as "outgoing" | "incoming" | "both" | undefined,
            max_paths: args.max_paths as number | undefined
        });

        const pathsData = result.paths.map(path => ({
            nodes: path.nodes,
            edges: path.edges.map(edge => ({
                from: edge.source_id,
                to: edge.target_id,
                relationship: edge.relationship,
                strength: edge.strength
            })),
            total_strength: path.total_strength
        }));

        return JSON.stringify({
            success: true,
            paths: pathsData,
            total_paths: result.total_paths,
            unique_nodes: result.unique_nodes
        }, null, 2);
    }

    async handleMemoryAudit(args: Record<string, unknown>): Promise<string> {
        const memoryId = args.memory_id as string | undefined;
        const autoResolve = args.auto_resolve as boolean || false;

        const result = await this.client.auditMemories(memoryId, autoResolve);

        return JSON.stringify(result, null, 2);
    }

    // TODO: handleMemoryCompress - commented out until compression logic is implemented
    // async handleMemoryCompress(args: Record<string, unknown>): Promise<string> {
    //   const result = await this.client.compressMemories({
    //     olderThanDays: args.older_than_days as number | undefined,
    //     minAccessCount: args.min_access_count as number | undefined,
    //     preserveImportant: args.preserve_important as boolean | undefined,
    //     dryRun: args.dry_run as boolean | undefined
    //   });
    //
    //   return JSON.stringify(result, null, 2);
    // }

    // ============================================================================
    // Session Management Handlers
    // ============================================================================

    async handleMemorySessionStart(args: Record<string, unknown>): Promise<string> {
        if (!this.sessionManager.isEnabled) {
            return JSON.stringify({
                success: false,
                error: "Session mode is not enabled"
            }, null, 2);
        }

        // If a session already exists, return it instead of creating a duplicate
        const existing = this.sessionManager.currentSession;
        if (existing) {
            console.error(`Session already active: ${existing.id} (server: ${existing.serverSessionId}), reusing`);
            return JSON.stringify({
                success: true,
                session_id: existing.id,
                workspace_id: existing.workspaceId,
                server_session_id: existing.serverSessionId,
                created_at: existing.createdAt.toISOString(),
                reused: true,
                message: "Existing session reused. If resuming after context compaction, call memory_context_inspect to see existing sandbox variables."
            }, null, 2);
        }

        // Get workspace ID from client
        const workspaceId = this.client.getWorkspaceId();

        // Start server-side session if client supports it
        let serverSessionId: string | undefined;
        try {
            console.error(`Creating server session for workspace: ${workspaceId}`);
            const serverSession = await this.client.startSession({
                ttl_seconds: this.sessionManager.getTtlSeconds(),
                metadata: args.metadata as Record<string, unknown> | undefined
            });
            serverSessionId = serverSession.session_id;
            // Set session on SDK client so subsequent requests include X-Session-ID header,
            // enabling correct workspace resolution on the server
            this.client.setSessionId(serverSessionId);
            console.error(`Server session created: ${serverSessionId} for workspace: ${workspaceId}`);
        } catch (error) {
            // Server may not support sessions yet, continue with local-only
            // This is a critical issue - without a session, workspace resolution falls back to _default!
            console.error(`WARNING: Server session creation failed for workspace ${workspaceId}:`, error instanceof Error ? error.message : error);
            console.error("Operations will fall back to _default workspace without a valid session!");
        }

        const session = this.sessionManager.startSession(workspaceId, serverSessionId);

        return JSON.stringify({
            success: true,
            session_id: session.id,
            workspace_id: session.workspaceId,
            server_session_id: session.serverSessionId,
            created_at: session.createdAt.toISOString(),
            message: "Session started. If resuming after context compaction, call memory_context_inspect to see existing sandbox variables."
        }, null, 2);
    }

    async handleMemorySessionEnd(args: Record<string, unknown>): Promise<string> {
        if (!this.sessionManager.isEnabled) {
            return JSON.stringify({
                success: false,
                error: "Session mode is not enabled"
            }, null, 2);
        }

        const session = this.sessionManager.currentSession;
        if (!session) {
            return JSON.stringify({
                success: false,
                error: "No active session"
            }, null, 2);
        }

        const commit = args.commit !== false;
        const importanceThreshold = (args.importance_threshold as number) ?? 0.5;

        let commitResult: { memories_extracted?: number } = {};

        // If committing, send working memory to server for extraction
        if (commit && session.serverSessionId) {
            try {
                // Gather working memory as context for extraction
                const workingMemory = this.sessionManager.getAllWorkingMemory();
                const contextData = workingMemory.map(entry => ({
                    key: entry.key,
                    value: entry.value,
                    category: (entry.value as { category?: string })?.category
                }));

                commitResult = await this.client.endSession(session.serverSessionId, {
                    commit: true,
                    importance_threshold: importanceThreshold,
                    working_memory: contextData
                });
            } catch (error) {
                console.error("Server session commit failed:", error);
            }
        }

        this.sessionManager.markCommitted();
        const endedSession = this.sessionManager.endSession();
        // Ensure SDK client session is cleared even if server commit was skipped
        this.client.clearSessionId();

        return JSON.stringify({
            success: true,
            session_id: endedSession?.id,
            committed: commit,
            working_memory_count: endedSession?.workingMemory.size ?? 0,
            memories_extracted: commitResult.memories_extracted ?? 0,
            message: commit ? "Session ended and committed" : "Session ended without commit"
        }, null, 2);
    }

    async handleMemorySessionCommit(args: Record<string, unknown>): Promise<string> {
        if (!this.sessionManager.isEnabled) {
            return JSON.stringify({
                success: false,
                error: "Session mode is not enabled"
            }, null, 2);
        }

        const session = this.sessionManager.currentSession;
        if (!session) {
            return JSON.stringify({
                success: false,
                error: "No active session"
            }, null, 2);
        }

        if (!session.serverSessionId) {
            return JSON.stringify({
                success: false,
                error: "No server session available for commit"
            }, null, 2);
        }

        const importanceThreshold = (args.importance_threshold as number) ?? 0.5;
        const clearAfterCommit = (args.clear_after_commit as boolean) ?? false;

        try {
            const commitResult = await this.client.commitSession(session.serverSessionId, {
                importance_threshold: importanceThreshold
            });

            // Optionally clear local working memory after commit
            if (clearAfterCommit) {
                this.sessionManager.clearWorkingMemory();
            }

            // Mark session as committed (but don't end it)
            this.sessionManager.markCommitted();

            return JSON.stringify({
                success: true,
                session_id: session.id,
                server_session_id: session.serverSessionId,
                memories_extracted: commitResult.memories_extracted,
                memories_created: commitResult.memories_created,
                working_memory_cleared: clearAfterCommit,
                session_still_active: true,
                message: "Working memory committed to long-term storage (session still active)"
            }, null, 2);
        } catch (error) {
            return JSON.stringify({
                success: false,
                error: `Commit failed: ${error instanceof Error ? error.message : error}`
            }, null, 2);
        }
    }

    async handleMemorySessionStatus(_args: Record<string, unknown>): Promise<string> {
        if (!this.sessionManager.isEnabled) {
            return JSON.stringify({
                enabled: false,
                message: "Session mode is not enabled"
            }, null, 2);
        }

        const summary = this.sessionManager.getSessionSummary();

        return JSON.stringify({
            enabled: true,
            ...summary
        }, null, 2);
    }

    // ============================================================================
    // Context Environment Handlers
    // ============================================================================

    private ensureActiveSession(): void {
        if (!this.sessionManager.hasActiveSession) {
            throw new Error("No active session. Call memory_session_start first.");
        }
    }

    async handleMemoryContextExec(args: Record<string, unknown>): Promise<string> {
        const code = args.code as string;
        if (!code) {
            throw new Error("code is required");
        }

        this.ensureActiveSession();

        const input: ContextExecInput = {
            code,
            result_var: args.result_var as string | undefined,
            return_result: args.return_result as boolean | undefined,
            max_return_chars: args.max_return_chars as number | undefined,
        };

        const result = await this.client.contextExec(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextInspect(args: Record<string, unknown>): Promise<string> {
        this.ensureActiveSession();

        const input: ContextInspectInput = {
            variable: args.variable as string | undefined,
            preview_chars: args.preview_chars as number | undefined,
        };

        const result = await this.client.contextInspect(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextLoad(args: Record<string, unknown>): Promise<string> {
        const varName = args.var as string;
        const query = args.query as string;
        if (!varName || !query) {
            throw new Error("var and query are required");
        }

        this.ensureActiveSession();

        const input: ContextLoadInput = {
            var: varName,
            query,
            limit: args.limit as number | undefined,
            types: args.types as string[] | undefined,
            tags: args.tags as string[] | undefined,
            min_relevance: args.min_relevance as number | undefined,
            include_embeddings: args.include_embeddings as boolean | undefined,
        };

        const result = await this.client.contextLoad(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextInject(args: Record<string, unknown>): Promise<string> {
        const key = args.key as string;
        const value = args.value as string;
        if (!key || value === undefined) {
            throw new Error("key and value are required");
        }

        this.ensureActiveSession();

        const input: ContextInjectInput = {
            key,
            value,
            parse_json: args.parse_json as boolean | undefined,
        };

        const result = await this.client.contextInject(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextQuery(args: Record<string, unknown>): Promise<string> {
        const prompt = args.prompt as string;
        const variables = args.variables as string[];
        if (!prompt || !variables) {
            throw new Error("prompt and variables are required");
        }

        this.ensureActiveSession();

        const input: ContextQueryInput = {
            prompt,
            variables,
            max_context_chars: args.max_context_chars as number | undefined,
            result_var: args.result_var as string | undefined,
        };

        const result = await this.client.contextQuery(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextRlm(args: Record<string, unknown>): Promise<string> {
        const goal = args.goal as string;
        if (!goal) {
            throw new Error("goal is required");
        }

        this.ensureActiveSession();

        const input: ContextRlmInput = {
            goal,
            memory_query: args.memory_query as string | undefined,
            memory_limit: args.memory_limit as number | undefined,
            max_iterations: args.max_iterations as number | undefined,
            variables: args.variables as string[] | undefined,
            result_var: args.result_var as string | undefined,
            detail_level: args.detail_level as "brief" | "standard" | "detailed" | undefined,
        };

        const result = await this.client.contextRlm(input);
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextStatus(_args: Record<string, unknown>): Promise<string> {
        this.ensureActiveSession();

        const result = await this.client.contextStatus();
        return JSON.stringify(result, null, 2);
    }

    async handleMemoryContextCheckpoint(_args: Record<string, unknown>): Promise<string> {
        this.ensureActiveSession();

        // Call the checkpoint endpoint on the server
        await this.client.contextCheckpoint();
        return JSON.stringify({
            success: true,
            message: "Checkpoint completed"
        }, null, 2);
    }
}
