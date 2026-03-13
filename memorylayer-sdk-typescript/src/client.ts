import type {
  Memory, RecallResult, ReflectResult, Association, Session, SessionBriefing,
  Workspace, Context, RememberOptions, RecallOptions, ReflectOptions, ClientConfig,
  SessionCreateOptions, SessionStartResponse, CommitOptions, CommitResponse,
  GraphTraverseOptions, GraphQueryResult, BatchOperation, BatchResult,
  AssociationCreateOptions, WorkspaceSchema,
  ContextExecOptions, ContextExecResult, ContextInspectOptions, ContextInspectResult,
  ContextLoadOptions, ContextLoadResult, ContextInjectOptions, ContextInjectResult,
  ContextQueryOptions, ContextQueryResult, ContextRlmOptions, ContextRlmResult,
  ContextStatusResult, WorkspaceExportData, WorkspaceImportResult,
  DocumentInfo, JobInfo, DocumentUploadOptions, DocumentUploadResponse,
  PageSearchOptions, PageSearchResponse, PageListResponse, DocumentListResponse,
  ChatThread, ThreadCreateOptions, ThreadListOptions, MessageAppendInput,
  ThreadWithMessagesResponse, MessageListResponse, MessagesAppendResponse, DecomposeResponse,
} from "./types.js";
import { RelationshipType } from "./types.js";
import { MemoryLayerError, AuthenticationError, AuthorizationError, NotFoundError, ValidationError, EnterpriseRequiredError } from "./errors.js";

export class MemoryLayerClient {
  private baseUrl: string;
  private apiKey?: string;
  private workspaceId?: string;
  private sessionId?: string;
  private timeout: number;

  constructor(config: ClientConfig = {}) {
    this.baseUrl = config.baseUrl ?? "http://localhost:61001";
    this.apiKey = config.apiKey;
    this.workspaceId = config.workspaceId;
    this.sessionId = config.sessionId;
    this.timeout = config.timeout ?? 30000;
  }

  /**
   * Set the active session ID. All subsequent requests will include
   * this session ID in the X-Session-ID header, enabling session-based
   * workspace resolution.
   */
  setSession(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Clear the active session ID.
   */
  clearSession(): void {
    this.sessionId = undefined;
  }

  /**
   * Get the current session ID, if any.
   */
  getSessionId(): string | undefined {
    return this.sessionId;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    enterpriseFeature?: string,
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.sessionId) {
      headers["X-Session-ID"] = this.sessionId;
    }
    if (this.workspaceId) {
      headers["X-Workspace-ID"] = this.workspaceId;
    }

    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleError(response, enterpriseFeature);
      }

      if (response.status === 204) {
        return undefined as T;
      }

      return await response.json() as T;
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Request failed: ${error}`);
    }
  }

  private async handleError(response: Response, enterpriseFeature?: string): Promise<never> {
    const body = await response.json().catch(() => ({})) as any;
    const rawDetail = body.message ?? body.detail ?? response.statusText;
    const message = typeof rawDetail === 'string'
      ? rawDetail
      : JSON.stringify(rawDetail);

    switch (response.status) {
      case 401:
        throw new AuthenticationError(message);
      case 403:
        throw new AuthorizationError(message);
      case 404:
        if (enterpriseFeature) {
          throw new EnterpriseRequiredError(enterpriseFeature);
        }
        throw new NotFoundError(message);
      case 400:
      case 422:
        throw new ValidationError(message, body.details);
      default:
        throw new MemoryLayerError(message, response.status);
    }
  }

  // Memory operations
  async remember(content: string, options: RememberOptions = {}): Promise<Memory> {
    const body = {
      content,
      // Include workspace_id as fallback if session is not set
      workspace_id: options.workspaceId ?? this.workspaceId,
      type: options.type,
      subtype: options.subtype,
      importance: options.importance ?? 0.5,
      tags: options.tags ?? [],
      metadata: options.metadata ?? {},
      associations: options.associations ?? [],
      context_id: options.contextId,
    };
    const response = await this.request<{ memory: Memory }>("POST", "/v1/memories", body);
    return response.memory;
  }

  async recall(query: string, options: RecallOptions = {}): Promise<RecallResult> {
    const body = {
      query,
      // Include workspace_id as fallback if session is not set
      // Server priority: 1) body.workspace_id, 2) session.workspace_id, 3) _default
      workspace_id: options.workspaceId ?? this.workspaceId,
      types: options.types ?? [],
      subtypes: options.subtypes ?? [],
      tags: options.tags ?? [],
      context_id: options.contextId,
      mode: options.mode,
      tolerance: options.tolerance,
      limit: options.limit ?? 10,
      min_relevance: options.minRelevance,
      recency_weight: options.recencyWeight,
      include_associations: options.includeAssociations,
      traverse_depth: options.traverseDepth,
      max_expansion: options.maxExpansion,
      created_after: options.createdAfter?.toISOString(),
      created_before: options.createdBefore?.toISOString(),
      context: options.conversationContext ?? [],
      rag_threshold: options.ragThreshold,
      detail_level: options.detailLevel,
    };
    return this.request<RecallResult>("POST", "/v1/memories/recall", body);
  }

  async reflect(query: string, options: ReflectOptions = {}): Promise<ReflectResult> {
    const body = {
      query,
      // Include workspace_id as fallback if session is not set
      workspace_id: options.workspaceId ?? this.workspaceId,
      detail_level: options.detailLevel,
      include_sources: options.includeSources ?? true,
      depth: options.depth ?? 2,
      types: options.types ?? [],
      subtypes: options.subtypes ?? [],
      tags: options.tags ?? [],
      context_id: options.contextId,
    };
    return this.request<ReflectResult>("POST", "/v1/memories/reflect", body);
  }

  async getMemory(memoryId: string): Promise<Memory> {
    const response = await this.request<{ memory: Memory }>("GET", `/v1/memories/${memoryId}`);
    return response.memory;
  }

  async updateMemory(memoryId: string, updates: Partial<RememberOptions> & { content?: string }): Promise<Memory> {
    const response = await this.request<{ memory: Memory }>("PUT", `/v1/memories/${memoryId}`, updates);
    return response.memory;
  }

  async forget(memoryId: string, hard = false): Promise<void> {
    await this.request<void>("DELETE", `/v1/memories/${memoryId}?hard=${hard}`);
  }

  async decay(memoryId: string, decayRate = 0.1): Promise<Memory> {
    const response = await this.request<{ memory: Memory }>(
      "POST",
      `/v1/memories/${memoryId}/decay`,
      { decay_rate: decayRate }
    );
    return response.memory;
  }

  // Association operations
  async associate(
    sourceId: string,
    targetId: string,
    relationship: RelationshipType,
    strength = 0.5
  ): Promise<Association> {
    const response = await this.request<{ association: Association }>(
      "POST",
      `/v1/memories/${sourceId}/associate`,
      { target_id: targetId, relationship, strength }
    );
    return response.association;
  }

  async getAssociations(memoryId: string, direction: "outgoing" | "incoming" | "both" = "both"): Promise<Association[]> {
    const response = await this.request<{ associations: Association[] }>(
      "GET",
      `/v1/memories/${memoryId}/associations?direction=${direction}`
    );
    return response.associations;
  }

  // Session operations
  /**
   * Create a new session.
   *
   * @param options Session creation options
   * @param autoSetSession If true (default), automatically set this session
   *                       as the active session for subsequent requests
   */
  async createSession(options: SessionCreateOptions = {}, autoSetSession = true): Promise<SessionStartResponse> {
    const body = {
      session_id: options.sessionId,
      workspace_id: options.workspaceId ?? this.workspaceId,
      ttl_seconds: options.ttlSeconds ?? 3600,
      metadata: options.metadata ?? {},
      context_id: options.contextId,
      working_memory: options.workingMemory ?? {},
      briefing: options.briefing ?? false,
      briefing_options: options.briefingOptions ? {
        lookback_hours: options.briefingOptions.lookbackHours,
        detail_level: options.briefingOptions.detailLevel,
        limit: options.briefingOptions.limit,
      } : undefined,
    };
    const response = await this.request<SessionStartResponse>("POST", "/v1/sessions", body);
    if (autoSetSession) {
      this.sessionId = response.session.id;
    }
    return response;
  }

  async listSessions(options?: {
    workspaceId?: string;
    contextId?: string;
    includeExpired?: boolean;
  }): Promise<Session[]> {
    const params = new URLSearchParams();
    const wsId = options?.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options?.contextId) params.set("context_id", options.contextId);
    if (options?.includeExpired) params.set("include_expired", "true");
    const query = params.toString();
    const response = await this.request<{ sessions: Session[]; total_count: number }>(
      "GET",
      `/v1/sessions${query ? `?${query}` : ""}`
    );
    return response.sessions;
  }

  async getSession(sessionId: string): Promise<Session> {
    const response = await this.request<{ session: Session }>("GET", `/v1/sessions/${sessionId}`);
    return response.session;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.request<void>("DELETE", `/v1/sessions/${sessionId}`);
  }

  async setWorkingMemory(sessionId: string, key: string, value: unknown): Promise<void> {
    await this.request<void>(
      "POST",
      `/v1/sessions/${sessionId}/memory`,
      { key, value }
    );
  }

  async getWorkingMemory(sessionId: string, key?: string): Promise<Record<string, unknown>> {
    const url = key
      ? `/v1/sessions/${sessionId}/memory?key=${key}`
      : `/v1/sessions/${sessionId}/memory`;
    return this.request<Record<string, unknown>>("GET", url);
  }

  async commitSession(sessionId: string, options: CommitOptions = {}): Promise<CommitResponse> {
    const body = {
      min_importance: options.minImportance ?? 0.5,
      deduplicate: options.deduplicate ?? true,
      categories: options.categories,
      max_memories: options.maxMemories ?? 50,
    };
    return this.request<CommitResponse>(
      "POST",
      `/v1/sessions/${sessionId}/commit`,
      body
    );
  }

  async touchSession(sessionId: string, ttlSeconds?: number): Promise<Session> {
    const response = await this.request<{ session: Session }>(
      "POST",
      `/v1/sessions/${sessionId}/touch`,
      ttlSeconds ? { ttl_seconds: ttlSeconds } : {}
    );
    return response.session;
  }

  async getBriefing(
    optionsOrLookbackHours?: number | {
      lookbackMinutes?: number;
      detailLevel?: string;
      limit?: number;
      includeMemories?: boolean;
      includeContradictions?: boolean;
    },
    includeContradictions?: boolean
  ): Promise<SessionBriefing> {
    let params: URLSearchParams;

    if (typeof optionsOrLookbackHours === 'object' && optionsOrLookbackHours !== null) {
      const opts = optionsOrLookbackHours;
      params = new URLSearchParams();
      if (opts.lookbackMinutes !== undefined) params.set('lookback_minutes', String(opts.lookbackMinutes));
      if (opts.detailLevel !== undefined) params.set('detail_level', opts.detailLevel);
      if (opts.limit !== undefined) params.set('limit', String(opts.limit));
      if (opts.includeMemories !== undefined) params.set('include_memories', String(opts.includeMemories));
      if (opts.includeContradictions !== undefined) params.set('include_contradictions', String(opts.includeContradictions));
    } else {
      // Backward compatible: getBriefing(24, true)
      const lookbackHours = optionsOrLookbackHours ?? 24;
      const ic = includeContradictions ?? true;
      params = new URLSearchParams();
      params.set('lookback_minutes', String(lookbackHours * 60));
      params.set('include_contradictions', String(ic));
    }

    // Always include workspace_id if configured
    if (this.workspaceId) {
      params.set('workspace_id', this.workspaceId);
    }

    const response = await this.request<{ briefing: SessionBriefing }>(
      "GET",
      `/v1/sessions/briefing?${params.toString()}`
    );
    return response.briefing;
  }

  // Workspace operations
  async createWorkspace(name: string, settings?: Record<string, unknown>): Promise<Workspace> {
    const response = await this.request<{ workspace: Workspace }>(
      "POST",
      "/v1/workspaces",
      { name, settings: settings ?? {} }
    );
    return response.workspace;
  }

  async getWorkspace(workspaceId?: string): Promise<Workspace> {
    const id = workspaceId ?? this.workspaceId;
    if (!id) throw new ValidationError("Workspace ID required");
    const response = await this.request<{ workspace: Workspace }>("GET", `/v1/workspaces/${id}`);
    return response.workspace;
  }

  async listWorkspaces(): Promise<Workspace[]> {
    const response = await this.request<{ workspaces: Workspace[] }>("GET", "/v1/workspaces");
    return response.workspaces;
  }

  async updateWorkspace(workspaceId: string, updates: { name?: string; settings?: Record<string, unknown> }): Promise<Workspace> {
    const response = await this.request<{ workspace: Workspace }>(
      "PUT",
      `/v1/workspaces/${workspaceId}`,
      updates
    );
    return response.workspace;
  }

  async createContext(name: string, description?: string, settings?: Record<string, unknown>): Promise<Context> {
    if (!this.workspaceId) throw new ValidationError("Workspace ID required");
    const response = await this.request<{ context: Context }>(
      "POST",
      `/v1/workspaces/${this.workspaceId}/contexts`,
      { name, description, settings: settings ?? {} }
    );
    return response.context;
  }

  async listContexts(): Promise<Context[]> {
    if (!this.workspaceId) throw new ValidationError("Workspace ID required");
    const response = await this.request<{ contexts: Context[] }>(
      "GET",
      `/v1/workspaces/${this.workspaceId}/contexts`
    );
    return response.contexts;
  }

  // Batch operations
  async batchMemories(operations: BatchOperation[]): Promise<BatchResult> {
    return this.request<BatchResult>("POST", "/v1/memories/batch", { operations });
  }

  // Memory trace
  async traceMemory(memoryId: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("GET", `/v1/memories/${memoryId}/trace`);
  }

  // Graph traversal - uses nested endpoint under memories
  async traverseGraph(startMemoryId: string, options: GraphTraverseOptions = {}): Promise<GraphQueryResult> {
    const body = {
      relationship_types: options.relationshipTypes,
      relationship_categories: options.relationshipCategories,
      max_depth: options.maxDepth ?? 3,
      direction: options.direction ?? "both",
      min_strength: options.minStrength ?? 0.0,
      max_paths: options.maxPaths ?? 100,
      max_nodes: options.maxNodes ?? 50,
    };
    return this.request<GraphQueryResult>("POST", `/v1/memories/${startMemoryId}/traverse`, body);
  }

  // Create association with full options - uses nested endpoint under memories
  async createAssociation(options: AssociationCreateOptions): Promise<Association> {
    const body = {
      target_id: options.targetId,
      relationship: options.relationship,
      strength: options.strength ?? 0.5,
      metadata: options.metadata ?? {},
    };
    const response = await this.request<{ association: Association }>(
      "POST",
      `/v1/memories/${options.sourceId}/associate`,
      body
    );
    return response.association;
  }

  // Workspace schema
  async getWorkspaceSchema(workspaceId?: string): Promise<WorkspaceSchema> {
    const id = workspaceId ?? this.workspaceId;
    if (!id) throw new ValidationError("Workspace ID required");
    return this.request<WorkspaceSchema>("GET", `/v1/workspaces/${id}/schema`);
  }

  async exportWorkspace(
    workspaceId?: string,
    options?: { includeAssociations?: boolean; offset?: number; limit?: number }
  ): Promise<WorkspaceExportData> {
    const id = workspaceId ?? this.workspaceId;
    if (!id) throw new ValidationError("Workspace ID required");
    const params = new URLSearchParams();
    if (options?.includeAssociations === false) {
      params.set('include_associations', 'false');
    }
    if (options?.offset !== undefined) {
      params.set('offset', String(options.offset));
    }
    if (options?.limit !== undefined) {
      params.set('limit', String(options.limit));
    }
    const query = params.toString() ? `?${params.toString()}` : '';

    // Fetch NDJSON response
    const url = `${this.baseUrl}/v1/workspaces/${id}/export${query}`;
    const headers: Record<string, string> = {};
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.sessionId) {
      headers["X-Session-ID"] = this.sessionId;
    }
    if (this.workspaceId) {
      headers["X-Workspace-ID"] = this.workspaceId;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleError(response);
      }

      const text = await response.text();

      // Parse NDJSON: split by newlines, parse each line
      const lines = text.trim().split('\n').filter(line => line.trim());
      let header: any = null;
      const memories: any[] = [];
      const associations: any[] = [];

      for (const line of lines) {
        const parsed = JSON.parse(line);
        if (parsed.type === 'header') {
          header = parsed;
        } else if (parsed.type === 'memory') {
          memories.push(parsed.data);
        } else if (parsed.type === 'association') {
          associations.push(parsed.data);
        }
      }

      // Build backward-compatible response
      return {
        version: header?.version || '1.0',
        workspace_id: header?.workspace_id || id,
        exported_at: header?.exported_at || new Date().toISOString(),
        total_memories: header?.total_memories || memories.length,
        total_associations: header?.total_associations || associations.length,
        memories,
        associations,
      };
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Export request failed: ${error}`);
    }
  }

  async *exportWorkspaceStream(
    workspaceId?: string,
    options?: { includeAssociations?: boolean; offset?: number; limit?: number }
  ): AsyncGenerator<Record<string, unknown>> {
    const id = workspaceId ?? this.workspaceId;
    if (!id) throw new ValidationError("Workspace ID required");
    const params = new URLSearchParams();
    if (options?.includeAssociations === false) {
      params.set('include_associations', 'false');
    }
    if (options?.offset !== undefined) {
      params.set('offset', String(options.offset));
    }
    if (options?.limit !== undefined) {
      params.set('limit', String(options.limit));
    }
    const query = params.toString() ? `?${params.toString()}` : '';

    const url = `${this.baseUrl}/v1/workspaces/${id}/export${query}`;
    const headers: Record<string, string> = {};
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.sessionId) {
      headers["X-Session-ID"] = this.sessionId;
    }
    if (this.workspaceId) {
      headers["X-Workspace-ID"] = this.workspaceId;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleError(response);
      }

      const text = await response.text();
      const lines = text.trim().split('\n').filter(line => line.trim());

      for (const line of lines) {
        yield JSON.parse(line);
      }
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Export stream failed: ${error}`);
    }
  }

  async importWorkspace(
    workspaceId: string,
    data: WorkspaceExportData
  ): Promise<WorkspaceImportResult> {
    const response = await this.request<WorkspaceImportResult>(
      "POST", `/v1/workspaces/${workspaceId}/import`,
      { data }
    );
    return response;
  }

  async importWorkspaceStream(
    workspaceId: string,
    ndjsonBody: string
  ): Promise<WorkspaceImportResult> {
    const url = `${this.baseUrl}/v1/workspaces/${workspaceId}/import`;
    const headers: Record<string, string> = {
      "Content-Type": "application/x-ndjson",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.sessionId) {
      headers["X-Session-ID"] = this.sessionId;
    }
    if (this.workspaceId) {
      headers["X-Workspace-ID"] = this.workspaceId;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: ndjsonBody,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        await this.handleError(response);
      }

      return await response.json() as WorkspaceImportResult;
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Import stream failed: ${error}`);
    }
  }

  // Context Environment operations

  async contextExec(code: string, options: ContextExecOptions = {}): Promise<ContextExecResult> {
    const body = {
      code,
      result_var: options.resultVar,
      return_result: options.returnResult ?? true,
      max_return_chars: options.maxReturnChars,
    };
    return this.request<ContextExecResult>("POST", "/v1/context/execute", body);
  }

  async contextInspect(options: ContextInspectOptions = {}): Promise<ContextInspectResult> {
    const params = new URLSearchParams();
    if (options.variable) params.set("variable", options.variable);
    if (options.previewChars !== undefined) params.set("preview_chars", String(options.previewChars));
    const query = params.toString();
    return this.request<ContextInspectResult>("POST", `/v1/context/inspect${query ? `?${query}` : ""}`);
  }

  async contextLoad(varName: string, query: string, options: ContextLoadOptions = {}): Promise<ContextLoadResult> {
    const body = {
      var: varName,
      query,
      limit: options.limit,
      types: options.types,
      tags: options.tags,
      min_relevance: options.minRelevance,
      include_embeddings: options.includeEmbeddings,
    };
    return this.request<ContextLoadResult>("POST", "/v1/context/load", body);
  }

  async contextInject(key: string, value: unknown, options: ContextInjectOptions = {}): Promise<ContextInjectResult> {
    const body = {
      key,
      value,
      parse_json: options.parseJson,
    };
    return this.request<ContextInjectResult>("POST", "/v1/context/inject", body);
  }

  async contextQuery(prompt: string, variables: string[], options: ContextQueryOptions = {}): Promise<ContextQueryResult> {
    const body = {
      prompt,
      variables,
      max_context_chars: options.maxContextChars,
      result_var: options.resultVar,
    };
    return this.request<ContextQueryResult>("POST", "/v1/context/query", body);
  }

  async contextRlm(goal: string, options: ContextRlmOptions = {}): Promise<ContextRlmResult> {
    const body = {
      goal,
      memory_query: options.memoryQuery,
      memory_limit: options.memoryLimit,
      max_iterations: options.maxIterations,
      variables: options.variables,
      result_var: options.resultVar,
      detail_level: options.detailLevel,
    };
    return this.request<ContextRlmResult>("POST", "/v1/context/rlm", body);
  }

  async contextStatus(): Promise<ContextStatusResult> {
    return this.request<ContextStatusResult>("GET", "/v1/context/status");
  }

  async contextCleanup(): Promise<void> {
    await this.request<void>("DELETE", "/v1/context/cleanup");
  }

  async contextCheckpoint(): Promise<void> {
    await this.request<void>("POST", "/v1/context/checkpoint");
  }

  // ------------------------------------------------------------------ //
  // Document operations (Enterprise)
  // ------------------------------------------------------------------ //

  /**
   * Upload a document for ingestion.
   *
   * Requires MemoryLayer Enterprise. On OSS servers this throws
   * `EnterpriseRequiredError`.
   */
  async uploadDocument(
    file: Blob | File,
    filename: string,
    options: DocumentUploadOptions = {},
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file, filename);
    if (options.targetContextId) formData.append("target_context_id", options.targetContextId);
    if (options.chunkingStrategy) formData.append("chunking_strategy", options.chunkingStrategy);
    if (options.chunkSize !== undefined) formData.append("chunk_size", String(options.chunkSize));
    if (options.chunkOverlap !== undefined) formData.append("chunk_overlap", String(options.chunkOverlap));
    if (options.importance !== undefined) formData.append("importance", String(options.importance));
    if (options.retainOriginal !== undefined) formData.append("retain_original", String(options.retainOriginal));

    const headers: Record<string, string> = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    if (this.sessionId) headers["X-Session-ID"] = this.sessionId;
    if (this.workspaceId) headers["X-Workspace-ID"] = this.workspaceId;

    const url = `${this.baseUrl}/v1/documents`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        await this.handleError(response, "Document ingestion");
      }
      return await response.json() as DocumentUploadResponse;
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Document upload failed: ${error}`);
    }
  }

  /**
   * List documents in the workspace.
   */
  async listDocuments(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<DocumentListResponse> {
    const params = new URLSearchParams();
    if (options?.status) params.set("status", options.status);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    return this.request<DocumentListResponse>(
      "GET",
      `/v1/documents${query ? `?${query}` : ""}`,
      undefined,
      "Document management",
    );
  }

  /**
   * Get document metadata and processing status.
   */
  async getDocument(documentId: string): Promise<DocumentInfo> {
    const response = await this.request<{ document: DocumentInfo } & DocumentInfo>(
      "GET",
      `/v1/documents/${documentId}`,
      undefined,
      "Document management",
    );
    return response;
  }

  /**
   * Delete a document and optionally its extracted memories.
   */
  async deleteDocument(documentId: string, deleteMemories = false): Promise<void> {
    await this.request<void>(
      "DELETE",
      `/v1/documents/${documentId}?delete_memories=${deleteMemories}`,
      undefined,
      "Document management",
    );
  }

  /**
   * Search document pages using ColPali MaxSim visual similarity.
   *
   * Requires MemoryLayer Enterprise.
   */
  async searchDocumentPages(query: string, options: PageSearchOptions = {}): Promise<PageSearchResponse> {
    const body = {
      query,
      limit: options.limit ?? 10,
      doc_ids: options.docIds,
    };
    return this.request<PageSearchResponse>(
      "POST",
      "/v1/documents/search",
      body,
      "Document page search",
    );
  }

  /**
   * Get all pages for a document.
   */
  async getDocumentPages(documentId: string): Promise<PageListResponse> {
    return this.request<PageListResponse>(
      "GET",
      `/v1/documents/${documentId}/pages`,
      undefined,
      "Document pages",
    );
  }

  /**
   * Get a page image as a Blob.
   */
  async getPageImage(documentId: string, pageId: string): Promise<Blob> {
    const headers: Record<string, string> = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    if (this.sessionId) headers["X-Session-ID"] = this.sessionId;
    if (this.workspaceId) headers["X-Workspace-ID"] = this.workspaceId;

    const url = `${this.baseUrl}/v1/documents/${documentId}/pages/${pageId}/image`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, { method: "GET", headers, signal: controller.signal });
      clearTimeout(timeoutId);
      if (!response.ok) {
        await this.handleError(response, "Document page images");
      }
      return await response.blob();
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Page image request failed: ${error}`);
    }
  }

  /**
   * Get ingestion job status.
   */
  async getJob(jobId: string): Promise<JobInfo> {
    return this.request<JobInfo>(
      "GET",
      `/v1/documents/jobs/${jobId}`,
      undefined,
      "Document ingestion jobs",
    );
  }

  /**
   * List ingestion jobs in the workspace.
   */
  async listJobs(options?: { status?: string; limit?: number }): Promise<{ jobs: JobInfo[] }> {
    const params = new URLSearchParams();
    if (options?.status) params.set("status", options.status);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    const query = params.toString();
    return this.request<{ jobs: JobInfo[] }>(
      "GET",
      `/v1/documents/jobs${query ? `?${query}` : ""}`,
      undefined,
      "Document ingestion jobs",
    );
  }

  /**
   * Cancel a running ingestion job.
   */
  async cancelJob(jobId: string): Promise<void> {
    await this.request<void>(
      "POST",
      `/v1/documents/jobs/${jobId}/cancel`,
      undefined,
      "Document ingestion jobs",
    );
  }

  /**
   * Reprocess a document with optionally different extraction options.
   */
  async reprocessDocument(documentId: string, options?: Partial<DocumentUploadOptions>): Promise<JobInfo> {
    const body: Record<string, unknown> = {};
    if (options?.targetContextId) body.target_context_id = options.targetContextId;
    if (options?.chunkingStrategy) body.chunking_strategy = options.chunkingStrategy;
    if (options?.chunkSize !== undefined) body.chunk_size = options.chunkSize;
    if (options?.chunkOverlap !== undefined) body.chunk_overlap = options.chunkOverlap;
    if (options?.importance !== undefined) body.importance = options.importance;
    return this.request<JobInfo>(
      "POST",
      `/v1/documents/${documentId}/reprocess`,
      Object.keys(body).length ? body : undefined,
      "Document reprocessing",
    );
  }

  // ------------------------------------------------------------------ //
  // Chat History operations
  // ------------------------------------------------------------------ //

  async createThread(options: ThreadCreateOptions = {}): Promise<ChatThread> {
    const body = {
      thread_id: options.threadId,
      workspace_id: options.workspaceId ?? this.workspaceId,
      user_id: options.userId,
      context_id: options.contextId,
      observer_id: options.observerId,
      subject_id: options.subjectId,
      title: options.title,
      metadata: options.metadata,
      expires_at: options.expiresAt,
    };
    const response = await this.request<{ thread: ChatThread }>("POST", "/v1/threads", body);
    return response.thread;
  }

  async listThreads(options: ThreadListOptions = {}): Promise<ChatThread[]> {
    const params = new URLSearchParams();
    const wsId = options.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options.userId) params.set("user_id", options.userId);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    const response = await this.request<{ threads: ChatThread[]; total_count: number }>(
      "GET",
      `/v1/threads${query ? `?${query}` : ""}`
    );
    return response.threads;
  }

  async getThread(threadId: string, workspaceId?: string): Promise<ChatThread> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    const response = await this.request<{ thread: ChatThread }>(
      "GET",
      `/v1/threads/${threadId}${query ? `?${query}` : ""}`
    );
    return response.thread;
  }

  async getThreadFull(
    threadId: string,
    options?: { workspaceId?: string; limit?: number; offset?: number; order?: "asc" | "desc" }
  ): Promise<ThreadWithMessagesResponse> {
    const params = new URLSearchParams();
    const wsId = options?.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));
    if (options?.order) params.set("order", options.order);
    const query = params.toString();
    return this.request<ThreadWithMessagesResponse>(
      "GET",
      `/v1/threads/${threadId}/full${query ? `?${query}` : ""}`
    );
  }

  async deleteThread(threadId: string, workspaceId?: string): Promise<void> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    await this.request<void>(
      "DELETE",
      `/v1/threads/${threadId}${query ? `?${query}` : ""}`
    );
  }

  async appendMessages(
    threadId: string,
    messages: MessageAppendInput[],
    workspaceId?: string
  ): Promise<MessagesAppendResponse> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    return this.request<MessagesAppendResponse>(
      "POST",
      `/v1/threads/${threadId}/messages${query ? `?${query}` : ""}`,
      { messages }
    );
  }

  async getMessages(
    threadId: string,
    options?: { workspaceId?: string; limit?: number; offset?: number; afterIndex?: number; order?: "asc" | "desc" }
  ): Promise<MessageListResponse> {
    const params = new URLSearchParams();
    const wsId = options?.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));
    if (options?.afterIndex !== undefined) params.set("after_index", String(options.afterIndex));
    if (options?.order) params.set("order", options.order);
    const query = params.toString();
    return this.request<MessageListResponse>(
      "GET",
      `/v1/threads/${threadId}/messages${query ? `?${query}` : ""}`
    );
  }

  async decomposeThread(threadId: string, workspaceId?: string): Promise<DecomposeResponse> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    return this.request<DecomposeResponse>(
      "POST",
      `/v1/threads/${threadId}/decompose${query ? `?${query}` : ""}`
    );
  }
}
