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
  DatasetInfo, DatasetJobInfo, DatasetUploadOptions, DatasetUploadResponse,
  DatasetListResponse, DatasetSliceOptions, DatasetSliceResult, DatasetMemoriesResponse,
  AuthorityContext,
  ApiToken, ApiTokenWithSecret, TokenCreateOptions, TokenListResponse,
  MemoryListOptions, MemoryListResponse,
  Entity, EntityListResponse, EntityResponse, EntityResolveResponse,
  EntityListOptions, EntityResolveOptions, EntityMergeOptions,
  AssociationUpdateOptions,
  ThreadUpdateOptions, UserThreadListOptions,
  SessionCheckpoint, ContextPack, ContextDelta, ContextPackOptions,
} from "./types.js";
import { RelationshipType } from "./types.js";
import { MemoryLayerError, AuthenticationError, AuthorizationError, NotFoundError, ValidationError, EnterpriseRequiredError, RateLimitError } from "./errors.js";
import { sleep } from "./utils.js";
import { SkillsNamespace } from "./skills.js";
import { McpServersNamespace } from "./mcp_servers.js";
import { KnowledgebaseNamespace } from "./knowledgebase.js";
import { RpgNamespace } from "./rpg.js";

export class MemoryLayerClient {
  private baseUrl: string;
  private apiKey?: string;
  private workspaceId?: string;
  private sessionId?: string;
  private timeout: number;
  private defaultAuthority?: AuthorityContext;
  private fetchImpl: typeof fetch;
  private maxRetries: number;
  private retryBaseDelay: number;

  /** Skills namespace — access via `client.skills.list(...)` etc. */
  readonly skills: SkillsNamespace;

  /** MCP Servers namespace — access via `client.mcpServers.list(...)` etc. */
  readonly mcpServers: McpServersNamespace;

  /** Knowledgebase namespace — access via `client.kb.get(...)` etc. */
  readonly kb: KnowledgebaseNamespace;

  /** Repository Planning Graph namespace — access via `client.rpg.sync(...)` etc. */
  readonly rpg: RpgNamespace;

  constructor(config: ClientConfig = {}) {
    this.baseUrl = config.baseUrl ?? "http://localhost:61001";
    this.apiKey = config.apiKey;
    this.workspaceId = config.workspaceId;
    this.sessionId = config.sessionId;
    this.timeout = config.timeout ?? 30000;
    this.defaultAuthority = config.defaultAuthority;
    this.maxRetries = config.maxRetries ?? 3;
    this.retryBaseDelay = config.retryBaseDelay ?? 500;
    // Bind to globalThis so the default impl preserves `this === undefined`
    // (fetch is sensitive to its receiver in some runtimes).
    this.fetchImpl = config.fetch ?? fetch.bind(globalThis);
    this.skills = new SkillsNamespace(this);
    this.mcpServers = new McpServersNamespace(this);
    this.kb = new KnowledgebaseNamespace(this);
    this.rpg = new RpgNamespace(this);
  }

  /**
   * Returns a lightweight proxy that sends OBO headers for the given grant/subject
   * on every request. The proxy is synchronous and reuses the parent client's
   * connection settings — safe for concurrent use across multiple subjects.
   */
  actingFor(opts: { grantId: string; subject: { type: string; id: string } }): OboProxy {
    return new OboProxy(this, { grantId: opts.grantId, subject: opts.subject });
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

  private buildAuthorityHeaders(authority?: AuthorityContext): Record<string, string> {
    const resolved = authority ?? this.defaultAuthority;
    if (!resolved) return {};
    const h: Record<string, string> = {
      "X-Aether-Grant-ID": resolved.grantId,
      "X-Aether-Authority-Mode": "on_behalf_of",
      "X-Aether-Subject-Type": resolved.subject.type,
      "X-Aether-Subject-ID": resolved.subject.id,
    };
    return h;
  }

  /**
   * Build the standard request headers (auth, session, workspace, OBO authority).
   * Centralized so every request helper applies identical header logic.
   *
   * Note: upload/stream paths (exportWorkspace, importWorkspaceStream,
   * uploadDocument, uploadDataset, getPageImage) call this method and therefore
   * intentionally inherit authority (X-Aether-*) headers when a defaultAuthority
   * is set — consistent with every other request type.
   *
   * @param includeContentType Whether to set `Content-Type: application/json`.
   *   Omit for multipart/form-data (browser sets the boundary) or NDJSON bodies
   *   that set their own content type.
   */
  private buildHeaders(
    authority?: AuthorityContext,
    includeContentType = true,
  ): Record<string, string> {
    const headers: Record<string, string> = {};
    if (includeContentType) {
      headers["Content-Type"] = "application/json";
    }
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.sessionId) {
      headers["X-Session-ID"] = this.sessionId;
    }
    if (this.workspaceId) {
      headers["X-Workspace-ID"] = this.workspaceId;
    }
    Object.assign(headers, this.buildAuthorityHeaders(authority));
    return headers;
  }

  /**
   * Determine whether a failed response is transient and should be retried.
   * Retries 429 (rate limit) and 5xx server errors. 501 is treated as a
   * permanent "not implemented" signal and is never retried.
   */
  private isRetryableStatus(status: number): boolean {
    if (status === 429) return true;
    if (status === 501) return false;
    return status >= 500 && status < 600;
  }

  /**
   * Parse a `Retry-After` header (delta-seconds or HTTP-date) into milliseconds.
   * Returns undefined when absent or unparseable.
   */
  private parseRetryAfter(response: Response): number | undefined {
    const raw = response.headers?.get?.("Retry-After");
    if (!raw) return undefined;
    const seconds = Number(raw);
    if (!Number.isNaN(seconds)) {
      return Math.max(0, seconds * 1000);
    }
    const dateMs = Date.parse(raw);
    if (!Number.isNaN(dateMs)) {
      return Math.max(0, dateMs - Date.now());
    }
    return undefined;
  }

  /**
   * Returns true when the HTTP method is safe to retry after a transient failure.
   * POST is excluded to prevent duplicate writes (e.g. duplicate memories/edges
   * after a post-commit 504). recall/reflect/mergeEntities are POST-but-read or
   * write-once, but are deliberately left non-retried for safety and consistency
   * with the Python SDK.
   */
  private isIdempotentMethod(method: string): boolean {
    return ["GET", "HEAD", "OPTIONS", "PUT", "DELETE", "PATCH"].includes(method.toUpperCase());
  }

  /**
   * Execute a fetch with bounded retry-with-backoff for transient failures
   * (5xx, 429). Honors the `Retry-After` header when present, otherwise uses
   * exponential backoff. Non-transient failures and successful responses are
   * returned/raised immediately.
   *
   * Retries are gated on method idempotency: only GET/HEAD/OPTIONS/PUT/DELETE/PATCH
   * are retried. POST is never retried automatically to avoid duplicate writes.
   *
   * This is the single network seam used by every request helper, so retry and
   * header handling stay consistent across the client.
   */
  private async executeFetch(
    url: string,
    init: RequestInit,
    enterpriseFeature?: string,
  ): Promise<Response> {
    const canRetry = this.isIdempotentMethod((init.method as string) ?? "GET");
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      try {
        const response = await this.fetchImpl(url, { ...init, signal: controller.signal });
        clearTimeout(timeoutId);

        if (response.ok) {
          return response;
        }

        // Transient server-side failure: retry if attempts remain AND method is idempotent.
        if (canRetry && attempt < this.maxRetries && this.isRetryableStatus(response.status)) {
          const retryAfter = this.parseRetryAfter(response);
          const delay = retryAfter ?? this.retryBaseDelay * Math.pow(2, attempt);
          await sleep(delay);
          continue;
        }

        // Permanent failure, non-idempotent method, or retries exhausted: surface a typed error.
        await this.handleError(response, enterpriseFeature);
      } catch (error) {
        clearTimeout(timeoutId);
        // Typed errors from handleError are terminal — never retry them.
        if (error instanceof MemoryLayerError) throw error;
        // Network/abort error: retry if attempts remain AND method is idempotent.
        lastError = error;
        if (canRetry && attempt < this.maxRetries) {
          await sleep(this.retryBaseDelay * Math.pow(2, attempt));
          continue;
        }
        throw new MemoryLayerError(`Request failed: ${error}`);
      }
    }
    // Unreachable in practice, but satisfies the type checker.
    throw new MemoryLayerError(`Request failed: ${lastError}`);
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    enterpriseFeature?: string,
    authority?: AuthorityContext,
    responseMode: "json" | "text" | "arraybuffer" = "json",
  ): Promise<T> {
    const headers = this.buildHeaders(authority);
    const url = `${this.baseUrl}${path}`;

    const response = await this.executeFetch(
      url,
      {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      },
      enterpriseFeature,
    );

    if (response.status === 204) {
      return undefined as T;
    }

    // Non-JSON endpoints (e.g. GET /v1/skills/{id}/manifest → text/markdown,
    // GET /v1/skills/{id}/files/{path} → text/x-python etc.) must be read as
    // raw text/bytes; calling response.json() on them throws. Default stays
    // JSON so existing callers are unaffected.
    if (responseMode === "text") {
      return await response.text() as T;
    }
    if (responseMode === "arraybuffer") {
      return await response.arrayBuffer() as T;
    }

    return await response.json() as T;
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
        throw new NotFoundError(message);
      case 400:
      case 422:
        throw new ValidationError(message, body.details);
      case 429: {
        const retryAfterMs = this.parseRetryAfter(response);
        throw new RateLimitError(message, retryAfterMs !== undefined ? retryAfterMs / 1000 : undefined);
      }
      case 501:
        if (enterpriseFeature) {
          throw new EnterpriseRequiredError(enterpriseFeature);
        }
        throw new NotFoundError(message);
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
      relations: options.relations ?? [],
      context_id: options.contextId,
      user_id: options.userId,
    };
    const response = await this.request<{ memory: Memory }>("POST", "/v1/memories", body, undefined, options.authority);
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
      offset: options.offset,
      event_after: options.eventAfter?.toISOString(),
      event_before: options.eventBefore?.toISOString(),
      time_order: options.timeOrder,
      include_global: options.includeGlobal,
      include_global_user: options.includeGlobalUser,
      context: options.conversationContext ?? [],
      rag_threshold: options.ragThreshold,
      detail_level: options.detailLevel,
      user_id: options.userId,
      budget_tokens: options.budgetTokens,
      include_confidence: options.includeConfidence ?? true,
      include_relations: options.includeRelations ?? true,
    };
    return this.request<RecallResult>("POST", "/v1/memories/recall", body, undefined, options.authority);
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
      user_id: options.userId,
    };
    return this.request<ReflectResult>("POST", "/v1/memories/reflect", body, undefined, options.authority);
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

  /**
   * Update an association's strength and/or metadata.
   *
   * `memoryId` must be one of the association's endpoints (source or target);
   * otherwise the server returns 404. Re-typing an edge is not supported —
   * delete and recreate instead.
   */
  async updateAssociation(
    memoryId: string,
    associationId: string,
    updates: AssociationUpdateOptions,
  ): Promise<void> {
    const body: Record<string, unknown> = {};
    if (updates.strength !== undefined) body.strength = updates.strength;
    if (updates.metadata !== undefined) body.metadata = updates.metadata;
    await this.request<void>(
      "PATCH",
      `/v1/memories/${memoryId}/associations/${associationId}`,
      body,
    );
  }

  /**
   * Delete an association (graph edge) by ID.
   *
   * `memoryId` must be one of the association's endpoints (source or target);
   * otherwise the server returns 404.
   */
  async deleteAssociation(memoryId: string, associationId: string): Promise<void> {
    await this.request<void>(
      "DELETE",
      `/v1/memories/${memoryId}/associations/${associationId}`,
    );
  }

  /**
   * List/browse memories in the workspace ordered by recency (created_at desc).
   *
   * Unlike {@link recall} this performs no vector search — it is a plain
   * filtered enumeration for browsing. Supports limit/offset pagination and
   * optional type/subtype/tag/context filters.
   */
  async listMemories(options: MemoryListOptions = {}): Promise<MemoryListResponse> {
    const params = new URLSearchParams();
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    if (options.type) params.set("type", String(options.type));
    if (options.subtype) params.set("subtype", String(options.subtype));
    if (options.tag) params.set("tag", options.tag);
    if (options.contextId) params.set("context_id", options.contextId);
    const query = params.toString();
    return this.request<MemoryListResponse>(
      "GET",
      `/v1/memories${query ? `?${query}` : ""}`,
    );
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

  async createCheckpoint(
    sessionId: string,
    input: {
      transcriptSegment: string;
      contentHash: string;
      idempotencyKey: string;
      sourceKind?: string;
      sourceSequence?: number;
      sourceBoundary?: number;
    },
  ): Promise<SessionCheckpoint> {
    return this.request<SessionCheckpoint>(
      "POST",
      `/v1/sessions/${sessionId}/checkpoints`,
      {
        transcript_segment: input.transcriptSegment,
        content_hash: input.contentHash,
        idempotency_key: input.idempotencyKey,
        source_kind: input.sourceKind ?? "transcript",
        source_sequence: input.sourceSequence,
        source_boundary: input.sourceBoundary,
      },
    );
  }

  async getCheckpoint(sessionId: string, checkpointId: string): Promise<SessionCheckpoint> {
    return this.request<SessionCheckpoint>(
      "GET",
      `/v1/sessions/${sessionId}/checkpoints/${checkpointId}`,
    );
  }

  async getContextPack(sessionId: string, options: ContextPackOptions = {}): Promise<ContextPack> {
    return this.request<ContextPack>(
      "POST",
      `/v1/sessions/${sessionId}/context-pack`,
      {
        topic: options.topic,
        entity_ids: options.entityIds ?? [],
        entity_names: options.entityNames ?? [],
        budget_tokens: options.budgetTokens ?? 2048,
        section_limits: options.sectionLimits,
        include_directives: options.includeDirectives,
        include_working_memory: options.includeWorkingMemory,
        include_recent_activity: options.includeRecentActivity,
        include_contradictions: options.includeContradictions,
        include_sandbox_summary: options.includeSandboxSummary,
        include_checkpoint_recovery: options.includeCheckpointRecovery,
      },
    );
  }

  async getContextDelta(
    sessionId: string,
    cursor: string,
    budgetTokens = 2048,
  ): Promise<ContextDelta> {
    return this.request<ContextDelta>(
      "POST",
      `/v1/sessions/${sessionId}/context-delta`,
      { cursor, budget_tokens: budgetTokens },
    );
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
  async createWorkspace(name: string, settings?: Record<string, unknown>, tags?: string[]): Promise<Workspace> {
    const response = await this.request<{ workspace: Workspace }>(
      "POST",
      "/v1/workspaces",
      { name, settings: settings ?? {}, tags: tags ?? [] }
    );
    return response.workspace;
  }

  async getWorkspace(workspaceId?: string): Promise<Workspace> {
    const id = workspaceId ?? this.workspaceId;
    if (!id) throw new ValidationError("Workspace ID required");
    const response = await this.request<{ workspace: Workspace }>("GET", `/v1/workspaces/${id}`);
    return response.workspace;
  }

  async listWorkspaces(filter?: { tags?: string[]; match?: "any" | "all" }): Promise<Workspace[]> {
    let path = "/v1/workspaces";
    if (filter?.tags && filter.tags.length > 0) {
      const params = new URLSearchParams();
      for (const tag of filter.tags) params.append("tags", tag);
      params.append("match", filter.match ?? "all");
      path += `?${params.toString()}`;
    }
    const response = await this.request<{ workspaces: Workspace[] }>("GET", path);
    return response.workspaces;
  }

  async updateWorkspace(workspaceId: string, updates: { name?: string; settings?: Record<string, unknown>; tags?: string[] }): Promise<Workspace> {
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

  async deleteContext(contextId: string, workspaceId?: string): Promise<void> {
    const wsId = workspaceId ?? this.workspaceId;
    if (!wsId) throw new ValidationError("Workspace ID required");
    await this.request<void>(
      "DELETE",
      `/v1/workspaces/${wsId}/contexts/${contextId}`,
    );
  }

  // ------------------------------------------------------------------ //
  // Entity Registry operations
  //
  // Gated server-side by MEMORYLAYER_ENTITY_REGISTRY_ENABLED. When the
  // registry is disabled the server returns 501, surfaced here as
  // EnterpriseRequiredError via the `enterpriseFeature` arg.
  // ------------------------------------------------------------------ //

  /** List canonical entities in a workspace (deterministic order by id). */
  async listEntities(options: EntityListOptions = {}): Promise<EntityListResponse> {
    const params = new URLSearchParams();
    const wsId = options.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options.status) params.set("status", options.status);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    const query = params.toString();
    return this.request<EntityListResponse>(
      "GET",
      `/v1/entities${query ? `?${query}` : ""}`,
      undefined,
      "Entity registry",
    );
  }

  /** Get a single canonical entity by id. */
  async getEntity(entityId: string, workspaceId?: string): Promise<Entity> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    const response = await this.request<EntityResponse>(
      "GET",
      `/v1/entities/${entityId}${query ? `?${query}` : ""}`,
      undefined,
      "Entity registry",
    );
    return response.entity;
  }

  /**
   * Resolve a surface name/alias to an existing canonical entity (never
   * creates). Throws NotFoundError if no entity matches.
   */
  async resolveEntity(name: string, options: EntityResolveOptions = {}): Promise<EntityResolveResponse> {
    const params = new URLSearchParams();
    params.set("name", name);
    if (options.entityType) params.set("entity_type", String(options.entityType));
    const wsId = options.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    return this.request<EntityResolveResponse>(
      "GET",
      `/v1/entities/resolve?${params.toString()}`,
      undefined,
      "Entity registry",
    );
  }

  /**
   * Merge `sourceId` into `targetId`; returns the surviving target entity.
   * Requires the entity registry to be enabled (501 -> EnterpriseRequiredError).
   */
  async mergeEntities(options: EntityMergeOptions): Promise<Entity> {
    const params = new URLSearchParams();
    const wsId = options.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    const response = await this.request<EntityResponse>(
      "POST",
      `/v1/entities/merge${query ? `?${query}` : ""}`,
      {
        source_id: options.sourceId,
        target_id: options.targetId,
        reason: options.reason,
      },
      "Entity registry",
    );
    return response.entity;
  }

  // ------------------------------------------------------------------ //
  // API Token operations (admin scope; gRPC-backed via Aether)
  // ------------------------------------------------------------------ //

  /** List API tokens. Set `includeRevoked` to also return revoked tokens. */
  async listTokens(includeRevoked = false): Promise<ApiToken[]> {
    const params = new URLSearchParams();
    if (includeRevoked) params.set("include_revoked", "true");
    const query = params.toString();
    const response = await this.request<TokenListResponse>(
      "GET",
      `/v1/tokens${query ? `?${query}` : ""}`,
    );
    return response.tokens;
  }

  /**
   * Create a new API token. The returned object includes the plaintext
   * `token` value, which is only ever available at creation time.
   */
  async createToken(options: TokenCreateOptions): Promise<ApiTokenWithSecret> {
    const body: Record<string, unknown> = { name: options.name };
    if (options.principalType !== undefined) body.principal_type = options.principalType;
    if (options.workspacePatterns !== undefined) body.workspace_patterns = options.workspacePatterns;
    if (options.scopes !== undefined) body.scopes = options.scopes;
    if (options.expiresInDays !== undefined) body.expires_in_days = options.expiresInDays;
    return this.request<ApiTokenWithSecret>("POST", "/v1/tokens", body);
  }

  /** Get details for a single API token. */
  async getToken(tokenId: string): Promise<ApiToken> {
    return this.request<ApiToken>("GET", `/v1/tokens/${tokenId}`);
  }

  /** Delete an API token. */
  async deleteToken(tokenId: string): Promise<void> {
    await this.request<void>("DELETE", `/v1/tokens/${tokenId}`);
  }

  /**
   * Revoke an API token. Revoked tokens are invalidated immediately but remain
   * visible in listings (with `revoked=true`).
   */
  async revokeToken(tokenId: string): Promise<void> {
    await this.request<void>("POST", `/v1/tokens/${tokenId}/revoke`);
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
    const headers = this.buildHeaders(undefined, false);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, {
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
    const headers = this.buildHeaders(undefined, false);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, {
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
    const headers = this.buildHeaders(undefined, false);
    headers["Content-Type"] = "application/x-ndjson";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, {
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

    const headers = this.buildHeaders(undefined, false);

    const url = `${this.baseUrl}/v1/documents`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, {
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
    const headers = this.buildHeaders(undefined, false);

    const url = `${this.baseUrl}/v1/documents/${documentId}/pages/${pageId}/image`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, { method: "GET", headers, signal: controller.signal });
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

  /**
   * List chat threads owned by a user across all workspaces.
   *
   * Unlike {@link listThreads}, this is keyed on (tenant, user, ownership) and
   * does NOT require a workspace filter — used for user-session-scoped views.
   */
  async listUserThreads(userId: string, options: UserThreadListOptions = {}): Promise<ChatThread[]> {
    const params = new URLSearchParams();
    if (options.ownership) params.set("ownership", options.ownership);
    if (options.scopeFilter) params.set("scope_filter", options.scopeFilter);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    const response = await this.request<{ threads: ChatThread[]; total_count: number }>(
      "GET",
      `/v1/threads/user/${userId}${query ? `?${query}` : ""}`,
    );
    return response.threads;
  }

  /** Update a thread (e.g. rename or change metadata). */
  async updateThread(threadId: string, updates: ThreadUpdateOptions): Promise<ChatThread> {
    const params = new URLSearchParams();
    const wsId = updates.workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    const body: Record<string, unknown> = {};
    if (updates.title !== undefined) body.title = updates.title;
    if (updates.metadata !== undefined) body.metadata = updates.metadata;
    const response = await this.request<{ thread: ChatThread }>(
      "PUT",
      `/v1/threads/${threadId}${query ? `?${query}` : ""}`,
      body,
    );
    return response.thread;
  }

  /** Delete a single message from a thread. */
  async deleteMessage(threadId: string, messageId: string, workspaceId?: string): Promise<void> {
    const params = new URLSearchParams();
    const wsId = workspaceId ?? this.workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    const query = params.toString();
    await this.request<void>(
      "DELETE",
      `/v1/threads/${threadId}/messages/${messageId}${query ? `?${query}` : ""}`,
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

  // ------------------------------------------------------------------ //
  // Dataset operations (Enterprise)
  // ------------------------------------------------------------------ //

  /**
   * Upload a dataset for profiling and memory extraction.
   *
   * Requires MemoryLayer Enterprise. On OSS servers this throws
   * `EnterpriseRequiredError`.
   */
  async uploadDataset(
    file: Blob | File,
    filename: string,
    options: DatasetUploadOptions = {},
  ): Promise<DatasetUploadResponse> {
    const formData = new FormData();
    formData.append("file", file, filename);
    if (options.name) formData.append("name", options.name);
    if (options.targetContextId) formData.append("target_context_id", options.targetContextId);
    if (options.importance !== undefined) formData.append("importance", String(options.importance));
    if (options.sampleRows !== undefined) formData.append("sample_rows", String(options.sampleRows));
    if (options.detectTimeSeries !== undefined) formData.append("detect_time_series", String(options.detectTimeSeries));
    if (options.generateSummaries !== undefined) formData.append("generate_summaries", String(options.generateSummaries));

    const headers = this.buildHeaders(undefined, false);

    const url = `${this.baseUrl}/v1/datasets`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await this.fetchImpl(url, {
        method: "POST",
        headers,
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        await this.handleError(response, "Dataset management");
      }
      return await response.json() as DatasetUploadResponse;
    } catch (error) {
      if (error instanceof MemoryLayerError) throw error;
      throw new MemoryLayerError(`Dataset upload failed: ${error}`);
    }
  }

  /**
   * List datasets in the workspace.
   */
  async listDatasets(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<DatasetListResponse> {
    const params = new URLSearchParams();
    if (options?.status) params.set("status", options.status);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    return this.request<DatasetListResponse>(
      "GET",
      `/v1/datasets${query ? `?${query}` : ""}`,
      undefined,
      "Dataset management",
    );
  }

  /**
   * Get dataset metadata, schema, and profile.
   */
  async getDataset(datasetId: string): Promise<DatasetInfo> {
    return this.request<DatasetInfo>(
      "GET",
      `/v1/datasets/${datasetId}`,
      undefined,
      "Dataset management",
    );
  }

  /**
   * Delete a dataset and optionally its extracted memories.
   */
  async deleteDataset(datasetId: string, deleteMemories = false): Promise<void> {
    await this.request<void>(
      "DELETE",
      `/v1/datasets/${datasetId}?delete_memories=${deleteMemories}`,
      undefined,
      "Dataset management",
    );
  }

  /**
   * Get memories extracted from a dataset.
   */
  async getDatasetMemories(datasetId: string): Promise<DatasetMemoriesResponse> {
    return this.request<DatasetMemoriesResponse>(
      "GET",
      `/v1/datasets/${datasetId}/memories`,
      undefined,
      "Dataset management",
    );
  }

  /**
   * Query a slice of dataset data using DuckDB.
   *
   * Supports both structured filters and raw SQL (SELECT only).
   * The dataset is queried as a table named 'data'.
   */
  async queryDatasetSlice(datasetId: string, options: DatasetSliceOptions = {}): Promise<DatasetSliceResult> {
    const body: Record<string, unknown> = {
      limit: options.limit ?? 100,
      offset: options.offset ?? 0,
      descending: options.descending ?? false,
    };
    if (options.sql !== undefined) body.sql = options.sql;
    if (options.columns !== undefined) body.columns = options.columns;
    if (options.filters !== undefined) body.filters = options.filters;
    if (options.orderBy !== undefined) body.order_by = options.orderBy;

    return this.request<DatasetSliceResult>(
      "POST",
      `/v1/datasets/${datasetId}/slice`,
      body,
      "Dataset management",
    );
  }

  /**
   * Get dataset processing job status.
   */
  async getDatasetJob(jobId: string): Promise<DatasetJobInfo> {
    return this.request<DatasetJobInfo>(
      "GET",
      `/v1/datasets/jobs/${jobId}`,
      undefined,
      "Dataset processing jobs",
    );
  }

  /**
   * List dataset processing jobs in the workspace.
   */
  async listDatasetJobs(options?: { status?: string; limit?: number }): Promise<{ jobs: DatasetJobInfo[] }> {
    const params = new URLSearchParams();
    if (options?.status) params.set("status", options.status);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    const query = params.toString();
    return this.request<{ jobs: DatasetJobInfo[] }>(
      "GET",
      `/v1/datasets/jobs${query ? `?${query}` : ""}`,
      undefined,
      "Dataset processing jobs",
    );
  }

  /**
   * Cancel a running dataset processing job.
   */
  async cancelDatasetJob(jobId: string): Promise<void> {
    await this.request<void>(
      "POST",
      `/v1/datasets/jobs/${jobId}/cancel`,
      undefined,
      "Dataset processing jobs",
    );
  }

  /** Used internally by SkillsNamespace to make requests with OBO authority. */
  async _skillsRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
    responseMode: "json" | "text" | "arraybuffer" = "json",
  ): Promise<T> {
    return this.request<T>(method, path, body, undefined, authority, responseMode);
  }

  /** Used internally by McpServersNamespace to make requests with OBO authority. */
  async _mcpServersRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T> {
    return this.request<T>(method, path, body, undefined, authority);
  }

  /** Used internally by KnowledgebaseNamespace to make requests with OBO authority. */
  async _kbRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
    responseMode: "json" | "text" | "arraybuffer" = "json",
  ): Promise<T> {
    return this.request<T>(method, path, body, undefined, authority, responseMode);
  }

  /** Used internally by RpgNamespace to make requests with OBO authority. */
  async _rpgRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T> {
    return this.request<T>(method, path, body, undefined, authority);
  }
}

/**
 * Lightweight OBO proxy returned by `client.actingFor()`.
 * Delegates all calls to the parent client with a fixed AuthorityContext and
 * optional workspace override. Per-call authority headers are computed fresh
 * on each request — no shared mutable state, so concurrent calls for different
 * subjects on the same underlying client never interfere.
 */
export class OboProxy {
  private _client: MemoryLayerClient;
  private _authority: AuthorityContext;
  private _workspaceId?: string;

  /** Skills namespace scoped to this proxy's authority. */
  readonly skills: SkillsNamespace;

  /** MCP Servers namespace scoped to this proxy's authority. */
  readonly mcpServers: McpServersNamespace;

  constructor(client: MemoryLayerClient, authority: AuthorityContext, workspaceId?: string) {
    this._client = client;
    this._authority = authority;
    this._workspaceId = workspaceId;
    this.skills = new SkillsNamespace(client, authority, workspaceId);
    this.mcpServers = new McpServersNamespace(client, authority, workspaceId);
  }

  /** Further scope this proxy to a single workspace. */
  forWorkspace(workspaceId: string): OboProxy {
    return new OboProxy(this._client, this._authority, workspaceId);
  }

  async remember(content: string, options: RememberOptions = {}): Promise<Memory> {
    return this._client.remember(content, {
      ...options,
      workspaceId: options.workspaceId ?? this._workspaceId,
      authority: options.authority ?? this._authority,
    });
  }

  async recall(query: string, options: RecallOptions = {}): Promise<RecallResult> {
    return this._client.recall(query, {
      ...options,
      workspaceId: options.workspaceId ?? this._workspaceId,
      authority: options.authority ?? this._authority,
    });
  }

  async reflect(query: string, options: ReflectOptions = {}): Promise<ReflectResult> {
    return this._client.reflect(query, {
      ...options,
      workspaceId: options.workspaceId ?? this._workspaceId,
      authority: options.authority ?? this._authority,
    });
  }
}
