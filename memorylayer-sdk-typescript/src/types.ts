export enum MemoryType {
  EPISODIC = "episodic",
  SEMANTIC = "semantic",
  PROCEDURAL = "procedural",
  WORKING = "working"
}

export enum MemorySubtype {
  SOLUTION = "solution",
  PROBLEM = "problem",
  CODE_PATTERN = "code_pattern",
  FIX = "fix",
  ERROR = "error",
  WORKFLOW = "workflow",
  PREFERENCE = "preference",
  DECISION = "decision",
  PROFILE = "profile",
  ENTITY = "entity",
  EVENT = "event",
  DIRECTIVE = "directive",
  INFERENCE = "inference"
}

export enum RecallMode {
  RAG = "rag",
  LLM = "llm",
  HYBRID = "hybrid"
}

export enum SearchTolerance {
  LOOSE = "loose",
  MODERATE = "moderate",
  STRICT = "strict"
}

/**
 * Relationship types are strings. The server uses a unified ontology with ~65 types.
 * These constants cover the most common types for convenience.
 */
export type RelationshipType = string;

/** Common relationship type constants */
export const RELATIONSHIP_TYPES = {
  // Causal
  CAUSES: "causes",
  TRIGGERS: "triggers",
  LEADS_TO: "leads_to",
  PREVENTS: "prevents",
  // Solution
  SOLVES: "solves",
  ADDRESSES: "addresses",
  ALTERNATIVE_TO: "alternative_to",
  IMPROVES: "improves",
  // Context
  OCCURS_IN: "occurs_in",
  APPLIES_TO: "applies_to",
  WORKS_WITH: "works_with",
  REQUIRES: "requires",
  // Learning
  BUILDS_ON: "builds_on",
  CONTRADICTS: "contradicts",
  CONFIRMS: "confirms",
  SUPERSEDES: "supersedes",
  // Similarity
  SIMILAR_TO: "similar_to",
  VARIANT_OF: "variant_of",
  RELATED_TO: "related_to",
  // Workflow
  FOLLOWS: "follows",
  DEPENDS_ON: "depends_on",
  ENABLES: "enables",
  BLOCKS: "blocks",
  // Quality
  EFFECTIVE_FOR: "effective_for",
  PREFERRED_OVER: "preferred_over",
  DEPRECATED_BY: "deprecated_by",
  // Hierarchical (from ontology)
  PART_OF: "part_of",
  CONTAINS: "contains",
  INSTANCE_OF: "instance_of",
  SUBTYPE_OF: "subtype_of",
  // Temporal
  PRECEDES: "precedes",
  CONCURRENT_WITH: "concurrent_with",
} as const;

export interface Memory {
  id: string;
  workspace_id: string;
  tenant_id: string;
  context_id: string;
  user_id?: string;
  content: string;
  content_hash: string;
  type: MemoryType;
  subtype?: MemorySubtype;
  importance: number;
  tags: string[];
  metadata: Record<string, unknown>;
  abstract?: string;
  overview?: string;
  session_id?: string;
  category?: string;
  embedding?: number[];
  access_count: number;
  last_accessed_at?: string;
  decay_factor: number;
  source_scope?: string;
  relevance_score?: number;
  boosted_score?: number;
  archived_at?: string;
  deleted_at?: string;
  created_at: string;
  updated_at: string;
  match_signals?: string[];
  relation_write_result?: EntityRelationWriteResult;
}

export interface EntityRelationInput {
  source_entity_id?: string;
  source_entity_name?: string;
  target_entity_id?: string;
  target_entity_name?: string;
  relationship: string;
  confidence?: number;
  source_span_start?: number;
  source_span_end?: number;
}

export interface EntityRelationWriteResult {
  resolved: number;
  unresolved: number;
  rejected: number;
  duplicate: number;
  relation_ids: string[];
  errors: string[];
}

export interface BudgetSummary {
  requested?: number;
  used: number;
  estimator: string;
  truncated_items: number;
  omitted_items: number;
}

export interface GenerationSummary {
  policy: "deterministic" | "adaptive" | "generative";
  calls: number;
  input_tokens: number;
  output_tokens: number;
}

export interface EntityRelationPath {
  seed_entity_id: string;
  entity_ids: string[];
  relations: Array<Record<string, unknown>>;
  evidence_ids: string[];
  evidence_memory_ids: string[];
}

export interface RecallResult {
  memories: Memory[];
  mode_used: RecallMode;
  search_latency_ms: number;
  total_count: number;
  query_tokens: number;
  query_rewritten?: string;
  sufficiency_reached?: boolean;
  source_scope?: string;
  boosted_score?: number;
  token_summary?: {
    returned: number;
    full_would_be: number;
    savings_percent: number;
  };
  retrieval_confidence: "strong" | "moderate" | "weak";
  confidence_reasons: string[];
  budget_summary?: BudgetSummary;
  generation_summary?: GenerationSummary;
  relation_paths: EntityRelationPath[];
}

export interface ReflectResult {
  reflection: string;
  source_memories: string[];
  confidence?: number;
  tokens_processed: number;
}

export interface Association {
  id: string;
  workspace_id: string;
  source_id: string;
  target_id: string;
  relationship: string;
  strength: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface Session {
  id: string;
  workspace_id: string;
  tenant_id: string;
  context_id: string;
  user_id?: string;
  working_memory: Record<string, unknown>;
  metadata: Record<string, unknown>;
  expires_at: string;
  created_at: string;
}

export interface SessionBriefing {
  workspace_summary: {
    total_memories: number;
    recent_memories: number;
    active_topics: string[];
    total_categories: number;
    total_associations: number;
    memory_types: Record<string, number>;
  };
  recent_activity: Array<{
    timestamp: string;
    summary: string;
    memories_created: number;
    key_decisions: string[];
  }>;
  open_threads: Array<Record<string, unknown>>;
  contradictions_detected: Array<Record<string, unknown>>;
  memories: Array<Record<string, unknown>>;
}

export interface SessionCheckpoint {
  id: string;
  workspace_id: string;
  session_id: string;
  raw_memory_id: string;
  source_kind: string;
  source_sequence?: number;
  source_boundary?: number;
  content_hash: string;
  byte_count: number;
  capture_status: string;
  index_status: string;
  enrichment_status: string;
  idempotency_key: string;
  created_at: string;
  updated_at: string;
}

export interface ContextPackItem {
  id: string;
  kind: string;
  content: string;
  importance: number;
  event_time?: string;
  match_signals: string[];
  source_references: string[];
  tombstone: boolean;
}

export interface ContextPack {
  rendered: string;
  items: ContextPackItem[];
  open_threads: Array<Record<string, unknown>>;
  unresolved_contradictions: Array<Record<string, unknown>>;
  budget_summary: BudgetSummary;
  generation_summary: GenerationSummary;
  cursor: string;
  degradation_notices: string[];
}

export interface ContextDelta {
  rendered: string;
  items: ContextPackItem[];
  budget_summary: BudgetSummary;
  generation_summary: GenerationSummary;
  cursor: string;
  has_more: boolean;
  degradation_notices: string[];
}

export interface ContextPackOptions {
  topic?: string;
  entityIds?: string[];
  entityNames?: string[];
  budgetTokens?: number;
  sectionLimits?: Record<string, number>;
  includeDirectives?: boolean;
  includeWorkingMemory?: boolean;
  includeRecentActivity?: boolean;
  includeContradictions?: boolean;
  includeSandboxSummary?: boolean;
  includeCheckpointRecovery?: boolean;
}

export interface Workspace {
  id: string;
  tenant_id: string;
  name: string;
  settings: Record<string, unknown>;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface Context {
  id: string;
  workspace_id: string;
  tenant_id: string;
  name: string;
  description?: string;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

// Detail levels for recall
export enum DetailLevel {
  ABSTRACT = "abstract",
  OVERVIEW = "overview",
  FULL = "full"
}

// Request types
export interface RememberOptions {
  /** Override workspace for this operation (fallback if session not set) */
  workspaceId?: string;
  type?: MemoryType | string;
  subtype?: MemorySubtype | string;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
  associations?: string[];
  contextId?: string;
  userId?: string;
  authority?: AuthorityContext;
  relations?: EntityRelationInput[];
}

export interface RecallOptions {
  /** Override workspace for this query (fallback if session not set) */
  workspaceId?: string;
  types?: (MemoryType | string)[];
  subtypes?: (MemorySubtype | string)[];
  tags?: string[];
  contextId?: string;
  mode?: RecallMode | string;
  tolerance?: SearchTolerance | string;
  limit?: number;
  minRelevance?: number;
  recencyWeight?: number;
  includeAssociations?: boolean;
  traverseDepth?: number;
  maxExpansion?: number;
  createdAfter?: Date;
  createdBefore?: Date;
  /** Skip this many results (pagination). Maps to `offset`. */
  offset?: number;
  /** Keep memories whose effective event time is >= this. Maps to `event_after`. */
  eventAfter?: Date;
  /** Keep memories whose effective event time is <= this. Maps to `event_before`. */
  eventBefore?: Date;
  /** Order results by effective event time: 'asc' or 'desc' (omit = by relevance). */
  timeOrder?: 'asc' | 'desc';
  /** Include the _global workspace in search (default true server-side). */
  includeGlobal?: boolean;
  /** Include the user-scoped global workspace (_global_user), filtered by user_id. */
  includeGlobalUser?: boolean;
  conversationContext?: Array<{ role: string; content: string }>;
  ragThreshold?: number;
  detailLevel?: DetailLevel | 'abstract' | 'overview' | 'full';
  userId?: string;
  authority?: AuthorityContext;
  budgetTokens?: number;
  includeConfidence?: boolean;
  includeRelations?: boolean;
}

export interface ReflectOptions {
  /** Override workspace for this query (fallback if session not set) */
  workspaceId?: string;
  detailLevel?: DetailLevel | 'abstract' | 'overview' | 'full';
  includeSources?: boolean;
  depth?: number;
  types?: (MemoryType | string)[];
  subtypes?: (MemorySubtype | string)[];
  tags?: string[];
  contextId?: string;
  userId?: string;
  authority?: AuthorityContext;
}

// OBO / authority types
export interface PrincipalRef {
  type: string;
  id: string;
}

export interface AuthorityContext {
  grantId: string;
  subject: PrincipalRef;
}

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  workspaceId?: string;
  sessionId?: string;
  timeout?: number;
  defaultAuthority?: AuthorityContext;
  /**
   * Maximum number of retries for transient failures (5xx, 429). Defaults to 3.
   * Set to 0 to disable retries.
   */
  maxRetries?: number;
  /**
   * Base delay in milliseconds for exponential backoff between retries.
   * Defaults to 500. Actual delay is `retryBaseDelay * 2^attempt`, or the
   * `Retry-After` header value when the server provides one.
   */
  retryBaseDelay?: number;
  /**
   * Custom fetch implementation. Defaults to globalThis.fetch.
   *
   * Use this to route requests through an alternate transport — for example,
   * `AetherFetchTransport` from `@scitrera/aether-client` so requests tunnel
   * over an Aether sidecar instead of a direct HTTP call. The implementation
   * must match the WHATWG fetch signature.
   */
  fetch?: typeof fetch;
}

// Session options
export interface SessionCreateOptions {
  sessionId?: string;
  workspaceId?: string;
  ttlSeconds?: number;
  metadata?: Record<string, unknown>;
  contextId?: string;
  workingMemory?: Record<string, unknown>;
  briefing?: boolean;
  briefingOptions?: {
    lookbackHours?: number;
    detailLevel?: string;
    limit?: number;
  };
}

export interface SessionStartResponse {
  session: Session;
  briefing?: SessionBriefing;
}

export interface CommitOptions {
  minImportance?: number;
  deduplicate?: boolean;
  categories?: string[];
  maxMemories?: number;
}

export interface CommitResponse {
  session_id: string;
  memories_extracted: number;
  memories_deduplicated: number;
  memories_created: number;
  breakdown: Record<string, number>;
  extraction_time_ms: number;
}

// Graph traversal types
export enum RelationshipCategory {
  CAUSAL = "causal",
  SOLUTION = "solution",
  CONTEXT = "context",
  LEARNING = "learning",
  SIMILARITY = "similarity",
  WORKFLOW = "workflow",
  QUALITY = "quality"
}

export interface GraphTraverseOptions {
  relationshipTypes?: string[];
  relationshipCategories?: RelationshipCategory[];
  maxDepth?: number;
  direction?: "outgoing" | "incoming" | "both";
  minStrength?: number;
  maxPaths?: number;
  maxNodes?: number;
}

export interface GraphPath {
  nodes: string[];
  edges: Association[];
  total_strength: number;
  depth: number;
}

export interface GraphQueryResult {
  paths: GraphPath[];
  total_paths: number;
  unique_nodes: string[];
  query_latency_ms: number;
}

// Batch operations
export type BatchOperation =
  | { op: "create"; memory: RememberOptions & { content: string } }
  | { op: "update"; memory_id: string; updates: Partial<RememberOptions> & { content?: string } }
  | { op: "delete"; memory_id: string; hard?: boolean };

export interface BatchResult {
  results: Array<{
    index: number;
    success: boolean;
    memory?: Memory;
    error?: string;
  }>;
  total_processed: number;
  successful: number;
  failed: number;
}

// Association creation
export interface AssociationCreateOptions {
  sourceId: string;
  targetId: string;
  relationship: string;
  strength?: number;
  metadata?: Record<string, unknown>;
}

// Workspace schema
export interface WorkspaceSchema {
  relationship_types: string[];
  memory_subtypes: string[];
  can_customize: boolean;
}

// Workspace export/import
export interface WorkspaceExportData {
  version: string;
  exported_at: string;
  workspace_id: string;
  total_memories: number;
  total_associations: number;
  memories: Array<Record<string, unknown>>;
  associations: Array<Record<string, unknown>>;
}

export interface WorkspaceImportResult {
  imported: number;
  skipped_duplicates: number;
  errors: number;
  details: string[];
}

// Context Environment types
export interface ContextExecOptions {
  resultVar?: string;
  returnResult?: boolean;
  maxReturnChars?: number;
}

export interface ContextExecResult {
  output: string;
  result?: unknown;
  error?: string;
  variables_changed: string[];
}

export interface ContextInspectOptions {
  variable?: string;
  previewChars?: number;
}

export interface ContextInspectResult {
  variables: Record<string, { type: string; preview: string }>;
  variable_count: number;
}

export interface ContextLoadOptions {
  limit?: number;
  types?: string[];
  tags?: string[];
  minRelevance?: number;
  includeEmbeddings?: boolean;
}

export interface ContextLoadResult {
  var: string;
  count: number;
  query: string;
}

export interface ContextInjectOptions {
  parseJson?: boolean;
}

export interface ContextInjectResult {
  key: string;
  type: string;
}

// ------------------------------------------------------------------ //
// Document types (Enterprise)
// ------------------------------------------------------------------ //

export interface DocumentPage {
  id: string;
  document_id: string;
  workspace_id: string;
  page_no: number;
  image_storage_path?: string;
  transcript?: string;
  transcript_model?: string;
  metadata: Record<string, unknown>;
  created_at?: string;
  relevance_score?: number;
}

export interface DocumentInfo {
  id: string;
  workspace_id: string;
  filename: string;
  document_type: string;
  content_hash: string;
  size_bytes: number;
  mime_type?: string;
  status: string;
  target_context_id: string;
  page_count: number;
  chunk_count: number;
  memory_ids: string[];
  storage_path?: string;
  retain_original: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
  processing_started_at?: string;
  processing_completed_at?: string;
}

export interface JobInfo {
  id: string;
  workspace_id: string;
  document_ids: string[];
  status: string;
  progress_percent: number;
  documents_processed: number;
  total_memories_created: number;
  errors: Array<Record<string, unknown>>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface DocumentUploadOptions {
  targetContextId?: string;
  chunkingStrategy?: string;
  chunkSize?: number;
  chunkOverlap?: number;
  importance?: number;
  retainOriginal?: boolean;
}

export interface DocumentUploadResponse {
  document: DocumentInfo;
  job: JobInfo;
}

export interface PageSearchOptions {
  limit?: number;
  docIds?: string[];
}

export interface PageSearchResponse {
  pages: DocumentPage[];
  total_count: number;
  query: string;
}

export interface PageListResponse {
  document_id: string;
  pages: DocumentPage[];
  total_count: number;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total_count: number;
}

export interface ContextQueryOptions {
  maxContextChars?: number;
  resultVar?: string;
}

export interface ContextQueryResult {
  response: string;
  tokens_used: number;
  variables_included: string[];
}

export interface ContextRlmOptions {
  memoryQuery?: string;
  memoryLimit?: number;
  maxIterations?: number;
  variables?: string[];
  resultVar?: string;
  detailLevel?: "brief" | "standard" | "detailed";
}

export interface ContextRlmResult {
  result: string;
  iterations: number;
  trace: Array<Record<string, unknown>>;
  memories_loaded?: number;
}

export interface ContextStatusResult {
  active: boolean;
  variable_count: number;
  variables: Record<string, string>;
  execution_count: number;
  memory_bytes?: number;
}

// ------------------------------------------------------------------ //
// Chat History types
// ------------------------------------------------------------------ //

export interface ChatMessageContent {
  type: string;
  text?: string;
  data?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  thread_id: string;
  message_index: number;
  role: string;
  content: string | ChatMessageContent[];
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ChatThread {
  id: string;
  workspace_id: string;
  tenant_id: string;
  user_id?: string;
  context_id: string;
  observer_id?: string;
  subject_id?: string;
  title?: string;
  metadata: Record<string, unknown>;
  message_count: number;
  last_decomposed_at?: string;
  last_decomposed_index: number;
  expires_at?: string;
  created_at: string;
  updated_at: string;
}

export interface ThreadCreateOptions {
  threadId?: string;
  workspaceId?: string;
  userId?: string;
  contextId?: string;
  observerId?: string;
  subjectId?: string;
  title?: string;
  metadata?: Record<string, unknown>;
  expiresAt?: string;
}

export interface ThreadListOptions {
  workspaceId?: string;
  userId?: string;
  limit?: number;
  offset?: number;
}

export interface MessageAppendInput {
  role: string;
  content: string | ChatMessageContent[];
  metadata?: Record<string, unknown>;
}

export interface ThreadWithMessagesResponse {
  thread: ChatThread;
  messages: ChatMessage[];
  total_messages: number;
}

export interface ThreadListResponse {
  threads: ChatThread[];
  total_count: number;
}

export interface MessageListResponse {
  messages: ChatMessage[];
  thread_id: string;
  total_count: number;
}

export interface MessagesAppendResponse {
  messages: ChatMessage[];
  thread_id: string;
  new_message_count: number;
}

export interface DecomposeResponse {
  thread_id: string;
  workspace_id: string;
  messages_processed: number;
  memories_created: number;
  from_index: number;
  to_index: number;
}

// ------------------------------------------------------------------ //
// Dataset types (Enterprise)
// ------------------------------------------------------------------ //

export interface DatasetColumn {
  name: string;
  dtype: string;
  column_type: string;
  nullable: boolean;
  null_count: number;
  null_percent: number;
  unique_count: number;
  // Numeric stats
  min_value?: number;
  max_value?: number;
  mean_value?: number;
  median_value?: number;
  std_value?: number;
  p25_value?: number;
  p75_value?: number;
  // String stats
  min_length?: number;
  max_length?: number;
  avg_length?: number;
  // Categorical stats
  top_values?: Array<Record<string, unknown>>;
  // Time series detection
  is_temporal: boolean;
  temporal_resolution?: string;
  temporal_range_start?: string;
  temporal_range_end?: string;
  // Distribution
  histogram?: Record<string, unknown>;
}

export interface DatasetInfo {
  id: string;
  workspace_id: string;
  name: string;
  filename: string;
  format: string;
  content_hash: string;
  size_bytes: number;
  status: string;
  target_context_id: string;
  row_count: number;
  column_count: number;
  columns: DatasetColumn[];
  memory_ids: string[];
  profile_summary?: string;
  metadata: Record<string, unknown>;
  created_at: string;
  profiling_started_at?: string;
  profiling_completed_at?: string;
}

export interface DatasetJobInfo {
  id: string;
  workspace_id: string;
  dataset_ids: string[];
  status: string;
  progress_percent: number;
  datasets_processed: number;
  total_memories_created: number;
  errors: Array<Record<string, unknown>>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface DatasetUploadOptions {
  name?: string;
  targetContextId?: string;
  importance?: number;
  sampleRows?: number;
  detectTimeSeries?: boolean;
  generateSummaries?: boolean;
}

export interface DatasetUploadResponse {
  dataset: DatasetInfo;
  job: DatasetJobInfo;
}

export interface DatasetListResponse {
  datasets: DatasetInfo[];
  total_count: number;
}

export interface DatasetSliceOptions {
  sql?: string;
  columns?: string[];
  filters?: Array<Record<string, unknown>>;
  orderBy?: string;
  descending?: boolean;
  limit?: number;
  offset?: number;
}

export interface DatasetSliceResult {
  dataset_id: string;
  columns: string[];
  dtypes: string[];
  rows: unknown[][];
  total_matching: number;
  returned_count: number;
  sql_executed?: string;
}

export interface DatasetMemoriesResponse {
  dataset_id: string;
  memories: Array<{
    id: string;
    content: string;
    type: string;
    importance: number;
    tags: string[];
    created_at: string;
  }>;
  total_count: number;
}

// ------------------------------------------------------------------ //
// Token types — GET/POST /v1/tokens, GET/DELETE /v1/tokens/{id},
// POST /v1/tokens/{id}/revoke
// ------------------------------------------------------------------ //

export interface ApiToken {
  id: string;
  name: string;
  principal_type: string;
  workspace_patterns: string[];
  scopes: string[];
  created_at: string;
  expires_at?: string | null;
  revoked: boolean;
}

/**
 * Response from creating a token — extends {@link ApiToken} with the
 * plaintext `token` value, which is only ever returned at creation time.
 */
export interface ApiTokenWithSecret extends ApiToken {
  token: string;
}

export interface TokenCreateOptions {
  name: string;
  /** Principal type for the token. Defaults to "User" server-side. */
  principalType?: string;
  /** Workspace glob patterns the token may access. Defaults to ["*"]. */
  workspacePatterns?: string[];
  /** Permission scopes. Defaults to ["*"]. */
  scopes?: string[];
  /** Token lifetime in days. Omit for a non-expiring token. */
  expiresInDays?: number;
}

export interface TokenListResponse {
  tokens: ApiToken[];
}

// ------------------------------------------------------------------ //
// Memory list (browse) — GET /v1/memories
// ------------------------------------------------------------------ //

export interface MemoryListOptions {
  limit?: number;
  offset?: number;
  type?: MemoryType | string;
  subtype?: MemorySubtype | string;
  /** Filter by a single tag. */
  tag?: string;
  contextId?: string;
}

export interface MemoryListResponse {
  memories: Memory[];
  total_count: number;
}

// ------------------------------------------------------------------ //
// Entity Registry types — GET /v1/entities, GET /v1/entities/{id},
// GET /v1/entities/resolve, POST /v1/entities/merge
// (gated by MEMORYLAYER_ENTITY_REGISTRY_ENABLED; may return 501)
// ------------------------------------------------------------------ //

export type EntityTypeValue = "person" | "org" | "project" | "place" | "concept" | "event";

export interface Entity {
  id: string;
  workspace_id: string;
  entity_type: EntityTypeValue | string;
  canonical_name: string;
  normalized_name: string;
  aliases: string[];
  confidence: number;
  provenance: Record<string, unknown>;
  representative_memory_id?: string | null;
  status: string;
  merged_into?: string | null;
  created_at: string;
  updated_at: string;
}

export interface EntityResolution {
  entity: Entity;
  matched_via: "exact" | "alias" | "created" | "embedding";
  score: number;
}

export interface EntityListResponse {
  entities: Entity[];
  total_count: number;
}

export interface EntityResponse {
  entity: Entity;
}

export interface EntityResolveResponse {
  resolution: EntityResolution;
}

export interface EntityListOptions {
  workspaceId?: string;
  /** Entity status filter: 'active' (default) or 'merged'. */
  status?: "active" | "merged";
  limit?: number;
}

export interface EntityResolveOptions {
  workspaceId?: string;
  /** Entity type to resolve against. Defaults to 'person' server-side. */
  entityType?: EntityTypeValue | string;
}

export interface EntityMergeOptions {
  sourceId: string;
  targetId: string;
  /** Audit reason recorded in the merged entity's provenance. */
  reason: string;
  workspaceId?: string;
}

// ------------------------------------------------------------------ //
// Association update — PATCH /v1/memories/{memory_id}/associations/{id}
// ------------------------------------------------------------------ //

export interface AssociationUpdateOptions {
  /** New relationship strength (omit = unchanged). */
  strength?: number;
  /** New metadata dict, replaces existing (omit = unchanged). */
  metadata?: Record<string, unknown>;
}

// ------------------------------------------------------------------ //
// Thread update — PUT /v1/threads/{id}; user threads — GET /v1/threads/user/{id}
// ------------------------------------------------------------------ //

export interface ThreadUpdateOptions {
  title?: string;
  metadata?: Record<string, unknown>;
  workspaceId?: string;
}

export interface UserThreadListOptions {
  /** Ownership filter: 'user' (default) or 'workspace'. */
  ownership?: "user" | "workspace";
  /** Scope filter: 'web' | 'office' | undefined (all). */
  scopeFilter?: "web" | "office";
  limit?: number;
  offset?: number;
}

// ------------------------------------------------------------------ //
// Knowledgebase types — /v1/knowledgebase*
// ------------------------------------------------------------------ //

export interface GraphStats {
  node_count: number;
  edge_count: number;
  community_count: number;
  density: number;
  avg_degree: number;
  max_degree: number;
  god_node_count: number;
}

export interface Knowledgebase {
  workspace_id: string;
  article_count: number;
  community_count: number;
  generated_at: string;
  stats?: GraphStats | null;
}

export interface KBArticle {
  id: string;
  article_type: string;
  title: string;
  content_md: string;
  metadata: Record<string, unknown>;
  generated_at: string;
}

export interface KBArticleListResponse {
  articles: KBArticle[];
  total: number;
}

export interface KBGenerateOptions {
  workspaceId?: string;
  contextId?: string;
  includeRpg?: boolean;
  maxCommunities?: number;
  maxGodNodes?: number;
  regenerate?: boolean;
}

export interface KBArticleListOptions {
  /** Filter by article type: 'index', 'community', 'entity'. */
  articleType?: string;
  limit?: number;
  offset?: number;
}

export interface GraphSnapshot {
  workspace_id: string;
  context_id?: string | null;
  node_count: number;
  edge_count: number;
  includes_rpg: boolean;
}

export interface GraphCommunity {
  id: number;
  memory_ids: string[];
  size: number;
  cohesion_score: number;
  central_node_ids: string[];
  label?: string | null;
}

export interface GraphCentralNode {
  memory_id: string;
  degree: number;
  betweenness: number;
  community_id: number;
}

export interface GraphBridge {
  source_community_id: number;
  target_community_id: number;
  memory_id_source: string;
  memory_id_target: string;
  relationship_type: string;
  strength: number;
}

export interface GraphAnalysis {
  snapshot: GraphSnapshot;
  communities: GraphCommunity[];
  central_nodes: GraphCentralNode[];
  bridges: GraphBridge[];
  stats: GraphStats;
}

export interface GraphAnalysisResponse {
  analysis?: GraphAnalysis | null;
  cached: boolean;
}

// ------------------------------------------------------------------ //
// MCP server import/export — POST /v1/mcp-servers/import,
// GET /v1/mcp-servers/export
// ------------------------------------------------------------------ //

/** Standard .mcp.json document shape: { mcpServers: { name: config, ... } }. */
export interface McpJsonDocument {
  mcpServers: Record<string, Record<string, unknown>>;
}

export interface McpServerImportOptions {
  workspaceId?: string;
  userId?: string;
  /** Provenance: 'server' (default) | 'filesystem' | 'mirrored'. */
  sourceMode?: string;
}

export interface McpServerImportResult {
  imported: number;
  updated: number;
  skipped: number;
  errors: string[];
}
