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
  DIRECTIVE = "directive"
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

export interface Workspace {
  id: string;
  tenant_id: string;
  name: string;
  settings: Record<string, unknown>;
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
  conversationContext?: Array<{ role: string; content: string }>;
  ragThreshold?: number;
  detailLevel?: DetailLevel | 'abstract' | 'overview' | 'full';
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
}

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  workspaceId?: string;
  sessionId?: string;
  timeout?: number;
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
  | { action: "create"; memory: RememberOptions & { content: string } }
  | { action: "update"; memory_id: string; updates: Partial<RememberOptions> & { content?: string } }
  | { action: "delete"; memory_id: string; hard?: boolean };

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
