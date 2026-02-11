/**
 * TypeScript type definitions for MemoryLayer MCP Server
 *
 * Re-exports SDK types where compatible, defines MCP-specific types
 */

// Re-export core types from SDK
export type {
  Memory,
  RecallResult,
  Association
} from "@scitrera/memorylayer-sdk";

export {
  MemoryType,
  MemorySubtype,
  RecallMode,
  SearchTolerance,
} from "@scitrera/memorylayer-sdk";

export type { RelationshipType } from "@scitrera/memorylayer-sdk";

// MCP-specific input types (snake_case to match Python MCP server)
export interface RememberInput {
  content: string;
  type?: string;
  subtype?: string;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
  associations?: string[];
  context_id?: string;
  user_id?: string;
}

export interface RecallInput {
  query: string;
  types?: string[];
  subtypes?: string[];
  tags?: string[];
  context_id?: string;
  user_id?: string;
  mode?: string;
  tolerance?: string;
  limit?: number;
  min_relevance?: number;
  include_associations?: boolean;
  traverse_depth?: number;
  max_expansion?: number;
  created_after?: string;
  created_before?: string;
  context?: Array<{ role: string; content: string }>;
  rag_threshold?: number;
  detail_level?: 'abstract' | 'overview' | 'full';
}

export interface ReflectInput {
  query: string;
  detail_level?: 'abstract' | 'overview' | 'full';
  include_sources?: boolean;
  depth?: number;
  types?: string[];
  subtypes?: string[];
  tags?: string[];
  context_id?: string;
  user_id?: string;
}

export interface ReflectResult {
  reflection: string;
  source_memories: string[];  // MCP expects string[] (IDs), SDK returns Memory[]
  confidence: number;
  tokens_processed: number;
}

export interface AssociateInput {
  source_id: string;
  target_id: string;
  relationship: string;
  strength?: number;
  metadata?: Record<string, unknown>;
}

export interface GraphQueryInput {
  start_memory_id: string;
  relationship_types?: string[];
  relationship_categories?: string[];
  max_depth?: number;
  direction?: "outgoing" | "incoming" | "both";
  min_strength?: number;
  max_paths?: number;
  max_nodes?: number;
}

import type { Association } from "@scitrera/memorylayer-sdk";

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

// ============================================================================
// Context Environment input types
// ============================================================================

export interface ContextExecInput {
  code: string;
  result_var?: string;
  return_result?: boolean;
  max_return_chars?: number;
}

export interface ContextInspectInput {
  variable?: string;
  preview_chars?: number;
}

export interface ContextLoadInput {
  var: string;
  query: string;
  limit?: number;
  types?: string[];
  tags?: string[];
  min_relevance?: number;
  include_embeddings?: boolean;
}

export interface ContextInjectInput {
  key: string;
  value: string;
  parse_json?: boolean;
}

export interface ContextQueryInput {
  prompt: string;
  variables: string[];
  max_context_chars?: number;
  result_var?: string;
}

export interface ContextRlmInput {
  goal: string;
  memory_query?: string;
  memory_limit?: number;
  max_iterations?: number;
  variables?: string[];
  result_var?: string;
  detail_level?: "brief" | "standard" | "detailed";
}

/**
 * Tool handler response format (MCP-specific)
 */
export interface ToolResponse {
  success: boolean;
  [key: string]: unknown;
}
