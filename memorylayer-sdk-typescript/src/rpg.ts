// SPDX-License-Identifier: Apache-2.0
import type { MemoryLayerClient } from "./client.js";
import type { AuthorityContext } from "./types.js";

export interface RpgNodeInput {
  nodeId: string;
  nodeType: string;
  path: string;
  name: string;
  description?: string;
  language?: string;
  parentId?: string;
  metadata?: Record<string, unknown>;
}

export interface RpgEdgeInput {
  sourceId: string;
  targetId: string;
  relationship: string;
  strength?: number;
  metadata?: Record<string, unknown>;
}

export interface RpgNode {
  node_id: string;
  node_type: string;
  path: string;
  name: string;
  description: string;
  language?: string;
  parent_id?: string;
  depth: number;
  metadata: Record<string, unknown>;
  memory_id?: string;
  context_id?: string;
}

export interface RpgEdge {
  source_id: string;
  target_id: string;
  relationship: string;
  strength: number;
  metadata: Record<string, unknown>;
  association_id?: string;
}

export interface RpgSyncOptions {
  fullSync?: boolean;
  sourceCommit?: string;
  contextId?: string;
  taskId?: string;
}

export interface RpgSyncResult {
  nodes_created: number;
  nodes_updated: number;
  nodes_deleted: number;
  edges_created: number;
  edges_updated: number;
  edges_deleted: number;
  sync_time_ms: number;
  source_commit?: string;
}

export interface RpgSubgraph {
  nodes: RpgNode[];
  edges: RpgEdge[];
  root_id?: string;
  depth: number;
  total_nodes: number;
  total_edges: number;
  truncated?: boolean;
}

export interface RpgSearchResult {
  nodes: RpgNode[];
  total_count: number;
  query: string;
}

export interface RpgStatus {
  workspace_id: string;
  has_rpg: boolean;
  node_count: number;
  edge_count: number;
  last_sync_commit?: string;
  last_sync_at?: string;
}

export interface RpgOverlay {
  context_id: string;
  node_count: number;
}

export interface RpgOverlayList {
  overlays: RpgOverlay[];
  total: number;
}

export interface RpgDeleteResult {
  deleted: number;
  not_found: number;
}

export interface RpgOverlayDeleteResult {
  context_id: string;
  deleted: number;
}

export interface RpgNodeList {
  nodes: RpgNode[];
  total_count: number;
}

export interface RpgConflict {
  file_path: string;
  task_ids: string[];
  severity: string;
}

export interface RpgConflictResult {
  task_id: string;
  conflicts: RpgConflict[];
  total_conflicts: number;
}

export interface RpgSymbolConflict extends RpgConflict {
  symbol_id: string;
  description: string;
}

export interface RpgSymbolConflictResult {
  task_ids: string[];
  conflicts: RpgSymbolConflict[];
  total_conflicts: number;
}

export type RpgMaintenanceOperation =
  | "validate"
  | "cleanup"
  | "statistics"
  | "cleanup_intents"
  | "recompute_counts";

export interface RpgMaintenanceResult {
  operation: RpgMaintenanceOperation;
  workspace_id: string;
  result: Record<string, unknown>;
}

export interface RpgEnrichmentResult {
  workspace_id: string;
  context_id: string;
  async_mode: boolean;
  task_id?: string;
  stats?: Record<string, unknown>;
}

export interface RpgSubgraphOptions {
  path?: string;
  nodeId?: string;
  depth?: number;
  nodeTypes?: string[];
  relationshipTypes?: string[];
  contextId?: string;
  maxNodes?: number;
}

function nodeToApi(node: RpgNodeInput): Record<string, unknown> {
  return {
    node_id: node.nodeId,
    node_type: node.nodeType,
    path: node.path,
    name: node.name,
    description: node.description ?? "",
    language: node.language,
    parent_id: node.parentId,
    metadata: node.metadata ?? {},
  };
}

function edgeToApi(edge: RpgEdgeInput): Record<string, unknown> {
  return {
    source_id: edge.sourceId,
    target_id: edge.targetId,
    relationship: edge.relationship,
    strength: edge.strength ?? 1,
    metadata: edge.metadata ?? {},
  };
}

function queryPath(path: string, params: Record<string, string | number | undefined>): string {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") q.set(key, String(value));
  }
  const encoded = q.toString();
  return `${path}${encoded ? `?${encoded}` : ""}`;
}

/** RPG namespace, available as `client.rpg`. */
export class RpgNamespace {
  constructor(
    private readonly client: MemoryLayerClient,
    private readonly authority?: AuthorityContext,
  ) {}

  withAuthority(authority: AuthorityContext): RpgNamespace {
    return new RpgNamespace(this.client, authority);
  }

  private req<T>(method: string, path: string, body?: unknown): Promise<T> {
    return (this.client as unknown as RpgClientInternal)._rpgRequest<T>(
      method,
      path,
      body,
      this.authority,
    );
  }

  sync(nodes: RpgNodeInput[], edges: RpgEdgeInput[], options: RpgSyncOptions = {}): Promise<RpgSyncResult> {
    return this.req("POST", "/v1/rpg/sync", {
      nodes: nodes.map(nodeToApi),
      edges: edges.map(edgeToApi),
      full_sync: options.fullSync ?? false,
      source_commit: options.sourceCommit,
      context_id: options.contextId ?? "rpg",
      task_id: options.taskId,
    });
  }

  getSubgraph(options: RpgSubgraphOptions): Promise<RpgSubgraph> {
    if (!options.path && !options.nodeId) throw new Error("path or nodeId is required");
    return this.req("GET", queryPath("/v1/rpg/subgraph", {
      path: options.path,
      node_id: options.nodeId,
      depth: options.depth ?? 3,
      node_types: options.nodeTypes?.join(","),
      relationship_types: options.relationshipTypes?.join(","),
      context_id: options.contextId ?? "rpg",
      max_nodes: options.maxNodes ?? 2000,
    }));
  }

  search(query: string, options: { nodeTypes?: string[]; limit?: number; contextId?: string } = {}): Promise<RpgSearchResult> {
    return this.req("GET", queryPath("/v1/rpg/search", {
      query,
      node_types: options.nodeTypes?.join(","),
      limit: options.limit ?? 20,
      context_id: options.contextId ?? "rpg",
    }));
  }

  getStatus(contextId?: string): Promise<RpgStatus> {
    return this.req("GET", queryPath("/v1/rpg/status", { context_id: contextId }));
  }

  listOverlays(): Promise<RpgOverlayList> {
    return this.req("GET", "/v1/rpg/overlays");
  }

  deleteOverlay(contextId: string): Promise<RpgOverlayDeleteResult> {
    return this.req("DELETE", `/v1/rpg/overlays/${encodeURIComponent(contextId)}`);
  }

  deleteNodes(nodeIds: string[], contextId = "rpg"): Promise<RpgDeleteResult> {
    return this.req("DELETE", "/v1/rpg/nodes", { node_ids: nodeIds, context_id: contextId });
  }

  deleteEdges(associationIds: string[]): Promise<RpgDeleteResult> {
    return this.req("DELETE", "/v1/rpg/edges", { association_ids: associationIds });
  }

  listNodes(options: { nodeType?: string; contextId?: string; limit?: number } = {}): Promise<RpgNodeList> {
    return this.req("GET", queryPath("/v1/rpg/nodes", {
      node_type: options.nodeType,
      context_id: options.contextId ?? "rpg",
      limit: options.limit ?? 5000,
    }));
  }

  getMergedSubgraph(
    overlayContext: string,
    options: { baseContext?: string; path?: string; depth?: number } = {},
  ): Promise<RpgSubgraph> {
    return this.req("GET", queryPath("/v1/rpg/merged-subgraph", {
      base_context: options.baseContext ?? "rpg",
      overlay_context: overlayContext,
      path: options.path,
      depth: options.depth ?? 3,
    }));
  }

  getConflicts(taskId: string): Promise<RpgConflictResult> {
    return this.req("GET", queryPath("/v1/rpg/conflicts", { task_id: taskId }));
  }

  getSymbolConflicts(taskIdA: string, taskIdB: string): Promise<RpgSymbolConflictResult> {
    return this.req("GET", queryPath("/v1/rpg/conflicts/symbols", {
      task_id_a: taskIdA,
      task_id_b: taskIdB,
    }));
  }

  maintenance(operation: RpgMaintenanceOperation): Promise<RpgMaintenanceResult> {
    return this.req("POST", `/v1/rpg/maintenance/${operation}`);
  }

  enrich(options: { contextId?: string; phases?: string[]; async?: boolean } = {}): Promise<RpgEnrichmentResult> {
    return this.req("POST", "/v1/rpg/enrich", {
      context_id: options.contextId ?? "rpg",
      phases: options.phases,
      async: options.async ?? true,
    });
  }
}

interface RpgClientInternal {
  _rpgRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T>;
}
