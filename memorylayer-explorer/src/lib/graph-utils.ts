import { type Node, type Edge } from "@xyflow/react";
import type { GraphQueryResult, Memory } from "@/types";
import {
  memoryTypeColors,
  relationshipCategoryColors,
  getRelationshipCategory,
} from "./colors";

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max - 1) + "\u2026";
}

/**
 * Convert a GraphQueryResult plus fetched Memory objects into React Flow
 * nodes and edges.
 */
export function convertToReactFlow(
  result: GraphQueryResult,
  memories: Map<string, Memory>,
): GraphData {
  const nodes: Node[] = result.unique_nodes.map((id) => {
    const memory = memories.get(id);
    const memType = memory?.type ?? "episodic";
    return {
      id,
      type: "memoryNode",
      position: { x: 0, y: 0 },
      data: {
        memory,
        label: memory ? truncate(memory.content, 60) : id,
        memoryType: memType,
        borderColor: memoryTypeColors[memType] ?? memoryTypeColors.episodic,
      },
    };
  });

  const edgeMap = new Map<string, Edge>();

  for (const path of result.paths) {
    for (const assoc of path.edges) {
      const key = `${assoc.source_id}-${assoc.target_id}-${assoc.relationship}`;
      if (edgeMap.has(key)) continue;

      const category = getRelationshipCategory(assoc.relationship);
      const color =
        relationshipCategoryColors[category] ??
        relationshipCategoryColors.context;

      edgeMap.set(key, {
        id: key,
        source: assoc.source_id,
        target: assoc.target_id,
        type: "relationshipEdge",
        data: {
          relationship: assoc.relationship,
          strength: assoc.strength,
          category,
          color,
        },
      });
    }
  }

  return { nodes, edges: Array.from(edgeMap.values()) };
}

/** Merge incoming graph data into an existing graph, deduplicating by id. */
export function mergeGraphData(
  existing: GraphData,
  incoming: GraphData,
): GraphData {
  const nodeIds = new Set(existing.nodes.map((n) => n.id));
  const edgeIds = new Set(existing.edges.map((e) => e.id));

  const mergedNodes = [
    ...existing.nodes,
    ...incoming.nodes.filter((n) => !nodeIds.has(n.id)),
  ];
  const mergedEdges = [
    ...existing.edges,
    ...incoming.edges.filter((e) => !edgeIds.has(e.id)),
  ];

  return { nodes: mergedNodes, edges: mergedEdges };
}
