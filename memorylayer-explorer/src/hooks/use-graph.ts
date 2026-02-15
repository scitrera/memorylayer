"use client";

import { useQuery } from "@tanstack/react-query";
import { useCallback } from "react";
import { useConnection } from "@/providers/connection-provider";
import type { GraphTraverseOptions, Memory } from "@/types";
import { convertToReactFlow, mergeGraphData, type GraphData } from "@/lib/graph-utils";

/**
 * Fetch a graph traversal from a starting memory and convert to React Flow format.
 */
export function useGraphTraversal(
  memoryId: string | undefined,
  options: GraphTraverseOptions = {},
) {
  const { client, isConnected } = useConnection();

  return useQuery<GraphData>({
    queryKey: ["graph", memoryId, JSON.stringify(options)],
    queryFn: async () => {
      const result = await client.traverseGraph(memoryId!, options);

      // Fetch Memory objects for all unique nodes
      const memories = new Map<string, Memory>();
      const fetches = result.unique_nodes.map(async (id) => {
        try {
          const mem = await client.getMemory(id);
          memories.set(id, mem);
        } catch {
          // Node might have been deleted; skip it
        }
      });
      await Promise.all(fetches);

      return convertToReactFlow(result, memories);
    },
    enabled: isConnected && !!memoryId,
  });
}

/**
 * Returns a function that traverses from a node and merges into the existing graph.
 */
export function useExpandNode() {
  const { client } = useConnection();

  const expand = useCallback(
    async (
      nodeId: string,
      existingGraph: GraphData,
      options: GraphTraverseOptions = {},
    ): Promise<GraphData> => {
      const result = await client.traverseGraph(nodeId, {
        maxDepth: 1,
        ...options,
      });

      const memories = new Map<string, Memory>();
      const fetches = result.unique_nodes.map(async (id) => {
        try {
          const mem = await client.getMemory(id);
          memories.set(id, mem);
        } catch {
          // skip deleted nodes
        }
      });
      await Promise.all(fetches);

      const incoming = convertToReactFlow(result, memories);
      return mergeGraphData(existingGraph, incoming);
    },
    [client],
  );

  return expand;
}
