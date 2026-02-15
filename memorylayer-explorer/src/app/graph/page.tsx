"use client";

import { useState, useCallback, useMemo, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from "@xyflow/react";
import { Search, Loader2, Network } from "lucide-react";

import { useConnection } from "@/providers/connection-provider";
import { useGraphTraversal, useExpandNode } from "@/hooks/use-graph";
import { MemoryGraph } from "@/components/graph/memory-graph";
import {
  GraphControls,
  type LayoutType,
  type DirectionType,
} from "@/components/graph/graph-controls";
import { GraphLegend } from "@/components/graph/graph-legend";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cn";
import { applyDagreLayout } from "@/components/graph/layouts/dagre-layout";
import { applyForceLayout } from "@/components/graph/layouts/force-layout";
import { applyRadialLayout } from "@/components/graph/layouts/radial-layout";
import { getRelationshipCategory } from "@/lib/colors";
import { MEMORY_TYPE_COLORS, MEMORY_TYPE_LABELS } from "@/lib/constants";
import { RelationshipCategory } from "@/types";
import type { Memory, GraphTraverseOptions } from "@/types";
import type { GraphData } from "@/lib/graph-utils";

const ALL_CATEGORIES = new Set(Object.values(RelationshipCategory) as string[]);

function GraphPageInner() {
  const searchParams = useSearchParams();
  const fromParam = searchParams.get("from");

  const { client, isConnected } = useConnection();

  // Memory search state (when no ?from param)
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Memory[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [startMemoryId, setStartMemoryId] = useState<string | undefined>(
    fromParam ?? undefined,
  );

  // Graph state
  const [layout, setLayout] = useState<LayoutType>("dagre");
  const [depth, setDepth] = useState(2);
  const [direction, setDirection] = useState<DirectionType>("both");
  const [minStrength, setMinStrength] = useState(0);
  const [enabledCategories, setEnabledCategories] = useState<Set<string>>(
    new Set(ALL_CATEGORIES),
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([] as Edge[]);

  // Traverse options
  const traverseOptions: GraphTraverseOptions = useMemo(
    () => ({
      maxDepth: depth,
      direction,
      minStrength,
    }),
    [depth, direction, minStrength],
  );

  // Graph query
  const {
    data: graphData,
    isLoading: isGraphLoading,
    error: graphError,
  } = useGraphTraversal(startMemoryId, traverseOptions);

  const expandNode = useExpandNode();

  // Apply layout whenever graph data or layout type changes
  useEffect(() => {
    if (!graphData || graphData.nodes.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    // Filter edges by category
    const filteredEdges = graphData.edges.filter((e) => {
      const cat = getRelationshipCategory(
        (e.data?.relationship as string) ?? "",
      );
      return enabledCategories.has(cat);
    });

    // Find nodes that have at least one visible edge, plus the start node
    const connectedNodeIds = new Set<string>();
    if (startMemoryId) connectedNodeIds.add(startMemoryId);
    for (const edge of filteredEdges) {
      connectedNodeIds.add(edge.source as string);
      connectedNodeIds.add(edge.target as string);
    }

    const filteredNodes =
      filteredEdges.length === 0
        ? graphData.nodes
        : graphData.nodes.filter((n) => connectedNodeIds.has(n.id));

    let laid: { nodes: Node[]; edges: typeof filteredEdges };
    switch (layout) {
      case "force":
        laid = applyForceLayout(filteredNodes, filteredEdges);
        break;
      case "radial":
        laid = applyRadialLayout(filteredNodes, filteredEdges, startMemoryId);
        break;
      default:
        laid = applyDagreLayout(filteredNodes, filteredEdges);
    }

    setNodes(laid.nodes);
    setEdges(laid.edges);
  }, [graphData, layout, enabledCategories, startMemoryId, setNodes, setEdges]);

  // Search handler
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim() || !isConnected) return;
    setIsSearching(true);
    try {
      const result = await client.recall(searchQuery, { limit: 10 });
      setSearchResults(result.memories);
    } catch {
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, client, isConnected]);

  const handleSearchKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleSearch();
    },
    [handleSearch],
  );

  // Node click -> open detail panel
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [],
  );

  // Node double-click -> expand
  const handleNodeDoubleClick = useCallback(
    async (_event: React.MouseEvent, node: Node) => {
      if (!graphData) return;
      try {
        const merged = await expandNode(node.id, graphData, {
          maxDepth: 1,
          direction,
          minStrength,
        });

        let laid: { nodes: Node[]; edges: typeof merged.edges };
        switch (layout) {
          case "force":
            laid = applyForceLayout(merged.nodes, merged.edges);
            break;
          case "radial":
            laid = applyRadialLayout(merged.nodes, merged.edges, startMemoryId);
            break;
          default:
            laid = applyDagreLayout(merged.nodes, merged.edges);
        }

        setNodes(laid.nodes);
        setEdges(laid.edges);
      } catch {
        // expansion failed, ignore
      }
    },
    [graphData, expandNode, direction, minStrength, layout, startMemoryId, setNodes, setEdges],
  );

  // Category toggle
  const handleCategoryToggle = useCallback((category: string) => {
    setEnabledCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  // Reset filters
  const handleReset = useCallback(() => {
    setDepth(2);
    setDirection("both");
    setMinStrength(0);
    setEnabledCategories(new Set(ALL_CATEGORIES));
    setLayout("dagre");
  }, []);

  // Selected node data for the side panel
  const selectedMemory = useMemo(() => {
    if (!selectedNodeId) return null;
    const node = nodes.find((n) => n.id === selectedNodeId);
    return (node?.data?.memory as Memory) ?? null;
  }, [selectedNodeId, nodes]);

  // If no start memory, show search UI
  if (!startMemoryId) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-4rem)] gap-6">
        <div className="flex flex-col items-center gap-2">
          <Network className="w-12 h-12 text-slate-300" />
          <h1 className="text-2xl font-semibold tracking-tight">
            Graph Explorer
          </h1>
          <p className="text-muted-foreground text-center max-w-md">
            Search for a memory to use as the starting point for graph
            traversal.
          </p>
        </div>

        <div className="w-full max-w-md flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder="Search memories..."
              className="w-full h-10 pl-9 pr-4 rounded-lg border border-slate-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
            {isSearching ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              "Search"
            )}
          </Button>
        </div>

        {searchResults.length > 0 && (
          <div className="w-full max-w-md rounded-lg border border-slate-200 bg-white shadow-sm divide-y divide-slate-100 max-h-80 overflow-y-auto">
            {searchResults.map((mem) => {
              const typeStyle = MEMORY_TYPE_COLORS[mem.type];
              return (
                <button
                  key={mem.id}
                  onClick={() => setStartMemoryId(mem.id)}
                  className="w-full text-left px-4 py-3 hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Badge
                      variant="outline"
                      className={cn(typeStyle?.bg, typeStyle?.text, "text-xs")}
                    >
                      {MEMORY_TYPE_LABELS[mem.type] ?? mem.type}
                    </Badge>
                    <span className="text-xs text-slate-400 font-mono truncate">
                      {mem.id}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700 line-clamp-2">
                    {mem.content}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // Graph view
  return (
    <div className="relative h-[calc(100vh-4rem)]">
      {isGraphLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/60 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
            <p className="text-sm text-slate-500">Loading graph...</p>
          </div>
        </div>
      )}

      {graphError && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <div className="bg-white rounded-lg border border-red-200 p-6 max-w-sm text-center">
            <p className="text-sm text-red-600 mb-3">
              Failed to load graph data
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setStartMemoryId(undefined)}
            >
              Try Different Memory
            </Button>
          </div>
        </div>
      )}

      <MemoryGraph
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
      />

      <GraphControls
        layout={layout}
        onLayoutChange={setLayout}
        depth={depth}
        onDepthChange={setDepth}
        direction={direction}
        onDirectionChange={setDirection}
        minStrength={minStrength}
        onMinStrengthChange={setMinStrength}
        enabledCategories={enabledCategories}
        onCategoryToggle={handleCategoryToggle}
        onReset={handleReset}
      />

      <GraphLegend />

      {/* Back button */}
      <div className="absolute top-4 right-4 z-10">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setStartMemoryId(undefined);
            setSearchResults([]);
            setSearchQuery("");
            setSelectedNodeId(null);
          }}
          className="bg-white/80 backdrop-blur-xl"
        >
          New Search
        </Button>
      </div>

      {/* Side panel for selected node */}
      <Sheet
        open={!!selectedNodeId}
        onOpenChange={(open) => {
          if (!open) setSelectedNodeId(null);
        }}
      >
        <SheetContent side="right" className="w-[380px] overflow-y-auto">
          {selectedMemory ? (
            <>
              <SheetHeader>
                <SheetTitle className="text-base">Memory Detail</SheetTitle>
                <SheetDescription className="font-mono text-xs">
                  {selectedMemory.id}
                </SheetDescription>
              </SheetHeader>

              <div className="mt-4 space-y-4">
                {/* Type badge */}
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={cn(
                      MEMORY_TYPE_COLORS[selectedMemory.type]?.bg,
                      MEMORY_TYPE_COLORS[selectedMemory.type]?.text,
                    )}
                  >
                    {MEMORY_TYPE_LABELS[selectedMemory.type] ?? selectedMemory.type}
                  </Badge>
                  {selectedMemory.subtype && (
                    <Badge variant="secondary" className="text-xs">
                      {selectedMemory.subtype}
                    </Badge>
                  )}
                </div>

                {/* Content */}
                <div>
                  <p className="text-xs font-medium text-slate-500 mb-1">
                    Content
                  </p>
                  <p className="text-sm text-slate-700 whitespace-pre-wrap">
                    {selectedMemory.content}
                  </p>
                </div>

                {/* Importance */}
                <div>
                  <p className="text-xs font-medium text-slate-500 mb-1">
                    Importance
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${selectedMemory.importance * 100}%`,
                          backgroundColor:
                            selectedMemory.importance >= 0.7
                              ? "#ef4444"
                              : selectedMemory.importance >= 0.3
                                ? "#f59e0b"
                                : "#94a3b8",
                        }}
                      />
                    </div>
                    <span className="text-xs text-slate-500 tabular-nums">
                      {Math.round(selectedMemory.importance * 100)}%
                    </span>
                  </div>
                </div>

                {/* Tags */}
                {selectedMemory.tags.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-slate-500 mb-1">
                      Tags
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {selectedMemory.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Timestamps */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <p className="font-medium text-slate-500">Created</p>
                    <p className="text-slate-600">
                      {new Date(selectedMemory.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium text-slate-500">Updated</p>
                    <p className="text-slate-600">
                      {new Date(selectedMemory.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => {
                      setStartMemoryId(selectedMemory.id);
                      setSelectedNodeId(null);
                    }}
                  >
                    Walk from here
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    asChild
                  >
                    <a href={`/memories/${selectedMemory.id}`}>View Detail</a>
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <SheetHeader>
              <SheetTitle>No memory selected</SheetTitle>
            </SheetHeader>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
}

export default function GraphPage() {
  return (
    <ReactFlowProvider>
      <Suspense fallback={<div className="flex items-center justify-center h-[calc(100vh-4rem)]"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>}>
        <GraphPageInner />
      </Suspense>
    </ReactFlowProvider>
  );
}
