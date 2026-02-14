"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  type EdgeTypes,
  type OnNodesChange,
  type OnEdgesChange,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { MemoryNode } from "./nodes/memory-node";
import { ClusterNode } from "./nodes/cluster-node";
import { GhostNode } from "./nodes/ghost-node";
import { RelationshipEdge } from "./edges/relationship-edge";
import { GraphMinimap } from "./graph-minimap";

const nodeTypes: NodeTypes = {
  memoryNode: MemoryNode,
  clusterNode: ClusterNode,
  ghostNode: GhostNode,
};

const edgeTypes: EdgeTypes = {
  relationshipEdge: RelationshipEdge,
};

interface MemoryGraphProps {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onNodeClick?: (event: React.MouseEvent, node: Node) => void;
  onNodeDoubleClick?: (event: React.MouseEvent, node: Node) => void;
}

export function MemoryGraph({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onNodeClick,
  onNodeDoubleClick,
}: MemoryGraphProps) {
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      onNodeDoubleClick={onNodeDoubleClick}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.1}
      maxZoom={2}
      defaultEdgeOptions={{ animated: false }}
      proOptions={{ hideAttribution: true }}
      className="bg-slate-50"
    >
      <Background variant={BackgroundVariant.Dots} color="#cbd5e1" gap={20} size={1} />
      <Controls
        position="bottom-right"
        className="!bottom-20 !right-4 rounded-lg border border-slate-200 shadow-sm"
      />
      <GraphMinimap />
    </ReactFlow>
  );
}
