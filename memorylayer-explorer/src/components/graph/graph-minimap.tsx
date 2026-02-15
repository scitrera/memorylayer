"use client";

import { MiniMap } from "@xyflow/react";
import { memoryTypeColors } from "@/lib/colors";

function nodeColor(node: { data?: Record<string, unknown> }): string {
  const memType = (node.data?.memoryType as string) ?? "episodic";
  return memoryTypeColors[memType] ?? "#94a3b8";
}

export function GraphMinimap() {
  return (
    <MiniMap
      nodeColor={nodeColor}
      maskColor="rgba(240, 240, 240, 0.6)"
      className="!bottom-4 !right-4 rounded-lg border border-slate-200 shadow-sm"
      pannable
      zoomable
    />
  );
}
