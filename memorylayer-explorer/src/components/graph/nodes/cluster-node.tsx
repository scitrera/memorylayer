"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Layers } from "lucide-react";

function ClusterNodeComponent({ data }: NodeProps) {
  const count = (data.count as number) ?? 0;
  const onExpand = data.onExpand as (() => void) | undefined;

  return (
    <>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-slate-400" />
      <button
        onClick={onExpand}
        className="w-[180px] h-[70px] rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 flex flex-col items-center justify-center gap-1 cursor-pointer hover:border-slate-400 hover:bg-slate-100 transition-colors"
      >
        <Layers className="w-5 h-5 text-slate-400" />
        <span className="text-xs font-medium text-slate-500">
          {count} memories
        </span>
      </button>
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-slate-400" />
    </>
  );
}

export const ClusterNode = memo(ClusterNodeComponent);
