"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

function GhostNodeComponent({ data }: NodeProps) {
  const count = (data.count as number) ?? 0;
  const onExpand = data.onExpand as (() => void) | undefined;

  return (
    <>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-transparent" />
      <button
        onClick={onExpand}
        className="w-[60px] h-[60px] rounded-full border-2 border-dashed border-slate-300 bg-white/50 flex items-center justify-center cursor-pointer hover:border-slate-400 hover:bg-slate-50 transition-colors"
      >
        <span className="text-sm font-semibold text-slate-400">
          +{count}
        </span>
      </button>
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-transparent" />
    </>
  );
}

export const GhostNode = memo(GhostNodeComponent);
