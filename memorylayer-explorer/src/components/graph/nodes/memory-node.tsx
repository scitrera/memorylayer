"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Brain, BookOpen, Cog, Clock } from "lucide-react";
import { cn } from "@/lib/cn";

const typeIcons: Record<string, typeof Brain> = {
  episodic: Clock,
  semantic: BookOpen,
  procedural: Cog,
  working: Brain,
};

function ImportanceIndicator({ value }: { value: number }) {
  const radius = 8;
  const circumference = 2 * Math.PI * radius;
  const filled = circumference * value;

  return (
    <svg width="20" height="20" viewBox="0 0 20 20" className="shrink-0">
      <circle
        cx="10"
        cy="10"
        r={radius}
        fill="none"
        stroke="#e2e8f0"
        strokeWidth="2"
      />
      <circle
        cx="10"
        cy="10"
        r={radius}
        fill="none"
        stroke={value >= 0.7 ? "#ef4444" : value >= 0.3 ? "#f59e0b" : "#94a3b8"}
        strokeWidth="2"
        strokeDasharray={`${filled} ${circumference - filled}`}
        strokeDashoffset={circumference * 0.25}
        strokeLinecap="round"
      />
    </svg>
  );
}

function MemoryNodeComponent({ data, selected }: NodeProps) {
  const memoryType = (data.memoryType as string) ?? "episodic";
  const borderColor = (data.borderColor as string) ?? "#3b82f6";
  const Icon = typeIcons[memoryType] ?? Brain;
  const importance = (data.memory as { importance?: number } | undefined)?.importance ?? 0.5;
  const label = (data.label as string) ?? "";

  return (
    <>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-slate-400" />
      <div
        className={cn(
          "w-[240px] h-[80px] rounded-xl bg-white shadow-sm border border-slate-200 flex items-center gap-3 px-3 cursor-pointer transition-shadow",
          selected && "ring-2 ring-blue-500 shadow-md",
        )}
        style={{ borderLeftWidth: 4, borderLeftColor: borderColor }}
      >
        <Icon className="w-5 h-5 shrink-0 text-slate-500" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-800 truncate leading-snug">
            {label}
          </p>
          <p className="text-xs text-slate-400 capitalize">{memoryType}</p>
        </div>
        <ImportanceIndicator value={importance} />
      </div>
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-slate-400" />
    </>
  );
}

export const MemoryNode = memo(MemoryNodeComponent);
