"use client";

import Link from "next/link";
import type { Memory } from "@/types";
import { truncateContent } from "@/lib/format";
import { cn } from "@/lib/cn";
import { MemoryTypeBadge } from "./memory-type-badge";
import { ImportanceIndicator } from "./importance-indicator";
import { TimeAgo } from "@/components/shared/time-ago";

const LEFT_BORDER_COLORS: Record<string, string> = {
  episodic: "border-l-blue-400",
  semantic: "border-l-emerald-400",
  procedural: "border-l-amber-400",
  working: "border-l-purple-400",
};

interface MemoryCardProps {
  memory: Memory;
}

export function MemoryCard({ memory }: MemoryCardProps) {
  const leftBorder = LEFT_BORDER_COLORS[memory.type] ?? "border-l-slate-300";

  return (
    <Link
      href={`/memories/${memory.id}`}
      className={cn(
        "group block rounded-2xl border border-slate-200 border-l-4 bg-white p-4 shadow-sm transition-shadow hover:shadow-md",
        leftBorder
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <MemoryTypeBadge type={memory.type} subtype={memory.subtype} />
        <ImportanceIndicator value={memory.importance} size="sm" />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-foreground">
        {truncateContent(memory.content, 120)}
      </p>

      {memory.tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {memory.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="rounded-md bg-secondary px-1.5 py-0.5 text-xs text-secondary-foreground"
            >
              {tag}
            </span>
          ))}
          {memory.tags.length > 4 && (
            <span className="rounded-md bg-secondary px-1.5 py-0.5 text-xs text-muted-foreground">
              +{memory.tags.length - 4}
            </span>
          )}
        </div>
      )}

      <div className="mt-3 text-xs text-muted-foreground">
        <TimeAgo date={memory.created_at} />
      </div>
    </Link>
  );
}
