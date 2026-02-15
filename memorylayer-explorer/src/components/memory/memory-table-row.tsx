"use client";

import Link from "next/link";
import type { Memory } from "@/types";
import { truncateContent } from "@/lib/format";
import { MemoryTypeBadge } from "./memory-type-badge";
import { ImportanceIndicator } from "./importance-indicator";
import { MemoryActions } from "./memory-actions";
import { TimeAgo } from "@/components/shared/time-ago";

interface MemoryTableRowProps {
  memory: Memory;
}

export function MemoryTableRow({ memory }: MemoryTableRowProps) {
  return (
    <tr className="border-b transition-colors hover:bg-muted/50">
      <td className="px-4 py-3">
        <MemoryTypeBadge type={memory.type} subtype={memory.subtype} />
      </td>
      <td className="max-w-md px-4 py-3">
        <Link
          href={`/memories/${memory.id}`}
          className="text-sm text-foreground hover:text-primary hover:underline"
        >
          {truncateContent(memory.content, 100)}
        </Link>
      </td>
      <td className="px-4 py-3">
        <ImportanceIndicator value={memory.importance} size="sm" showLabel />
      </td>
      <td className="px-4 py-3">
        <div className="flex flex-wrap gap-1">
          {memory.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="rounded-md bg-secondary px-1.5 py-0.5 text-xs text-secondary-foreground"
            >
              {tag}
            </span>
          ))}
          {memory.tags.length > 3 && (
            <span className="text-xs text-muted-foreground">
              +{memory.tags.length - 3}
            </span>
          )}
        </div>
      </td>
      <td className="px-4 py-3 text-xs text-muted-foreground">
        <TimeAgo date={memory.created_at} />
      </td>
      <td className="px-4 py-3">
        <MemoryActions memory={memory} />
      </td>
    </tr>
  );
}
