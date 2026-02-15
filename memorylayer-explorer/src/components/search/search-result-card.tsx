'use client';

import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/cn';
import { truncateContent, formatRelativeTime, formatImportance } from '@/lib/format';
import {
  MEMORY_TYPE_COLORS,
  MEMORY_TYPE_LABELS,
  getImportanceColor,
} from '@/lib/constants';
import type { Memory } from '@/types';

interface SearchResultCardProps {
  memory: Memory;
}

function getRelevanceColor(score: number): string {
  if (score >= 0.7) return 'bg-emerald-500';
  if (score >= 0.4) return 'bg-amber-500';
  return 'bg-red-400';
}

export function SearchResultCard({ memory }: SearchResultCardProps) {
  const score = memory.relevance_score ?? 0;
  const typeColors = MEMORY_TYPE_COLORS[memory.type] ?? {
    bg: 'bg-slate-50',
    text: 'text-slate-700',
    border: 'border-slate-200',
  };

  return (
    <Link
      href={`/memories/${memory.id}`}
      className="group flex gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md"
    >
      {/* Relevance bar */}
      <div className="flex w-2 flex-shrink-0 items-stretch">
        <div className="relative w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className={cn('absolute bottom-0 w-full rounded-full transition-all', getRelevanceColor(score))}
            style={{ height: `${Math.max(score * 100, 4)}%` }}
          />
        </div>
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              variant="outline"
              className={cn('text-[10px]', typeColors.bg, typeColors.text, typeColors.border)}
            >
              {MEMORY_TYPE_LABELS[memory.type] ?? memory.type}
            </Badge>
            {memory.subtype && (
              <Badge variant="outline" className="text-[10px]">
                {memory.subtype}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground flex-shrink-0">
            <span className={cn('font-medium', getImportanceColor(memory.importance))}>
              {formatImportance(memory.importance)}
            </span>
            <span>{formatRelativeTime(memory.created_at)}</span>
          </div>
        </div>

        <p className="text-sm text-foreground leading-relaxed">
          {truncateContent(memory.content, 200)}
        </p>

        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-wrap gap-1">
            {memory.tags.slice(0, 5).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[10px] font-normal">
                {tag}
              </Badge>
            ))}
            {memory.tags.length > 5 && (
              <Badge variant="secondary" className="text-[10px] font-normal">
                +{memory.tags.length - 5}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs flex-shrink-0">
            <span className="text-muted-foreground">
              Score: {(score * 100).toFixed(0)}%
            </span>
            <Link
              href={`/graph?from=${memory.id}`}
              className="text-primary hover:underline"
              onClick={(e) => e.stopPropagation()}
            >
              Graph
            </Link>
          </div>
        </div>
      </div>
    </Link>
  );
}
