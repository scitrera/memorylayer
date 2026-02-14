'use client';

import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { formatLatency } from '@/lib/format';
import { SearchResultCard } from './search-result-card';
import type { RecallResult } from '@/types';

interface SearchResultsProps {
  data: RecallResult | undefined;
  isLoading: boolean;
  hasQuery: boolean;
}

function ResultSkeleton() {
  return (
    <div className="flex gap-3 rounded-2xl border border-slate-200 bg-white p-4">
      <Skeleton className="h-16 w-2 rounded-full" />
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-16 rounded-full" />
          <Skeleton className="h-5 w-12 rounded-full" />
        </div>
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <div className="flex gap-1">
          <Skeleton className="h-4 w-10 rounded-full" />
          <Skeleton className="h-4 w-12 rounded-full" />
        </div>
      </div>
    </div>
  );
}

function getModeColor(mode: string): string {
  switch (mode) {
    case 'rag':
      return 'bg-blue-50 text-blue-700 border-blue-200';
    case 'llm':
      return 'bg-purple-50 text-purple-700 border-purple-200';
    case 'hybrid':
      return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    default:
      return 'bg-slate-50 text-slate-700 border-slate-200';
  }
}

export function SearchResults({ data, isLoading, hasQuery }: SearchResultsProps) {
  if (!hasQuery) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-48" />
        <ResultSkeleton />
        <ResultSkeleton />
        <ResultSkeleton />
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const { memories, mode_used, search_latency_ms, total_count, query_rewritten } = data;

  if (memories.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center">
        <p className="text-lg font-medium text-foreground">No results found</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Try adjusting your query or relaxing the filters
        </p>
        <ul className="mt-4 space-y-1 text-sm text-muted-foreground">
          <li>Use more general terms</li>
          <li>Lower the minimum relevance threshold</li>
          <li>Remove type or subtype filters</li>
          <li>Try a different search mode (LLM or Hybrid)</li>
        </ul>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Results header */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge variant="outline" className={cn('text-xs', getModeColor(mode_used))}>
          {mode_used.toUpperCase()}
        </Badge>
        <span className="text-muted-foreground">
          {formatLatency(search_latency_ms)}
        </span>
        <span className="text-muted-foreground">
          {total_count} {total_count === 1 ? 'result' : 'results'}
        </span>
        {query_rewritten && (
          <span className="text-xs text-muted-foreground italic">
            Rewritten: &quot;{query_rewritten}&quot;
          </span>
        )}
      </div>

      {/* Result cards */}
      <div className="space-y-2">
        {memories.map((memory) => (
          <SearchResultCard key={memory.id} memory={memory} />
        ))}
      </div>
    </div>
  );
}
