'use client';

import { useRouter } from 'next/navigation';
import { Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { StatsCards } from '@/components/analytics/stats-cards';
import { TypeDistribution } from '@/components/analytics/type-distribution';
import { ActivityTimeline } from '@/components/analytics/activity-timeline';
import { ImportanceHistogram } from '@/components/analytics/importance-histogram';
import { MemoryTypeBadge } from '@/components/memory/memory-type-badge';
import { TimeAgo } from '@/components/shared/time-ago';
import { useAnalytics } from '@/hooks/use-analytics';
import { truncateContent } from '@/lib/format';
import type { Memory } from '@/types';

export default function DashboardPage() {
  const router = useRouter();
  const { data: briefing, isLoading } = useAnalytics();

  const recentMemories = (briefing?.memories ?? []).slice(0, 10) as Array<
    Partial<Memory> & { id: string; type: string; created_at: string; abstract?: string }
  >;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-serif text-2xl font-semibold tracking-tight">
          Dashboard
        </h1>
        <p className="text-muted-foreground">
          Overview of your MemoryLayer workspace
        </p>
      </div>

      <StatsCards />

      <div className="grid gap-6 lg:grid-cols-2">
        <TypeDistribution />
        <ActivityTimeline />
      </div>

      <ImportanceHistogram />

      <Card className="rounded-2xl shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Clock className="h-4 w-4 text-muted-foreground" />
            Recent Memories
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : recentMemories.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No recent memories
            </p>
          ) : (
            <div className="divide-y">
              {recentMemories.map((memory) => (
                <button
                  key={memory.id}
                  onClick={() => router.push(`/memories/${memory.id}`)}
                  className="flex w-full items-center gap-3 px-1 py-3 text-left transition-colors hover:bg-muted/50"
                >
                  <MemoryTypeBadge type={memory.type} />
                  <span className="min-w-0 flex-1 truncate text-sm">
                    {truncateContent(memory.abstract || memory.content, 100)}
                  </span>
                  <TimeAgo
                    date={memory.created_at}
                    className="shrink-0 text-xs text-muted-foreground"
                  />
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
