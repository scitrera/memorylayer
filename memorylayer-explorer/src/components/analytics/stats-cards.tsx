'use client';

import { Database, GitBranch, Tag, Clock } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useAnalytics } from '@/hooks/use-analytics';
import { formatNumber } from '@/lib/format';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: number | undefined;
  isLoading: boolean;
  accentColor: string;
  iconBgColor: string;
}

function StatCard({ icon: Icon, label, value, isLoading, accentColor, iconBgColor }: StatCardProps) {
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="flex items-center gap-4 p-6">
        <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full ${iconBgColor}`}>
          <Icon className={`h-6 w-6 ${accentColor}`} />
        </div>
        <div>
          {isLoading ? (
            <>
              <Skeleton className="h-7 w-16" />
              <Skeleton className="mt-1 h-4 w-24" />
            </>
          ) : (
            <>
              <p className="text-2xl font-semibold">{value != null ? formatNumber(value) : '--'}</p>
              <p className="text-sm text-muted-foreground">{label}</p>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function StatsCards() {
  const { data: briefing, isLoading } = useAnalytics();
  const summary = briefing?.workspace_summary;

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        icon={Database}
        label="Total Memories"
        value={summary?.total_memories}
        isLoading={isLoading}
        accentColor="text-brand-500"
        iconBgColor="bg-brand-50"
      />
      <StatCard
        icon={GitBranch}
        label="Total Associations"
        value={summary?.total_associations}
        isLoading={isLoading}
        accentColor="text-emerald-500"
        iconBgColor="bg-emerald-50"
      />
      <StatCard
        icon={Tag}
        label="Active Topics"
        value={summary?.active_topics?.length}
        isLoading={isLoading}
        accentColor="text-amber-500"
        iconBgColor="bg-amber-50"
      />
      <StatCard
        icon={Clock}
        label="Recent Memories 24h"
        value={summary?.recent_memories}
        isLoading={isLoading}
        accentColor="text-purple-500"
        iconBgColor="bg-purple-50"
      />
    </div>
  );
}
