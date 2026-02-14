'use client';

import { useQuery } from '@tanstack/react-query';
import { useConnection } from '@/providers/connection-provider';
import type { SessionBriefing } from '@/types';

export function useAnalytics() {
  const { client, connectionConfig } = useConnection();
  return useQuery<SessionBriefing>({
    queryKey: ['analytics', connectionConfig.workspaceId],
    queryFn: () => client.getBriefing(24, true),
    staleTime: 60 * 1000,
  });
}

export interface ImportanceBucket {
  range: string;
  count: number;
}

export function useMemoryStats() {
  const { client } = useConnection();
  return useQuery<ImportanceBucket[]>({
    queryKey: ['analytics', 'importance-distribution'],
    queryFn: async () => {
      const result = await client.recall('*', {
        limit: 100,
        detailLevel: 'abstract',
      });
      const buckets: ImportanceBucket[] = [
        { range: '0-0.2', count: 0 },
        { range: '0.2-0.4', count: 0 },
        { range: '0.4-0.6', count: 0 },
        { range: '0.6-0.8', count: 0 },
        { range: '0.8-1.0', count: 0 },
      ];
      for (const memory of result.memories) {
        const imp = memory.importance;
        if (imp < 0.2) buckets[0].count++;
        else if (imp < 0.4) buckets[1].count++;
        else if (imp < 0.6) buckets[2].count++;
        else if (imp < 0.8) buckets[3].count++;
        else buckets[4].count++;
      }
      return buckets;
    },
    staleTime: 60 * 1000,
  });
}
