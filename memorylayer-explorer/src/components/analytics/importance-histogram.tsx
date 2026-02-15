'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useMemoryStats } from '@/hooks/use-analytics';

const BAR_COLORS = ['#94a3b8', '#64748b', '#6366f1', '#4f46e5', '#3b82f6'];

export function ImportanceHistogram() {
  const { data: buckets, isLoading } = useMemoryStats();

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader>
        <CardTitle className="text-base">Importance Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex h-[250px] items-center justify-center">
            <Skeleton className="h-[200px] w-full" />
          </div>
        ) : !buckets || buckets.every((b) => b.count === 0) ? (
          <div className="flex h-[250px] items-center justify-center text-sm text-muted-foreground">
            No memory data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={buckets}>
              <XAxis
                dataKey="range"
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: '#e2e8f0' }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                }}
                formatter={(value: number) => [value, 'Memories']}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {buckets?.map((_, index) => (
                  <Cell key={index} fill={BAR_COLORS[index] ?? '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
