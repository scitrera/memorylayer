'use client';

import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useAnalytics } from '@/hooks/use-analytics';
import { MEMORY_TYPE_LABELS } from '@/lib/constants';

const CHART_COLORS: Record<string, string> = {
  episodic: '#3b82f6',
  semantic: '#10b981',
  procedural: '#f59e0b',
  working: '#6b7280',
};

export function TypeDistribution() {
  const { data: briefing, isLoading } = useAnalytics();
  const memoryTypes = briefing?.workspace_summary?.memory_types;

  const chartData = memoryTypes
    ? Object.entries(memoryTypes)
        .filter(([, count]) => count > 0)
        .map(([type, count]) => ({
          name: MEMORY_TYPE_LABELS[type] ?? type,
          value: count,
          type,
        }))
    : [];

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader>
        <CardTitle className="text-base">Memory Types</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex h-[250px] items-center justify-center">
            <Skeleton className="h-[200px] w-[200px] rounded-full" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex h-[250px] items-center justify-center text-sm text-muted-foreground">
            No memory data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {chartData.map((entry) => (
                  <Cell
                    key={entry.type}
                    fill={CHART_COLORS[entry.type] ?? '#94a3b8'}
                    stroke="none"
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number, name: string) => [value, name]}
                contentStyle={{
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value: string) => (
                  <span className="text-xs text-foreground">{value}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
