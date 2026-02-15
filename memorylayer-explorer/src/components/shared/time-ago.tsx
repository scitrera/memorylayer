"use client";

import { formatRelativeTime, formatDate } from "@/lib/format";

interface TimeAgoProps {
  date: string | undefined | null;
  className?: string;
}

export function TimeAgo({ date, className }: TimeAgoProps) {
  return (
    <time
      dateTime={date ?? undefined}
      title={formatDate(date)}
      className={className}
    >
      {formatRelativeTime(date)}
    </time>
  );
}
