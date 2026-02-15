import { format, formatDistanceToNow, parseISO, isValid } from "date-fns";

export function formatDate(dateStr: string | undefined | null): string {
  if (!dateStr) return "N/A";
  const date = parseISO(dateStr);
  if (!isValid(date)) return "Invalid date";
  return format(date, "MMM d, yyyy HH:mm");
}

export function formatRelativeTime(dateStr: string | undefined | null): string {
  if (!dateStr) return "N/A";
  const date = parseISO(dateStr);
  if (!isValid(date)) return "Invalid date";
  return formatDistanceToNow(date, { addSuffix: true });
}

export function formatShortDate(dateStr: string | undefined | null): string {
  if (!dateStr) return "N/A";
  const date = parseISO(dateStr);
  if (!isValid(date)) return "Invalid date";
  return format(date, "MMM d");
}

export function truncateContent(content: string | undefined | null, maxLength: number = 200): string {
  if (!content) return "";
  if (content.length <= maxLength) return content;
  return content.slice(0, maxLength).trimEnd() + "...";
}

export function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return num.toString();
}

export function formatImportance(value: number): string {
  return (value * 100).toFixed(0) + "%";
}

export function formatStrength(value: number): string {
  return (value * 100).toFixed(0) + "%";
}

export function formatLatency(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}
