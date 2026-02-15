import { cn } from "@/lib/cn";
import { MEMORY_TYPE_COLORS, MEMORY_TYPE_LABELS } from "@/lib/constants";

interface MemoryTypeBadgeProps {
  type: string;
  subtype?: string;
  className?: string;
}

export function MemoryTypeBadge({ type, subtype, className }: MemoryTypeBadgeProps) {
  const colors = MEMORY_TYPE_COLORS[type] ?? {
    bg: "bg-slate-50",
    text: "text-slate-700",
    border: "border-slate-200",
  };
  const label = MEMORY_TYPE_LABELS[type] ?? type;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        colors.bg,
        colors.text,
        colors.border,
        className
      )}
    >
      {label}
      {subtype && (
        <span className="opacity-70">/ {subtype}</span>
      )}
    </span>
  );
}
