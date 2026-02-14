import { cn } from "@/lib/cn";
import { getImportanceColor } from "@/lib/constants";
import { formatImportance } from "@/lib/format";

interface ImportanceIndicatorProps {
  value: number;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

const sizeMap = {
  sm: { outer: 24, inner: 18, stroke: 3 },
  md: { outer: 32, inner: 24, stroke: 3 },
  lg: { outer: 48, inner: 38, stroke: 4 },
};

export function ImportanceIndicator({
  value,
  size = "md",
  showLabel = false,
  className,
}: ImportanceIndicatorProps) {
  const { outer, inner, stroke } = sizeMap[size];
  const radius = (outer - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(1, Math.max(0, value)));
  const color = getImportanceColor(value);

  return (
    <div className={cn("inline-flex items-center gap-1.5", className)}>
      <svg
        width={outer}
        height={outer}
        viewBox={`0 0 ${outer} ${outer}`}
        className="-rotate-90"
      >
        <circle
          cx={outer / 2}
          cy={outer / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-slate-200"
        />
        <circle
          cx={outer / 2}
          cy={outer / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={color}
        />
      </svg>
      {showLabel && (
        <span className={cn("text-xs font-medium", color)}>
          {formatImportance(value)}
        </span>
      )}
    </div>
  );
}
