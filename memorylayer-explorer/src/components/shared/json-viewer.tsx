"use client";

import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import { cn } from "@/lib/cn";

interface JsonViewerProps {
  data: Record<string, unknown> | unknown[] | unknown;
  defaultExpanded?: boolean;
  className?: string;
}

function JsonNode({
  label,
  value,
  defaultExpanded = false,
}: {
  label?: string;
  value: unknown;
  defaultExpanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  if (value === null || value === undefined) {
    return (
      <div className="flex items-baseline gap-1 py-0.5">
        {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
        <span className="text-muted-foreground font-mono text-xs italic">
          {value === null ? "null" : "undefined"}
        </span>
      </div>
    );
  }

  if (typeof value === "string") {
    return (
      <div className="flex items-baseline gap-1 py-0.5">
        {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
        <span className="text-green-700 font-mono text-xs">&quot;{value}&quot;</span>
      </div>
    );
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return (
      <div className="flex items-baseline gap-1 py-0.5">
        {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
        <span className="text-blue-700 font-mono text-xs">{String(value)}</span>
      </div>
    );
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <div className="flex items-baseline gap-1 py-0.5">
          {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
          <span className="text-muted-foreground font-mono text-xs">[]</span>
        </div>
      );
    }

    return (
      <div className="py-0.5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-0.5 text-xs"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
          {label && <span className="text-violet-600 font-mono">{label}:</span>}
          <span className="text-muted-foreground font-mono">
            [{value.length}]
          </span>
        </button>
        {expanded && (
          <div className="ml-4 border-l border-slate-200 pl-2">
            {value.map((item, idx) => (
              <JsonNode key={idx} label={String(idx)} value={item} />
            ))}
          </div>
        )}
      </div>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return (
        <div className="flex items-baseline gap-1 py-0.5">
          {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
          <span className="text-muted-foreground font-mono text-xs">{"{}"}</span>
        </div>
      );
    }

    return (
      <div className="py-0.5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-0.5 text-xs"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
          {label && <span className="text-violet-600 font-mono">{label}:</span>}
          <span className="text-muted-foreground font-mono">
            {"{"}
            {entries.length}
            {"}"}
          </span>
        </button>
        {expanded && (
          <div className="ml-4 border-l border-slate-200 pl-2">
            {entries.map(([key, val]) => (
              <JsonNode key={key} label={key} value={val} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-baseline gap-1 py-0.5">
      {label && <span className="text-violet-600 font-mono text-xs">{label}:</span>}
      <span className="text-muted-foreground font-mono text-xs">{String(value)}</span>
    </div>
  );
}

export function JsonViewer({ data, defaultExpanded = true, className }: JsonViewerProps) {
  return (
    <div className={cn("rounded-lg bg-slate-50 p-3 font-mono text-xs", className)}>
      <JsonNode value={data} defaultExpanded={defaultExpanded} />
    </div>
  );
}
