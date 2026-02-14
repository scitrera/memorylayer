"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { JsonViewer } from "@/components/shared/json-viewer";

interface MemoryMetadataProps {
  metadata: Record<string, unknown>;
}

export function MemoryMetadata({ metadata }: MemoryMetadataProps) {
  const [expanded, setExpanded] = useState(false);
  const entries = Object.keys(metadata);

  if (entries.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">No metadata</div>
    );
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-sm font-medium text-foreground"
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        Metadata ({entries.length} {entries.length === 1 ? "field" : "fields"})
      </button>
      {expanded && (
        <div className="mt-2">
          <JsonViewer data={metadata} defaultExpanded />
        </div>
      )}
    </div>
  );
}
