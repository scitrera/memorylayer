"use client";

import { useState } from "react";
import { Save, X } from "lucide-react";
import { toast } from "sonner";
import type { Memory } from "@/types";
import { MemoryType, MemorySubtype } from "@/types";
import { MEMORY_TYPE_LABELS, MEMORY_SUBTYPE_LABELS } from "@/lib/constants";
import { TagInput } from "@/components/shared/tag-input";
import { Slider } from "@/components/ui/slider";
import { useUpdateMemory } from "@/hooks/use-memories";

interface MemoryEditorProps {
  memory: Memory;
  onClose: () => void;
}

export function MemoryEditor({ memory, onClose }: MemoryEditorProps) {
  const [content, setContent] = useState(memory.content);
  const [type, setType] = useState<string>(memory.type);
  const [subtype, setSubtype] = useState<string>(memory.subtype ?? "");
  const [importance, setImportance] = useState(memory.importance);
  const [tags, setTags] = useState(memory.tags);
  const [metadataStr, setMetadataStr] = useState(
    JSON.stringify(memory.metadata, null, 2)
  );

  const updateMutation = useUpdateMemory();

  function handleSave() {
    let metadata: Record<string, unknown>;
    try {
      metadata = JSON.parse(metadataStr);
    } catch {
      toast.error("Invalid JSON in metadata");
      return;
    }

    updateMutation.mutate(
      {
        id: memory.id,
        updates: {
          content,
          type,
          subtype: subtype || undefined,
          importance,
          tags,
          metadata,
        },
      },
      {
        onSuccess: () => {
          toast.success("Memory updated");
          onClose();
        },
        onError: (err) => toast.error(`Update failed: ${err.message}`),
      }
    );
  }

  return (
    <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">Edit Memory</h3>
        <div className="flex gap-2">
          <button
            onClick={onClose}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-input bg-background px-3 text-xs font-medium shadow-sm hover:bg-accent"
          >
            <X className="h-3 w-3" />
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={updateMutation.isPending}
            className="inline-flex h-8 items-center gap-1 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground shadow hover:bg-primary/90 disabled:opacity-50"
          >
            <Save className="h-3 w-3" />
            {updateMutation.isPending ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-foreground">Content</label>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={6}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-xs font-medium text-foreground">Type</label>
          <select
            value={type}
            onChange={(e) => setType(e.target.value)}
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {Object.values(MemoryType).map((t) => (
              <option key={t} value={t}>
                {MEMORY_TYPE_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-foreground">Subtype</label>
          <select
            value={subtype}
            onChange={(e) => setSubtype(e.target.value)}
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="">None</option>
            {Object.values(MemorySubtype).map((s) => (
              <option key={s} value={s}>
                {MEMORY_SUBTYPE_LABELS[s] ?? s}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-foreground">
          Importance: {Math.round(importance * 100)}%
        </label>
        <Slider
          value={[importance]}
          onValueChange={([v]) => setImportance(v)}
          min={0}
          max={1}
          step={0.05}
        />
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-foreground">Tags</label>
        <TagInput tags={tags} onChange={setTags} />
      </div>

      <div className="space-y-2">
        <label className="text-xs font-medium text-foreground">
          Metadata (JSON)
        </label>
        <textarea
          value={metadataStr}
          onChange={(e) => setMetadataStr(e.target.value)}
          rows={4}
          className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>
    </div>
  );
}
