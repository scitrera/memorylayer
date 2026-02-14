"use client";

import { useState } from "react";
import Link from "next/link";
import { Pencil, GitBranch, ArrowLeft } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Memory, Association } from "@/types";
import { MemoryTypeBadge } from "./memory-type-badge";
import { ImportanceIndicator } from "./importance-indicator";
import { MemoryMetadata } from "./memory-metadata";
import { MemoryAssociationsList } from "./memory-associations-list";
import { MemoryActions } from "./memory-actions";
import { MemoryEditor } from "./memory-editor";
import { TimeAgo } from "@/components/shared/time-ago";
import { formatDate } from "@/lib/format";

interface MemoryDetailProps {
  memory: Memory;
  associations: Association[];
}

export function MemoryDetail({ memory, associations }: MemoryDetailProps) {
  const [editing, setEditing] = useState(false);

  if (editing) {
    return <MemoryEditor memory={memory} onClose={() => setEditing(false)} />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link
          href="/memories"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground hover:bg-accent"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <MemoryTypeBadge type={memory.type} subtype={memory.subtype} />
            <span className="font-mono text-xs text-muted-foreground">
              {memory.id}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setEditing(true)}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-input bg-background px-3 text-xs font-medium shadow-sm hover:bg-accent"
          >
            <Pencil className="h-3 w-3" />
            Edit
          </button>
          <Link
            href={`/graph?from=${memory.id}`}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-input bg-background px-3 text-xs font-medium shadow-sm hover:bg-accent"
          >
            <GitBranch className="h-3 w-3" />
            Graph
          </Link>
          <MemoryActions memory={memory} />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <Tabs defaultValue="full">
              <TabsList>
                <TabsTrigger value="full">Full Content</TabsTrigger>
                <TabsTrigger value="abstract" disabled={!memory.abstract}>
                  Abstract
                </TabsTrigger>
                <TabsTrigger value="overview" disabled={!memory.overview}>
                  Overview
                </TabsTrigger>
              </TabsList>
              <TabsContent value="full" className="mt-4">
                <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {memory.content}
                </div>
              </TabsContent>
              <TabsContent value="abstract" className="mt-4">
                <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {memory.abstract ?? "No abstract available"}
                </div>
              </TabsContent>
              <TabsContent value="overview" className="mt-4">
                <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                  {memory.overview ?? "No overview available"}
                </div>
              </TabsContent>
            </Tabs>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <h3 className="mb-4 text-sm font-medium text-foreground">
              Associations ({associations.length})
            </h3>
            <MemoryAssociationsList
              associations={associations}
              currentMemoryId={memory.id}
            />
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 space-y-4">
            <h3 className="text-sm font-medium text-foreground">Properties</h3>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Importance</span>
                <ImportanceIndicator value={memory.importance} size="md" showLabel />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Access Count</span>
                <span className="text-sm font-medium">{memory.access_count}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Decay Factor</span>
                <span className="text-sm font-medium">
                  {(memory.decay_factor * 100).toFixed(0)}%
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Created</span>
                <span className="text-xs text-foreground">
                  {formatDate(memory.created_at)}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Updated</span>
                <span className="text-xs text-foreground">
                  <TimeAgo date={memory.updated_at} />
                </span>
              </div>

              {memory.last_accessed_at && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Last Accessed</span>
                  <span className="text-xs text-foreground">
                    <TimeAgo date={memory.last_accessed_at} />
                  </span>
                </div>
              )}

              {memory.session_id && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Session</span>
                  <Link
                    href={`/sessions/${memory.session_id}`}
                    className="font-mono text-xs text-primary hover:underline"
                  >
                    {memory.session_id.slice(0, 12)}...
                  </Link>
                </div>
              )}

              {memory.context_id && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Context</span>
                  <span className="text-xs text-foreground">{memory.context_id}</span>
                </div>
              )}
            </div>
          </div>

          {memory.tags.length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-6">
              <h3 className="mb-3 text-sm font-medium text-foreground">Tags</h3>
              <div className="flex flex-wrap gap-1.5">
                {memory.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-md bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <MemoryMetadata metadata={memory.metadata} />
          </div>
        </div>
      </div>
    </div>
  );
}
