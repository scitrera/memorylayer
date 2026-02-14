"use client";

import { useState } from "react";
import { LayoutGrid, List, Database, SlidersHorizontal } from "lucide-react";
import { useMemoryList } from "@/hooks/use-memories";
import { MemoryCard } from "@/components/memory/memory-card";
import { MemoryTableRow } from "@/components/memory/memory-table-row";
import { MemoryFilters } from "@/components/memory/memory-filters";
import { MemorySort } from "@/components/memory/memory-sort";
import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cn";
import type { ViewMode, SortField, SortOrder, FilterState } from "@/types";

const PAGE_SIZE = 50;

export default function MemoriesPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    types: [],
    subtypes: [],
    tags: [],
    status: "all",
    importanceRange: [0, 1],
    dateRange: {},
  });

  const { data, isLoading, isError, error } = useMemoryList({
    types: filters.types.length > 0 ? filters.types : undefined,
    subtypes: filters.subtypes.length > 0 ? filters.subtypes : undefined,
    tags: filters.tags.length > 0 ? filters.tags : undefined,
    limit: PAGE_SIZE,
    offset: (currentPage - 1) * PAGE_SIZE,
  });

  const memories = data?.memories ?? [];
  const totalCount = data?.total_count ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Memories</h1>
          <p className="text-muted-foreground">
            {totalCount > 0
              ? `${totalCount} memories in workspace`
              : "Browse and manage your stored memories"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors",
              showFilters
                ? "border-primary bg-primary/5 text-primary"
                : "border-input bg-background text-muted-foreground hover:bg-accent"
            )}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Filters
          </button>
          <div className="flex rounded-md border">
            <button
              onClick={() => setViewMode("grid")}
              className={cn(
                "inline-flex h-8 w-8 items-center justify-center transition-colors",
                viewMode === "grid"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent"
              )}
            >
              <LayoutGrid className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={cn(
                "inline-flex h-8 w-8 items-center justify-center transition-colors",
                viewMode === "list"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent"
              )}
            >
              <List className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      <div className="flex gap-6">
        {showFilters && (
          <div className="w-56 shrink-0 rounded-2xl border border-slate-200 bg-white p-4">
            <MemoryFilters filters={filters} onChange={setFilters} />
          </div>
        )}

        <div className="flex-1 space-y-4">
          <MemorySort
            sortField={sortField}
            sortOrder={sortOrder}
            onSortChange={(field, order) => {
              setSortField(field);
              setSortOrder(order);
            }}
          />

          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full rounded-2xl" />
              ))}
            </div>
          ) : isError ? (
            <EmptyState
              title="Failed to load memories"
              description={
                error instanceof Error
                  ? error.message
                  : "Check your connection settings"
              }
            />
          ) : memories.length === 0 ? (
            <EmptyState
              icon={Database}
              title="No memories found"
              description="No memories match your current filters. Try adjusting the filters or create new memories."
            />
          ) : viewMode === "grid" ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {memories.map((memory) => (
                <MemoryCard key={memory.id} memory={memory} />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white">
              <table className="w-full">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">Content</th>
                    <th className="px-4 py-3 font-medium">Importance</th>
                    <th className="px-4 py-3 font-medium">Tags</th>
                    <th className="px-4 py-3 font-medium">Created</th>
                    <th className="px-4 py-3 font-medium w-10"></th>
                  </tr>
                </thead>
                <tbody>
                  {memories.map((memory) => (
                    <MemoryTableRow key={memory.id} memory={memory} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex justify-center">
            <Pagination
              currentPage={currentPage}
              totalPages={totalPages}
              onPageChange={setCurrentPage}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
