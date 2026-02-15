"use client";

import { use } from "react";
import { useMemory, useMemoryAssociations } from "@/hooks/use-memories";
import { MemoryDetail } from "@/components/memory/memory-detail";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";

export default function MemoryDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: memory, isLoading, isError, error } = useMemory(id);
  const { data: associations = [] } = useMemoryAssociations(id);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-96 w-full rounded-2xl" />
      </div>
    );
  }

  if (isError || !memory) {
    return (
      <EmptyState
        title="Memory not found"
        description={
          error instanceof Error
            ? error.message
            : "The memory could not be loaded"
        }
      />
    );
  }

  return <MemoryDetail memory={memory} associations={associations} />;
}
