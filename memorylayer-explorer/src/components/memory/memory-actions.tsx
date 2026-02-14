"use client";

import { useState } from "react";
import Link from "next/link";
import { MoreHorizontal, Trash2, TrendingDown, GitBranch, Eye } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { useForget, useDecay } from "@/hooks/use-memories";
import { toast } from "sonner";
import type { Memory } from "@/types";

interface MemoryActionsProps {
  memory: Memory;
}

export function MemoryActions({ memory }: MemoryActionsProps) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [hardDelete, setHardDelete] = useState(false);
  const forgetMutation = useForget();
  const decayMutation = useDecay();

  function handleDecay() {
    decayMutation.mutate(
      { id: memory.id, rate: 0.1 },
      {
        onSuccess: () => toast.success("Memory decay applied"),
        onError: (err) => toast.error(`Decay failed: ${err.message}`),
      }
    );
  }

  function handleDelete(hard: boolean) {
    forgetMutation.mutate(
      { id: memory.id, hard },
      {
        onSuccess: () =>
          toast.success(hard ? "Memory permanently deleted" : "Memory archived"),
        onError: (err) => toast.error(`Delete failed: ${err.message}`),
      }
    );
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent">
            <MoreHorizontal className="h-4 w-4" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <Link href={`/memories/${memory.id}`}>
              <Eye className="mr-2 h-4 w-4" />
              View Details
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href={`/graph?from=${memory.id}`}>
              <GitBranch className="mr-2 h-4 w-4" />
              View in Graph
            </Link>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={handleDecay}>
            <TrendingDown className="mr-2 h-4 w-4" />
            Apply Decay
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => {
              setHardDelete(false);
              setDeleteOpen(true);
            }}
            className="text-destructive"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Archive
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => {
              setHardDelete(true);
              setDeleteOpen(true);
            }}
            className="text-destructive"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete Permanently
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <ConfirmDialog
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        title={hardDelete ? "Permanently Delete Memory" : "Archive Memory"}
        description={
          hardDelete
            ? "This will permanently delete the memory and cannot be undone."
            : "This will soft-delete the memory. It can be recovered later."
        }
        confirmLabel={hardDelete ? "Delete" : "Archive"}
        destructive
        onConfirm={() => handleDelete(hardDelete)}
      />
    </>
  );
}
