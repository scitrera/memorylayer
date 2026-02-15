"use client";

import { useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useConnection } from "@/providers/connection-provider";
import { useWorkspaceList } from "@/hooks/use-workspaces";

export function WorkspaceSwitcher() {
  const { connectionConfig, setConnectionConfig, isConnected } = useConnection();
  const { data: workspaces, isLoading } = useWorkspaceList();
  const [filter, setFilter] = useState("");

  if (!isConnected) {
    return (
      <span className="rounded-md bg-secondary px-2 py-1 text-xs font-medium text-secondary-foreground">
        {connectionConfig.workspaceId || "_default"}
      </span>
    );
  }

  const filteredWorkspaces = workspaces?.filter(
    (w) =>
      w.name.toLowerCase().includes(filter.toLowerCase()) ||
      w.id.toLowerCase().includes(filter.toLowerCase())
  );

  const currentWorkspace = workspaces?.find(
    (w) => w.id === connectionConfig.workspaceId
  );

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button className="flex items-center gap-1.5 rounded-md bg-secondary px-2 py-1 text-xs font-medium text-secondary-foreground transition-colors hover:bg-secondary/80">
          <span>{currentWorkspace?.name || connectionConfig.workspaceId || "_default"}</span>
          <ChevronDown className="h-3 w-3 opacity-50" />
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="z-50 min-w-[200px] overflow-hidden rounded-md border bg-card p-1 shadow-md animate-in fade-in-80"
          sideOffset={5}
          align="end"
        >
          <div className="px-2 py-1.5">
            <input
              type="text"
              placeholder="Filter workspaces..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="flex h-7 w-full rounded-md border border-input bg-background px-2 text-xs shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
          </div>

          <DropdownMenu.Separator className="my-1 h-px bg-border" />

          <div className="max-h-[300px] overflow-y-auto">
            {isLoading ? (
              <div className="px-2 py-6 text-center text-xs text-muted-foreground">
                Loading...
              </div>
            ) : filteredWorkspaces && filteredWorkspaces.length > 0 ? (
              filteredWorkspaces.map((workspace) => (
                <DropdownMenu.Item
                  key={workspace.id}
                  className="relative flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-xs outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                  onSelect={() => {
                    setConnectionConfig({
                      ...connectionConfig,
                      workspaceId: workspace.id,
                    });
                    setFilter("");
                  }}
                >
                  <div className="flex h-4 w-4 items-center justify-center">
                    {workspace.id === connectionConfig.workspaceId && (
                      <Check className="h-3.5 w-3.5" />
                    )}
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="font-medium">{workspace.name}</span>
                    {workspace.name !== workspace.id && (
                      <span className="text-[10px] text-muted-foreground">
                        {workspace.id}
                      </span>
                    )}
                  </div>
                </DropdownMenu.Item>
              ))
            ) : (
              <div className="px-2 py-6 text-center text-xs text-muted-foreground">
                No workspaces found
              </div>
            )}
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
