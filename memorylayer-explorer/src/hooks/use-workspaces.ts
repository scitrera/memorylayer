"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import type { Workspace, WorkspaceSchema } from "@/types";

export function useWorkspaceList() {
  const { client, isConnected } = useConnection();

  return useQuery<Workspace[]>({
    queryKey: ["workspaces"],
    queryFn: () => client.listWorkspaces(),
    enabled: isConnected,
  });
}

export function useWorkspace(workspaceId?: string) {
  const { client, isConnected, connectionConfig } = useConnection();
  const id = workspaceId ?? connectionConfig.workspaceId;

  return useQuery<Workspace>({
    queryKey: ["workspaces", id],
    queryFn: () => client.getWorkspace(id),
    enabled: isConnected && !!id,
  });
}

export function useWorkspaceSchema(workspaceId?: string) {
  const { client, isConnected, connectionConfig } = useConnection();
  const id = workspaceId ?? connectionConfig.workspaceId;

  return useQuery<WorkspaceSchema>({
    queryKey: ["workspaces", id, "schema"],
    queryFn: () => client.getWorkspaceSchema(id),
    enabled: isConnected && !!id,
  });
}

export function useCreateWorkspace() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      name,
      settings,
    }: {
      name: string;
      settings?: Record<string, unknown>;
    }) => client.createWorkspace(name, settings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}
