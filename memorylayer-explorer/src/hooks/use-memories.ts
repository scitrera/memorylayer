"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import type { Memory, RecallResult, RecallOptions, RememberOptions, Association } from "@/types";

interface MemoryListParams {
  types?: string[];
  subtypes?: string[];
  tags?: string[];
  limit?: number;
  offset?: number;
  sortField?: string;
  sortOrder?: "asc" | "desc";
}

export function useMemoryList(params: MemoryListParams = {}) {
  const { client, isConnected } = useConnection();
  const { types, subtypes, tags, limit = 50, offset = 0 } = params;

  return useQuery<RecallResult>({
    queryKey: ["memories", { types, subtypes, tags, limit, offset }],
    queryFn: () =>
      client.recall("*", {
        types,
        subtypes,
        tags,
        limit,
        detailLevel: "overview",
      }),
    enabled: isConnected,
  });
}

export function useMemory(id: string | undefined) {
  const { client, isConnected } = useConnection();

  return useQuery<Memory>({
    queryKey: ["memories", id],
    queryFn: () => client.getMemory(id!),
    enabled: isConnected && !!id,
  });
}

export function useMemoryAssociations(memoryId: string | undefined) {
  const { client, isConnected } = useConnection();

  return useQuery<Association[]>({
    queryKey: ["memories", memoryId, "associations"],
    queryFn: () => client.getAssociations(memoryId!),
    enabled: isConnected && !!memoryId,
  });
}

export function useRemember() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      content,
      options,
    }: {
      content: string;
      options?: RememberOptions;
    }) => client.remember(content, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
}

export function useUpdateMemory() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      updates,
    }: {
      id: string;
      updates: Partial<RememberOptions> & { content?: string };
    }) => client.updateMemory(id, updates),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["memories", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
}

export function useForget() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, hard = false }: { id: string; hard?: boolean }) =>
      client.forget(id, hard),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
}

export function useDecay() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, rate = 0.1 }: { id: string; rate?: number }) =>
      client.decay(id, rate),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["memories", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });
}
