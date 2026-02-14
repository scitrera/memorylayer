'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useConnection } from '@/providers/connection-provider';
import type { Session, SessionCreateOptions, CommitOptions, CommitResponse } from '@/types';

export function useListSessions() {
  const { client, isConnected, connectionConfig } = useConnection();
  return useQuery<Session[]>({
    queryKey: ['sessions', { workspaceId: connectionConfig.workspaceId }],
    queryFn: () => client.listSessions(),
    enabled: isConnected,
  });
}

export function useSession(sessionId: string) {
  const { client } = useConnection();
  return useQuery({
    queryKey: ['sessions', sessionId],
    queryFn: () => client.getSession(sessionId),
    enabled: !!sessionId,
    retry: false,
  });
}

export function useWorkingMemory(sessionId: string) {
  const { client } = useConnection();
  return useQuery({
    queryKey: ['sessions', sessionId, 'memory'],
    queryFn: () => client.getWorkingMemory(sessionId),
    enabled: !!sessionId,
  });
}

export function useCreateSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (options: SessionCreateOptions) => {
      const response = await client.createSession(options, false);
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
}

export function useCommitSession() {
  const { client } = useConnection();

  return useMutation({
    mutationFn: async ({
      sessionId,
      options,
    }: {
      sessionId: string;
      options?: CommitOptions;
    }): Promise<CommitResponse> => {
      return client.commitSession(sessionId, options);
    },
  });
}

export function useTouchSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      sessionId,
      ttlSeconds,
    }: {
      sessionId: string;
      ttlSeconds?: number;
    }) => {
      return client.touchSession(sessionId, ttlSeconds);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['sessions', data.id], data);
    },
  });
}

export function useDeleteSession() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      await client.deleteSession(sessionId);
      return sessionId;
    },
    onSuccess: (sessionId) => {
      queryClient.removeQueries({ queryKey: ['sessions', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
}

export function useSetWorkingMemory() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      sessionId,
      key,
      value,
    }: {
      sessionId: string;
      key: string;
      value: unknown;
    }) => {
      await client.setWorkingMemory(sessionId, key, value);
      return { sessionId, key, value };
    },
    onSuccess: ({ sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['sessions', sessionId, 'memory'] });
    },
  });
}
