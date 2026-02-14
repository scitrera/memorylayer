'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useConnection } from '@/providers/connection-provider';
import { useSyncExternalStore, useCallback } from 'react';
import type { SessionCreateOptions, CommitOptions, CommitResponse } from '@/types';

const STORAGE_KEY = 'memorylayer-sessions';

// Cache for useSyncExternalStore: must return the same reference
// if the underlying data hasn't changed to avoid infinite re-renders.
let _cachedRaw: string | null = null;
let _cachedIds: string[] = [];

function getSessionIds(): string[] {
  if (typeof window === 'undefined') return _cachedIds;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw !== _cachedRaw) {
      _cachedRaw = raw;
      _cachedIds = raw ? (JSON.parse(raw) as string[]) : [];
    }
  } catch {
    // ignore parse errors
  }
  return _cachedIds;
}

function setSessionIds(ids: string[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    // ignore storage errors
  }
  window.dispatchEvent(new StorageEvent('storage', { key: STORAGE_KEY }));
}

function addSessionId(id: string): void {
  const ids = getSessionIds();
  if (!ids.includes(id)) {
    setSessionIds([...ids, id]);
  }
}

function removeSessionId(id: string): void {
  const ids = getSessionIds().filter((sid) => sid !== id);
  setSessionIds(ids);
}

function subscribeToStorage(callback: () => void): () => void {
  const handler = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY || e.key === null) {
      callback();
    }
  };
  window.addEventListener('storage', handler);
  return () => window.removeEventListener('storage', handler);
}

const serverSnapshot: string[] = [];

export function useSessionIds(): string[] {
  return useSyncExternalStore(
    subscribeToStorage,
    getSessionIds,
    () => serverSnapshot
  );
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
    onSuccess: (data) => {
      addSessionId(data.session.id);
      queryClient.setQueryData(['sessions', data.session.id], data.session);
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
      removeSessionId(sessionId);
      queryClient.removeQueries({ queryKey: ['sessions', sessionId] });
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

export function useRemoveSessionId() {
  return useCallback((id: string) => {
    removeSessionId(id);
  }, []);
}
