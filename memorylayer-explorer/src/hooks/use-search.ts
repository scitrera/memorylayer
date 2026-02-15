'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useConnection } from '@/providers/connection-provider';
import type { RecallResult, RecallOptions } from '@/types';

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

export interface SearchOptions {
  mode?: string;
  types?: string[];
  subtypes?: string[];
  tags?: string[];
  minRelevance?: number;
  detailLevel?: 'abstract' | 'overview' | 'full';
  includeAssociations?: boolean;
  traverseDepth?: number;
  limit?: number;
}

export function useSearch(query: string, options: SearchOptions = {}) {
  const { client, isConnected } = useConnection();
  const debouncedQuery = useDebounce(query, 300);

  const recallOptions: RecallOptions = {
    mode: options.mode,
    types: options.types?.length ? options.types : undefined,
    subtypes: options.subtypes?.length ? options.subtypes : undefined,
    tags: options.tags?.length ? options.tags : undefined,
    minRelevance: options.minRelevance,
    detailLevel: options.detailLevel,
    includeAssociations: options.includeAssociations,
    traverseDepth: options.traverseDepth,
    limit: options.limit ?? 20,
  };

  return useQuery<RecallResult>({
    queryKey: ['search', debouncedQuery, recallOptions],
    queryFn: () => client.recall(debouncedQuery, recallOptions),
    enabled: isConnected && debouncedQuery.length > 0,
  });
}

export { useDebounce };
