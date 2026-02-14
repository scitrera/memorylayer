'use client';

import { useCallback, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Search } from 'lucide-react';
import { SearchBar } from '@/components/search/search-bar';
import { SearchFilters, type SearchFilterValues } from '@/components/search/search-filters';
import { SearchResults } from '@/components/search/search-results';
import { useSearch } from '@/hooks/use-search';
import { RecallMode } from '@/types';

function parseSearchParams(params: URLSearchParams): {
  query: string;
  mode: string;
  filters: SearchFilterValues;
} {
  return {
    query: params.get('q') ?? '',
    mode: params.get('mode') ?? RecallMode.RAG,
    filters: {
      types: params.getAll('type'),
      subtypes: params.getAll('subtype'),
      tags: params.getAll('tag'),
      minRelevance: Number(params.get('minRelevance') ?? 0),
      detailLevel: params.get('detailLevel') ?? 'full',
      includeAssociations: params.get('includeAssociations') === 'true',
      traverseDepth: Number(params.get('traverseDepth') ?? 0),
    },
  };
}

function buildSearchParams(query: string, mode: string, filters: SearchFilterValues): string {
  const params = new URLSearchParams();
  if (query) params.set('q', query);
  if (mode !== RecallMode.RAG) params.set('mode', mode);
  filters.types.forEach((t) => params.append('type', t));
  filters.subtypes.forEach((s) => params.append('subtype', s));
  filters.tags.forEach((t) => params.append('tag', t));
  if (filters.minRelevance > 0) params.set('minRelevance', String(filters.minRelevance));
  if (filters.detailLevel !== 'full') params.set('detailLevel', filters.detailLevel);
  if (filters.includeAssociations) params.set('includeAssociations', 'true');
  if (filters.traverseDepth > 0) params.set('traverseDepth', String(filters.traverseDepth));
  return params.toString();
}

function SearchPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { query, mode, filters } = parseSearchParams(searchParams);

  const updateUrl = useCallback(
    (newQuery: string, newMode: string, newFilters: SearchFilterValues) => {
      const qs = buildSearchParams(newQuery, newMode, newFilters);
      router.replace(`/search${qs ? `?${qs}` : ''}`, { scroll: false });
    },
    [router]
  );

  const handleQueryChange = useCallback(
    (newQuery: string) => updateUrl(newQuery, mode, filters),
    [updateUrl, mode, filters]
  );

  const handleModeChange = useCallback(
    (newMode: string) => updateUrl(query, newMode, filters),
    [updateUrl, query, filters]
  );

  const handleFiltersChange = useCallback(
    (newFilters: SearchFilterValues) => updateUrl(query, mode, newFilters),
    [updateUrl, query, mode]
  );

  const { data, isLoading } = useSearch(query, {
    mode,
    types: filters.types,
    subtypes: filters.subtypes,
    tags: filters.tags,
    minRelevance: filters.minRelevance || undefined,
    detailLevel: filters.detailLevel as 'abstract' | 'overview' | 'full',
    includeAssociations: filters.includeAssociations || undefined,
    traverseDepth: filters.traverseDepth || undefined,
  });

  return (
    <div className="space-y-6">
      <div className="mx-auto max-w-3xl space-y-4 pt-4">
        <SearchBar
          query={query}
          mode={mode}
          onQueryChange={handleQueryChange}
          onModeChange={handleModeChange}
        />
        <SearchFilters filters={filters} onFiltersChange={handleFiltersChange} />
      </div>

      <div className="mx-auto max-w-3xl">
        {query.length > 0 ? (
          <SearchResults data={data} isLoading={isLoading} hasQuery={query.length > 0} />
        ) : (
          <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center">
            <Search className="mx-auto h-12 w-12 text-muted-foreground/40" />
            <h2 className="mt-4 text-lg font-medium text-foreground">Search your memories</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Enter a query to search across all your stored memories
            </p>
            <ul className="mt-6 space-y-2 text-sm text-muted-foreground text-left max-w-sm mx-auto">
              <li className="flex items-start gap-2">
                <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <span>Use natural language queries for semantic search</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <span>Switch to LLM mode for AI-powered query understanding</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <span>Use Hybrid mode for the best of both approaches</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                <span>Filter by type, tags, and relevance using Advanced Filters</span>
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen"><div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" /></div>}>
      <SearchPageInner />
    </Suspense>
  );
}
