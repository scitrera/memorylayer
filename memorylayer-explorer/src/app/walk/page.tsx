'use client';

import { useState, useCallback, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import {
  Search,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  ArrowDownLeft,
  Loader2,
} from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/cn';
import {
  truncateContent,
  formatRelativeTime,
  formatImportance,
  formatStrength,
} from '@/lib/format';
import {
  MEMORY_TYPE_COLORS,
  MEMORY_TYPE_LABELS,
  RELATIONSHIP_TO_CATEGORY,
  RELATIONSHIP_CATEGORY_COLORS,
  RELATIONSHIP_CATEGORY_LABELS,
  getImportanceColor,
} from '@/lib/constants';
import { useConnection } from '@/providers/connection-provider';
import { useSearch } from '@/hooks/use-search';
import type { Memory, Association } from '@/types';

interface WalkStep {
  memoryId: string;
  relationship?: string;
  direction?: 'outgoing' | 'incoming';
}

interface NeighborInfo {
  association: Association;
  memory: Memory;
  direction: 'outgoing' | 'incoming';
}

// --- Memory Picker Component ---

interface MemoryPickerProps {
  onSelect: (memory: Memory) => void;
  initialMemoryId?: string;
}

function MemoryPicker({ onSelect, initialMemoryId }: MemoryPickerProps) {
  const [pickerQuery, setPickerQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const { client, isConnected } = useConnection();

  const { data: searchResults, isLoading: searchLoading } = useSearch(pickerQuery, {
    limit: 8,
    detailLevel: 'abstract',
  });

  // Auto-load initial memory if provided
  const { data: initialMemory } = useQuery<Memory>({
    queryKey: ['memory', initialMemoryId],
    queryFn: () => client.getMemory(initialMemoryId!),
    enabled: isConnected && !!initialMemoryId,
  });

  useEffect(() => {
    if (initialMemory) {
      onSelect(initialMemory);
    }
  }, [initialMemory, onSelect]);

  return (
    <div className="relative">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search for a starting memory..."
          value={pickerQuery}
          onChange={(e) => {
            setPickerQuery(e.target.value);
            setShowDropdown(true);
          }}
          onFocus={() => setShowDropdown(true)}
          onBlur={() => {
            // Delay to allow click on dropdown items
            setTimeout(() => setShowDropdown(false), 200);
          }}
          className="pl-10"
        />
      </div>

      {showDropdown && pickerQuery.length > 0 && (
        <div className="absolute z-50 mt-1 w-full rounded-xl border border-slate-200 bg-white shadow-lg">
          {searchLoading ? (
            <div className="space-y-2 p-3">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : searchResults && searchResults.memories.length > 0 ? (
            <ScrollArea className="max-h-64">
              <div className="p-1">
                {searchResults.memories.map((mem) => {
                  const typeColors = MEMORY_TYPE_COLORS[mem.type];
                  return (
                    <button
                      key={mem.id}
                      className="flex w-full items-start gap-2 rounded-lg px-3 py-2 text-left hover:bg-slate-50"
                      onMouseDown={(e) => {
                        e.preventDefault();
                        onSelect(mem);
                        setPickerQuery('');
                        setShowDropdown(false);
                      }}
                    >
                      <Badge
                        variant="outline"
                        className={cn(
                          'mt-0.5 text-[10px] flex-shrink-0',
                          typeColors?.bg,
                          typeColors?.text,
                          typeColors?.border
                        )}
                      >
                        {MEMORY_TYPE_LABELS[mem.type] ?? mem.type}
                      </Badge>
                      <span className="text-sm line-clamp-2">
                        {truncateContent(mem.content, 120)}
                      </span>
                    </button>
                  );
                })}
              </div>
            </ScrollArea>
          ) : (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No memories found
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// --- Traversal Breadcrumbs ---

interface TraversalBreadcrumbsProps {
  history: WalkStep[];
  memories: Map<string, Memory>;
  currentIndex: number;
  onJumpTo: (index: number) => void;
}

function TraversalBreadcrumbs({
  history,
  memories,
  currentIndex,
  onJumpTo,
}: TraversalBreadcrumbsProps) {
  return (
    <div className="flex flex-wrap items-center gap-1">
      {history.slice(0, currentIndex + 1).map((step, idx) => {
        const mem = memories.get(step.memoryId);
        const category = step.relationship
          ? RELATIONSHIP_TO_CATEGORY[step.relationship]
          : undefined;
        const categoryColors = category
          ? RELATIONSHIP_CATEGORY_COLORS[category]
          : undefined;

        return (
          <div key={idx} className="flex items-center gap-1">
            {idx > 0 && step.relationship && (
              <Badge
                variant="outline"
                className={cn(
                  'text-[10px] font-normal',
                  categoryColors?.bg,
                  categoryColors?.text
                )}
              >
                {step.relationship.replace(/_/g, ' ')}
              </Badge>
            )}
            {idx > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
            <button
              onClick={() => onJumpTo(idx)}
              className={cn(
                'max-w-[180px] truncate rounded-full border px-2.5 py-0.5 text-xs transition-colors',
                idx === currentIndex
                  ? 'border-primary bg-primary/10 text-primary font-medium'
                  : 'border-slate-200 text-muted-foreground hover:border-slate-300 hover:text-foreground'
              )}
            >
              {mem ? truncateContent(mem.content, 30) : step.memoryId.slice(0, 12)}
            </button>
          </div>
        );
      })}
    </div>
  );
}

// --- Current Memory Detail ---

interface CurrentMemoryDetailProps {
  memory: Memory;
}

function CurrentMemoryDetail({ memory }: CurrentMemoryDetailProps) {
  const typeColors = MEMORY_TYPE_COLORS[memory.type] ?? {
    bg: 'bg-slate-50',
    text: 'text-slate-700',
    border: 'border-slate-200',
  };

  return (
    <Card className="rounded-2xl border-slate-200">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={cn('text-xs', typeColors.bg, typeColors.text, typeColors.border)}
            >
              {MEMORY_TYPE_LABELS[memory.type] ?? memory.type}
            </Badge>
            {memory.subtype && (
              <Badge variant="outline" className="text-xs">
                {memory.subtype}
              </Badge>
            )}
          </div>
          <Link
            href={`/memories/${memory.id}`}
            className="text-xs text-primary hover:underline"
          >
            Full Detail
          </Link>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{memory.content}</p>

        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          <span>
            Importance:{' '}
            <span className={cn('font-medium', getImportanceColor(memory.importance))}>
              {formatImportance(memory.importance)}
            </span>
          </span>
          <span>Created: {formatRelativeTime(memory.created_at)}</span>
          <span>Accessed: {memory.access_count}x</span>
        </div>

        {memory.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {memory.tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[10px] font-normal">
                {tag}
              </Badge>
            ))}
          </div>
        )}

        {memory.metadata && Object.keys(memory.metadata).length > 0 && (
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
              Metadata ({Object.keys(memory.metadata).length} keys)
            </summary>
            <pre className="mt-1 overflow-x-auto rounded bg-slate-50 p-2 text-[10px]">
              {JSON.stringify(memory.metadata, null, 2)}
            </pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

// --- Neighbor List ---

interface NeighborListProps {
  neighbors: NeighborInfo[];
  isLoading: boolean;
  onNavigate: (neighbor: NeighborInfo) => void;
}

function NeighborList({ neighbors, isLoading, onNavigate }: NeighborListProps) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-20 w-full rounded-xl" />
        ))}
      </div>
    );
  }

  if (neighbors.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 p-6 text-center text-sm text-muted-foreground">
        No relationships found for this memory
      </div>
    );
  }

  const outgoing = neighbors.filter((n) => n.direction === 'outgoing');
  const incoming = neighbors.filter((n) => n.direction === 'incoming');

  // Group by relationship type within each direction
  const groupByRelationship = (items: NeighborInfo[]) => {
    const groups = new Map<string, NeighborInfo[]>();
    for (const item of items) {
      const rel = item.association.relationship;
      const existing = groups.get(rel) ?? [];
      existing.push(item);
      groups.set(rel, existing);
    }
    return groups;
  };

  const renderGroup = (
    title: string,
    icon: React.ReactNode,
    items: NeighborInfo[]
  ) => {
    if (items.length === 0) return null;
    const groups = groupByRelationship(items);

    return (
      <div className="space-y-2">
        <h3 className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
          {icon}
          {title}
          <Badge variant="secondary" className="text-[10px] ml-1">
            {items.length}
          </Badge>
        </h3>
        {Array.from(groups.entries()).map(([relationship, groupItems]) => {
          const category = RELATIONSHIP_TO_CATEGORY[relationship];
          const categoryColors = category
            ? RELATIONSHIP_CATEGORY_COLORS[category]
            : undefined;
          const categoryLabel = category
            ? RELATIONSHIP_CATEGORY_LABELS[category]
            : undefined;

          return (
            <div key={relationship} className="space-y-1">
              <div className="flex items-center gap-1.5">
                <Badge
                  variant="outline"
                  className={cn(
                    'text-[10px]',
                    categoryColors?.bg,
                    categoryColors?.text
                  )}
                >
                  {relationship.replace(/_/g, ' ')}
                </Badge>
                {categoryLabel && (
                  <span className="text-[10px] text-muted-foreground">
                    ({categoryLabel})
                  </span>
                )}
              </div>
              {groupItems.map((neighbor) => {
                const mem = neighbor.memory;
                const typeColors = MEMORY_TYPE_COLORS[mem.type];
                return (
                  <button
                    key={neighbor.association.id}
                    onClick={() => onNavigate(neighbor)}
                    className="flex w-full items-start gap-2 rounded-xl border border-slate-200 bg-white p-3 text-left transition-colors hover:border-slate-300 hover:shadow-sm"
                  >
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className={cn(
                            'text-[10px] flex-shrink-0',
                            typeColors?.bg,
                            typeColors?.text,
                            typeColors?.border
                          )}
                        >
                          {MEMORY_TYPE_LABELS[mem.type] ?? mem.type}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground">
                          Strength: {formatStrength(neighbor.association.strength)}
                        </span>
                      </div>
                      <p className="text-xs text-foreground line-clamp-2">
                        {truncateContent(mem.content, 100)}
                      </p>
                    </div>
                    <ChevronRight className="h-4 w-4 flex-shrink-0 text-muted-foreground mt-1" />
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {renderGroup(
        'Outgoing Relationships',
        <ArrowUpRight className="h-3 w-3" />,
        outgoing
      )}
      {renderGroup(
        'Incoming Relationships',
        <ArrowDownLeft className="h-3 w-3" />,
        incoming
      )}
    </div>
  );
}

// --- Main Walk Page ---

function WalkPageInner() {
  const searchParams = useSearchParams();
  const initialFromId = searchParams.get('from') ?? undefined;

  const { client, isConnected } = useConnection();

  const [history, setHistory] = useState<WalkStep[]>([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [memoryCache, setMemoryCache] = useState<Map<string, Memory>>(new Map());

  const currentStep = currentIndex >= 0 ? history[currentIndex] : undefined;
  const currentMemory = currentStep ? memoryCache.get(currentStep.memoryId) : undefined;

  // Fetch associations for current memory
  const {
    data: associations,
    isLoading: associationsLoading,
  } = useQuery<Association[]>({
    queryKey: ['associations', currentStep?.memoryId],
    queryFn: () => client.getAssociations(currentStep!.memoryId, 'both'),
    enabled: isConnected && !!currentStep?.memoryId,
  });

  // Resolve neighbor memories
  const [neighbors, setNeighbors] = useState<NeighborInfo[]>([]);
  const [neighborsLoading, setNeighborsLoading] = useState(false);

  useEffect(() => {
    if (!associations || !currentStep) {
      setNeighbors([]);
      return;
    }

    let cancelled = false;
    setNeighborsLoading(true);

    const resolve = async () => {
      const results: NeighborInfo[] = [];

      for (const assoc of associations) {
        const isOutgoing = assoc.source_id === currentStep.memoryId;
        const targetId = isOutgoing ? assoc.target_id : assoc.source_id;

        // Check cache first
        let mem = memoryCache.get(targetId);
        if (!mem) {
          try {
            mem = await client.getMemory(targetId);
            if (!cancelled) {
              setMemoryCache((prev) => new Map(prev).set(targetId, mem!));
            }
          } catch {
            continue; // Skip memories we can't fetch
          }
        }

        if (!cancelled) {
          results.push({
            association: assoc,
            memory: mem,
            direction: isOutgoing ? 'outgoing' : 'incoming',
          });
        }
      }

      if (!cancelled) {
        setNeighbors(results);
        setNeighborsLoading(false);
      }
    };

    resolve();
    return () => {
      cancelled = true;
    };
  }, [associations, currentStep, client, memoryCache]);

  const selectMemory = useCallback(
    (memory: Memory) => {
      setMemoryCache((prev) => new Map(prev).set(memory.id, memory));
      const newStep: WalkStep = { memoryId: memory.id };
      setHistory([newStep]);
      setCurrentIndex(0);
    },
    []
  );

  const navigateToNeighbor = useCallback(
    (neighbor: NeighborInfo) => {
      const newStep: WalkStep = {
        memoryId: neighbor.memory.id,
        relationship: neighbor.association.relationship,
        direction: neighbor.direction,
      };

      // Truncate forward history if we navigated back then forward
      const newHistory = [...history.slice(0, currentIndex + 1), newStep];
      setHistory(newHistory);
      setCurrentIndex(newHistory.length - 1);
    },
    [history, currentIndex]
  );

  const goBack = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  }, [currentIndex]);

  const goForward = useCallback(() => {
    if (currentIndex < history.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  }, [currentIndex, history.length]);

  const jumpTo = useCallback((index: number) => {
    setCurrentIndex(index);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Relationship Walker</h1>
        <p className="text-muted-foreground">
          Traverse relationships between memories step by step
        </p>
      </div>

      {/* Memory Picker */}
      <div className="mx-auto max-w-xl">
        <MemoryPicker onSelect={selectMemory} initialMemoryId={initialFromId} />
      </div>

      {/* Walk Interface */}
      {currentMemory && (
        <div className="space-y-4">
          {/* Navigation Bar */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              disabled={currentIndex <= 0}
              onClick={goBack}
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Back</span>
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              disabled={currentIndex >= history.length - 1}
              onClick={goForward}
            >
              <ArrowRight className="h-4 w-4" />
              <span className="sr-only">Forward</span>
            </Button>
            <div className="flex-1 overflow-x-auto">
              <TraversalBreadcrumbs
                history={history}
                memories={memoryCache}
                currentIndex={currentIndex}
                onJumpTo={jumpTo}
              />
            </div>
          </div>

          {/* Two Column Layout */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Left: Current Memory */}
            <div>
              <CurrentMemoryDetail memory={currentMemory} />
            </div>

            {/* Right: Neighbors */}
            <div>
              <Card className="rounded-2xl border-slate-200">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Connections
                    {(associationsLoading || neighborsLoading) && (
                      <Loader2 className="ml-2 inline h-3 w-3 animate-spin" />
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <NeighborList
                    neighbors={neighbors}
                    isLoading={associationsLoading || neighborsLoading}
                    onNavigate={navigateToNeighbor}
                  />
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!currentMemory && !initialFromId && (
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center">
          <Search className="mx-auto h-12 w-12 text-muted-foreground/40" />
          <h2 className="mt-4 text-lg font-medium text-foreground">
            Start exploring relationships
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Search for a memory above to begin walking its relationship graph
          </p>
          <p className="mt-4 text-xs text-muted-foreground">
            You can also navigate here from a memory detail page or search result
          </p>
        </div>
      )}
    </div>
  );
}

export default function WalkPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-[calc(100vh-4rem)]"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>}>
      <WalkPageInner />
    </Suspense>
  );
}
