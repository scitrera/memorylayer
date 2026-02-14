'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/cn';
import {
  MemoryType,
  MemorySubtype,
  DetailLevel,
} from '@/types';
import {
  MEMORY_TYPE_LABELS,
  MEMORY_TYPE_COLORS,
  MEMORY_SUBTYPE_LABELS,
} from '@/lib/constants';

export interface SearchFilterValues {
  types: string[];
  subtypes: string[];
  tags: string[];
  minRelevance: number;
  detailLevel: string;
  includeAssociations: boolean;
  traverseDepth: number;
}

interface SearchFiltersProps {
  filters: SearchFilterValues;
  onFiltersChange: (filters: SearchFilterValues) => void;
}

const DETAIL_LEVELS = [
  { value: DetailLevel.ABSTRACT, label: 'Abstract' },
  { value: DetailLevel.OVERVIEW, label: 'Overview' },
  { value: DetailLevel.FULL, label: 'Full' },
] as const;

export function SearchFilters({ filters, onFiltersChange }: SearchFiltersProps) {
  const [expanded, setExpanded] = useState(false);
  const [tagInput, setTagInput] = useState('');

  const update = (partial: Partial<SearchFilterValues>) => {
    onFiltersChange({ ...filters, ...partial });
  };

  const toggleType = (type: string) => {
    const types = filters.types.includes(type)
      ? filters.types.filter((t) => t !== type)
      : [...filters.types, type];
    update({ types });
  };

  const toggleSubtype = (subtype: string) => {
    const subtypes = filters.subtypes.includes(subtype)
      ? filters.subtypes.filter((s) => s !== subtype)
      : [...filters.subtypes, subtype];
    update({ subtypes });
  };

  const addTag = () => {
    const tag = tagInput.trim();
    if (tag && !filters.tags.includes(tag)) {
      update({ tags: [...filters.tags, tag] });
    }
    setTagInput('');
  };

  const removeTag = (tag: string) => {
    update({ tags: filters.tags.filter((t) => t !== tag) });
  };

  const activeFilterCount =
    filters.types.length +
    filters.subtypes.length +
    filters.tags.length +
    (filters.minRelevance > 0 ? 1 : 0) +
    (filters.detailLevel !== 'full' ? 1 : 0) +
    (filters.includeAssociations ? 1 : 0) +
    (filters.traverseDepth > 0 ? 1 : 0);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white">
      <Button
        variant="ghost"
        className="flex w-full items-center justify-between px-4 py-3 hover:bg-transparent"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Filters</span>
          {activeFilterCount > 0 && (
            <Badge variant="secondary" className="text-xs">
              {activeFilterCount}
            </Badge>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </Button>

      {expanded && (
        <div className="space-y-5 border-t border-slate-100 px-4 pb-4 pt-3">
          {/* Memory Types */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Memory Types
            </label>
            <div className="flex flex-wrap gap-2">
              {Object.values(MemoryType).map((type) => {
                const colors = MEMORY_TYPE_COLORS[type];
                const selected = filters.types.includes(type);
                return (
                  <button
                    key={type}
                    onClick={() => toggleType(type)}
                    className={cn(
                      'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                      selected
                        ? `${colors.bg} ${colors.text} ${colors.border}`
                        : 'border-slate-200 text-muted-foreground hover:border-slate-300'
                    )}
                  >
                    {MEMORY_TYPE_LABELS[type] ?? type}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Subtypes */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Subtypes
            </label>
            <div className="flex flex-wrap gap-2">
              {Object.values(MemorySubtype).map((subtype) => {
                const selected = filters.subtypes.includes(subtype);
                return (
                  <button
                    key={subtype}
                    onClick={() => toggleSubtype(subtype)}
                    className={cn(
                      'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                      selected
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-slate-200 text-muted-foreground hover:border-slate-300'
                    )}
                  >
                    {MEMORY_SUBTYPE_LABELS[subtype] ?? subtype}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tags */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Tags
            </label>
            <div className="flex gap-2">
              <Input
                placeholder="Add a tag..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addTag();
                  }
                }}
                className="h-8 text-sm"
              />
              <Button size="sm" variant="outline" onClick={addTag} className="h-8">
                Add
              </Button>
            </div>
            {filters.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {filters.tags.map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    className="cursor-pointer text-xs hover:bg-destructive/10 hover:text-destructive"
                    onClick={() => removeTag(tag)}
                  >
                    {tag} &times;
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Min Relevance */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Min Relevance: {(filters.minRelevance * 100).toFixed(0)}%
            </label>
            <Slider
              value={[filters.minRelevance]}
              onValueChange={([val]) => update({ minRelevance: val })}
              min={0}
              max={1}
              step={0.05}
              className="w-full"
            />
            <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
              <span>Any</span>
              <span>Exact</span>
            </div>
          </div>

          {/* Detail Level */}
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Detail Level
            </label>
            <div className="flex gap-1.5">
              {DETAIL_LEVELS.map((dl) => (
                <button
                  key={dl.value}
                  onClick={() => update({ detailLevel: dl.value })}
                  className={cn(
                    'rounded-full px-3 py-1 text-xs font-medium transition-colors',
                    filters.detailLevel === dl.value
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                  )}
                >
                  {dl.label}
                </button>
              ))}
            </div>
          </div>

          {/* Include Associations + Traverse Depth */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Include Associations
              </label>
              <button
                onClick={() => update({ includeAssociations: !filters.includeAssociations })}
                className={cn(
                  'rounded-full px-3 py-1 text-xs font-medium transition-colors',
                  filters.includeAssociations
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                )}
              >
                {filters.includeAssociations ? 'On' : 'Off'}
              </button>
            </div>
            <div>
              <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Traverse Depth: {filters.traverseDepth}
              </label>
              <Slider
                value={[filters.traverseDepth]}
                onValueChange={([val]) => update({ traverseDepth: val })}
                min={0}
                max={5}
                step={1}
                className="w-full"
              />
              <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                <span>0</span>
                <span>5</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
