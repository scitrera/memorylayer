'use client';

import { Search, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/cn';
import { RecallMode } from '@/types';

const MODES = [
  { value: RecallMode.RAG, label: 'RAG' },
  { value: RecallMode.LLM, label: 'LLM' },
  { value: RecallMode.HYBRID, label: 'Hybrid' },
] as const;

interface SearchBarProps {
  query: string;
  mode: string;
  onQueryChange: (query: string) => void;
  onModeChange: (mode: string) => void;
}

export function SearchBar({ query, mode, onQueryChange, onModeChange }: SearchBarProps) {
  return (
    <div className="space-y-3">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search your memories..."
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          className="h-14 pl-12 pr-12 text-lg rounded-2xl border-slate-200 bg-white shadow-sm focus-visible:ring-brand-500"
        />
        {query.length > 0 && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 text-muted-foreground hover:text-foreground"
            onClick={() => onQueryChange('')}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Clear search</span>
          </Button>
        )}
      </div>
      <div className="flex items-center gap-1.5 justify-center">
        <span className="text-xs text-muted-foreground mr-1">Mode:</span>
        {MODES.map((m) => (
          <button
            key={m.value}
            onClick={() => onModeChange(m.value)}
            className={cn(
              'rounded-full px-3 py-1 text-xs font-medium transition-colors',
              mode === m.value
                ? 'bg-primary text-primary-foreground'
                : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
            )}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}
