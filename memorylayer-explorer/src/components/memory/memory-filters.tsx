"use client";

import { MemoryType, MemorySubtype } from "@/types";
import { MEMORY_TYPE_LABELS, MEMORY_SUBTYPE_LABELS } from "@/lib/constants";
import { Slider } from "@/components/ui/slider";
import type { FilterState } from "@/types";

interface MemoryFiltersProps {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
}

export function MemoryFilters({ filters, onChange }: MemoryFiltersProps) {
  function toggleType(type: string) {
    const types = filters.types.includes(type)
      ? filters.types.filter((t) => t !== type)
      : [...filters.types, type];
    onChange({ ...filters, types });
  }

  function toggleSubtype(subtype: string) {
    const subtypes = filters.subtypes.includes(subtype)
      ? filters.subtypes.filter((s) => s !== subtype)
      : [...filters.subtypes, subtype];
    onChange({ ...filters, subtypes });
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">Type</h3>
        <div className="space-y-1.5">
          {Object.values(MemoryType).map((type) => (
            <label key={type} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={filters.types.includes(type)}
                onChange={() => toggleType(type)}
                className="h-4 w-4 rounded border-slate-300"
              />
              <span className="text-foreground">
                {MEMORY_TYPE_LABELS[type] ?? type}
              </span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">Subtype</h3>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {Object.values(MemorySubtype).map((subtype) => (
            <label key={subtype} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={filters.subtypes.includes(subtype)}
                onChange={() => toggleSubtype(subtype)}
                className="h-4 w-4 rounded border-slate-300"
              />
              <span className="text-foreground">
                {MEMORY_SUBTYPE_LABELS[subtype] ?? subtype}
              </span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">
          Importance: {Math.round(filters.importanceRange[0] * 100)}% -{" "}
          {Math.round(filters.importanceRange[1] * 100)}%
        </h3>
        <Slider
          value={filters.importanceRange}
          onValueChange={(value) =>
            onChange({
              ...filters,
              importanceRange: value as [number, number],
            })
          }
          min={0}
          max={1}
          step={0.1}
        />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">Status</h3>
        <div className="flex gap-2">
          {["all", "active", "archived"].map((status) => (
            <button
              key={status}
              onClick={() => onChange({ ...filters, status })}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                filters.status === status
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={() =>
          onChange({
            types: [],
            subtypes: [],
            tags: [],
            status: "all",
            importanceRange: [0, 1],
            dateRange: {},
          })
        }
        className="text-xs text-muted-foreground hover:text-foreground"
      >
        Clear all filters
      </button>
    </div>
  );
}
