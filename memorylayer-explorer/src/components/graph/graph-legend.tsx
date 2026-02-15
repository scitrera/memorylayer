"use client";

import { memoryTypeColors } from "@/lib/colors";
import {
  RELATIONSHIP_CATEGORY_COLORS,
  RELATIONSHIP_CATEGORY_LABELS,
  MEMORY_TYPE_LABELS,
} from "@/lib/constants";
import { RelationshipCategory, MemoryType } from "@/types";

export function GraphLegend() {
  const memTypes = Object.values(MemoryType);
  const categories = Object.values(RelationshipCategory);

  return (
    <div className="absolute bottom-4 left-4 z-10 rounded-xl bg-white/80 backdrop-blur-xl border border-slate-200 shadow-lg p-3 max-w-[200px]">
      {/* Memory types */}
      <p className="text-xs font-semibold text-slate-600 mb-1.5">Memory Types</p>
      <div className="flex flex-col gap-1 mb-3">
        {memTypes.map((type) => (
          <div key={type} className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ backgroundColor: memoryTypeColors[type] ?? "#94a3b8" }}
            />
            <span className="text-xs text-slate-600">
              {MEMORY_TYPE_LABELS[type] ?? type}
            </span>
          </div>
        ))}
      </div>

      {/* Relationship categories */}
      <p className="text-xs font-semibold text-slate-600 mb-1.5">Relationships</p>
      <div className="flex flex-col gap-1">
        {categories.map((cat) => {
          const colors = RELATIONSHIP_CATEGORY_COLORS[cat];
          return (
            <div key={cat} className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: colors?.stroke ?? "#94a3b8" }}
              />
              <span className="text-xs text-slate-600">
                {RELATIONSHIP_CATEGORY_LABELS[cat] ?? cat}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
