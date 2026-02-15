"use client";

import { GitBranch, Workflow, Target, RotateCcw, ArrowDown, ArrowRight, ArrowLeftRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/cn";
import {
  RELATIONSHIP_CATEGORY_COLORS,
  RELATIONSHIP_CATEGORY_LABELS,
} from "@/lib/constants";
import { RelationshipCategory } from "@/types";

export type LayoutType = "dagre" | "force" | "radial";
export type DirectionType = "outgoing" | "incoming" | "both";

interface GraphControlsProps {
  layout: LayoutType;
  onLayoutChange: (layout: LayoutType) => void;
  depth: number;
  onDepthChange: (depth: number) => void;
  direction: DirectionType;
  onDirectionChange: (dir: DirectionType) => void;
  minStrength: number;
  onMinStrengthChange: (val: number) => void;
  enabledCategories: Set<string>;
  onCategoryToggle: (category: string) => void;
  onReset: () => void;
}

const layouts: { key: LayoutType; icon: typeof GitBranch; label: string }[] = [
  { key: "dagre", icon: GitBranch, label: "Hierarchy" },
  { key: "force", icon: Workflow, label: "Force" },
  { key: "radial", icon: Target, label: "Radial" },
];

const directions: { key: DirectionType; icon: typeof ArrowDown; label: string }[] = [
  { key: "outgoing", icon: ArrowRight, label: "Out" },
  { key: "incoming", icon: ArrowDown, label: "In" },
  { key: "both", icon: ArrowLeftRight, label: "Both" },
];

export function GraphControls({
  layout,
  onLayoutChange,
  depth,
  onDepthChange,
  direction,
  onDirectionChange,
  minStrength,
  onMinStrengthChange,
  enabledCategories,
  onCategoryToggle,
  onReset,
}: GraphControlsProps) {
  const categories = Object.values(RelationshipCategory);

  return (
    <div className="absolute top-4 left-4 z-10 w-[220px] flex flex-col gap-3 rounded-xl bg-white/80 backdrop-blur-xl border border-slate-200 shadow-lg p-4">
      {/* Layout selector */}
      <div>
        <p className="text-xs font-medium text-slate-500 mb-1.5">Layout</p>
        <div className="flex gap-1">
          {layouts.map(({ key, icon: Icon, label }) => (
            <Button
              key={key}
              variant={layout === key ? "default" : "outline"}
              size="sm"
              className="flex-1 text-xs h-8"
              onClick={() => onLayoutChange(key)}
              title={label}
            >
              <Icon className="w-3.5 h-3.5" />
            </Button>
          ))}
        </div>
      </div>

      {/* Depth slider */}
      <div>
        <p className="text-xs font-medium text-slate-500 mb-1.5">
          Depth: {depth}
        </p>
        <Slider
          value={[depth]}
          min={1}
          max={5}
          step={1}
          onValueChange={([v]) => onDepthChange(v)}
        />
      </div>

      {/* Direction */}
      <div>
        <p className="text-xs font-medium text-slate-500 mb-1.5">Direction</p>
        <div className="flex gap-1">
          {directions.map(({ key, icon: Icon, label }) => (
            <Button
              key={key}
              variant={direction === key ? "default" : "outline"}
              size="sm"
              className="flex-1 text-xs h-8"
              onClick={() => onDirectionChange(key)}
              title={label}
            >
              <Icon className="w-3.5 h-3.5" />
            </Button>
          ))}
        </div>
      </div>

      {/* Min strength */}
      <div>
        <p className="text-xs font-medium text-slate-500 mb-1.5">
          Min Strength: {Math.round(minStrength * 100)}%
        </p>
        <Slider
          value={[minStrength]}
          min={0}
          max={1}
          step={0.05}
          onValueChange={([v]) => onMinStrengthChange(v)}
        />
      </div>

      {/* Category filters */}
      <div>
        <p className="text-xs font-medium text-slate-500 mb-1.5">Categories</p>
        <div className="flex flex-col gap-1">
          {categories.map((cat) => {
            const colors = RELATIONSHIP_CATEGORY_COLORS[cat];
            const label = RELATIONSHIP_CATEGORY_LABELS[cat] ?? cat;
            const enabled = enabledCategories.has(cat);
            return (
              <label
                key={cat}
                className="flex items-center gap-2 cursor-pointer text-xs text-slate-600 hover:text-slate-900"
              >
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={() => onCategoryToggle(cat)}
                  className="rounded border-slate-300"
                />
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: colors?.stroke ?? "#94a3b8" }}
                />
                {label}
              </label>
            );
          })}
        </div>
      </div>

      {/* Reset */}
      <Button
        variant="outline"
        size="sm"
        className="w-full text-xs"
        onClick={onReset}
      >
        <RotateCcw className="w-3.5 h-3.5 mr-1" />
        Reset Filters
      </Button>
    </div>
  );
}
