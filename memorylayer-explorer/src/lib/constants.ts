import { MemoryType, MemorySubtype, RelationshipCategory } from "@scitrera/memorylayer-sdk";

export const MEMORY_TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  [MemoryType.EPISODIC]: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200" },
  [MemoryType.SEMANTIC]: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  [MemoryType.PROCEDURAL]: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200" },
  [MemoryType.WORKING]: { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200" },
};

export const MEMORY_TYPE_LABELS: Record<string, string> = {
  [MemoryType.EPISODIC]: "Episodic",
  [MemoryType.SEMANTIC]: "Semantic",
  [MemoryType.PROCEDURAL]: "Procedural",
  [MemoryType.WORKING]: "Working",
};

export const MEMORY_SUBTYPE_LABELS: Record<string, string> = {
  [MemorySubtype.SOLUTION]: "Solution",
  [MemorySubtype.PROBLEM]: "Problem",
  [MemorySubtype.CODE_PATTERN]: "Code Pattern",
  [MemorySubtype.FIX]: "Fix",
  [MemorySubtype.ERROR]: "Error",
  [MemorySubtype.WORKFLOW]: "Workflow",
  [MemorySubtype.PREFERENCE]: "Preference",
  [MemorySubtype.DECISION]: "Decision",
  [MemorySubtype.PROFILE]: "Profile",
  [MemorySubtype.ENTITY]: "Entity",
  [MemorySubtype.EVENT]: "Event",
  [MemorySubtype.DIRECTIVE]: "Directive",
};

export const RELATIONSHIP_CATEGORY_COLORS: Record<string, { bg: string; text: string; stroke: string }> = {
  [RelationshipCategory.CAUSAL]: { bg: "bg-red-50", text: "text-red-700", stroke: "#ef4444" },
  [RelationshipCategory.SOLUTION]: { bg: "bg-green-50", text: "text-green-700", stroke: "#22c55e" },
  [RelationshipCategory.CONTEXT]: { bg: "bg-sky-50", text: "text-sky-700", stroke: "#0ea5e9" },
  [RelationshipCategory.LEARNING]: { bg: "bg-violet-50", text: "text-violet-700", stroke: "#8b5cf6" },
  [RelationshipCategory.SIMILARITY]: { bg: "bg-orange-50", text: "text-orange-700", stroke: "#f97316" },
  [RelationshipCategory.WORKFLOW]: { bg: "bg-indigo-50", text: "text-indigo-700", stroke: "#6366f1" },
  [RelationshipCategory.QUALITY]: { bg: "bg-teal-50", text: "text-teal-700", stroke: "#14b8a6" },
};

export const RELATIONSHIP_CATEGORY_LABELS: Record<string, string> = {
  [RelationshipCategory.CAUSAL]: "Causal",
  [RelationshipCategory.SOLUTION]: "Solution",
  [RelationshipCategory.CONTEXT]: "Context",
  [RelationshipCategory.LEARNING]: "Learning",
  [RelationshipCategory.SIMILARITY]: "Similarity",
  [RelationshipCategory.WORKFLOW]: "Workflow",
  [RelationshipCategory.QUALITY]: "Quality",
};

/** Map common relationship types to their category */
export const RELATIONSHIP_TO_CATEGORY: Record<string, string> = {
  causes: "causal",
  triggers: "causal",
  leads_to: "causal",
  prevents: "causal",
  solves: "solution",
  addresses: "solution",
  alternative_to: "solution",
  improves: "solution",
  occurs_in: "context",
  applies_to: "context",
  works_with: "context",
  requires: "context",
  builds_on: "learning",
  contradicts: "learning",
  confirms: "learning",
  supersedes: "learning",
  similar_to: "similarity",
  variant_of: "similarity",
  related_to: "similarity",
  follows: "workflow",
  depends_on: "workflow",
  enables: "workflow",
  blocks: "workflow",
  effective_for: "quality",
  preferred_over: "quality",
  deprecated_by: "quality",
  part_of: "context",
  contains: "context",
  instance_of: "context",
  subtype_of: "context",
  precedes: "workflow",
  concurrent_with: "workflow",
};

export const IMPORTANCE_LABELS: Record<string, string> = {
  low: "Low (0-0.3)",
  medium: "Medium (0.3-0.7)",
  high: "High (0.7-1.0)",
};

export function getImportanceLevel(value: number): "low" | "medium" | "high" {
  if (value < 0.3) return "low";
  if (value < 0.7) return "medium";
  return "high";
}

export function getImportanceColor(value: number): string {
  if (value < 0.3) return "text-slate-500";
  if (value < 0.7) return "text-amber-600";
  return "text-red-600";
}
