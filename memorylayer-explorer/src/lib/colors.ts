import { RELATIONSHIP_TO_CATEGORY } from "./constants";

/** Memory type -> hex color for node borders */
export const memoryTypeColors: Record<string, string> = {
  episodic: "#3b82f6",
  semantic: "#10b981",
  procedural: "#f59e0b",
  working: "#a78bfa",
};

/** Relationship category -> hex color for edges */
export const relationshipCategoryColors: Record<string, string> = {
  causal: "#ef4444",
  solution: "#22c55e",
  context: "#0ea5e9",
  learning: "#8b5cf6",
  similarity: "#f97316",
  workflow: "#6366f1",
  quality: "#14b8a6",
};

/** Map a relationship type string to its category, defaulting to "context" */
export function getRelationshipCategory(relationship: string): string {
  return RELATIONSHIP_TO_CATEGORY[relationship] ?? "context";
}
