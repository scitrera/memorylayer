export type {
  Memory,
  RecallResult,
  ReflectResult,
  Association,
  Session,
  SessionBriefing,
  Workspace,
  Context,
  RememberOptions,
  RecallOptions,
  ReflectOptions,
  ClientConfig,
  SessionCreateOptions,
  SessionStartResponse,
  CommitOptions,
  CommitResponse,
  GraphTraverseOptions,
  GraphPath,
  GraphQueryResult,
  BatchOperation,
  BatchResult,
  AssociationCreateOptions,
  WorkspaceSchema,
} from "@scitrera/memorylayer-sdk";

export {
  MemoryType,
  MemorySubtype,
  RecallMode,
  SearchTolerance,
  DetailLevel,
  RelationshipCategory,
  RELATIONSHIP_TYPES,
} from "@scitrera/memorylayer-sdk";

export type ViewMode = "grid" | "list";
export type SortField = "created_at" | "importance" | "access_count" | "updated_at";
export type SortOrder = "asc" | "desc";

export interface FilterState {
  types: string[];
  subtypes: string[];
  tags: string[];
  status: string;
  importanceRange: [number, number];
  dateRange: { from?: Date; to?: Date };
}
