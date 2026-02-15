/**
 * MCP tool definitions for MemoryLayer.ai
 *
 * Mirrors the Python tool definitions from server/memorylayer/mcp/tools.py
 */

/**
 * All available MCP tools for MemoryLayer
 */
export const TOOLS = [
  {
    name: "memory_remember",
    description: "Store a new memory for later recall. Use for facts, preferences, decisions, or events worth remembering.",
    inputSchema: {
      type: "object",
      properties: {
        content: {
          type: "string",
          description: "The memory content to store"
        },
        type: {
          type: "string",
          enum: ["episodic", "semantic", "procedural", "working"],
          description: "Memory type: episodic (events), semantic (facts), procedural (how-to), working (current context)"
        },
        importance: {
          type: "number",
          minimum: 0,
          maximum: 1,
          default: 0.5,
          description: "How important (0-1, default 0.5). Higher values = retained longer and ranked higher in recall"
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Tags for categorization (e.g., ['python', 'bug-fix'])"
        },
        subtype: {
          type: "string",
          enum: ["solution", "problem", "code_pattern", "fix", "error", "workflow", "preference", "decision", "directive"],
          description: "Optional domain-specific classification"
        }
      },
      required: ["content"]
    }
  },
  {
    name: "memory_recall",
    description: "Search memories by semantic query. Returns relevant memories ranked by relevance. Use this to find previously stored information.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Natural language query (e.g., 'How do I fix authentication errors?')"
        },
        types: {
          type: "array",
          items: {
            type: "string",
            enum: ["episodic", "semantic", "procedural", "working"]
          },
          description: "Filter by memory types"
        },
        limit: {
          type: "integer",
          default: 10,
          minimum: 1,
          maximum: 100,
          description: "Max memories to return"
        },
        min_relevance: {
          type: "number",
          minimum: 0,
          maximum: 1,
          description: "Min relevance score (0-1). Omit for server default."
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Filter by tags (AND logic)"
        }
      },
      required: ["query"]
    }
  },
  {
    name: "memory_reflect",
    description: "Synthesize and summarize memories matching a query. Use when you need insights across multiple memories rather than individual recall results.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "What to reflect on (e.g., 'What patterns have we seen with database performance?')"
        },
        detail_level: {
          type: "string",
          enum: ["abstract", "overview", "full"],
          description: "Level of detail for reflection output: abstract (brief), overview (medium), full (detailed). Omit for server default."
        },
        include_sources: {
          type: "boolean",
          default: true,
          description: "Include source memory IDs in response"
        },
        depth: {
          type: "integer",
          default: 2,
          minimum: 1,
          maximum: 5,
          description: "Association traversal depth (how many hops to follow)"
        }
      },
      required: ["query"]
    }
  },
  {
    name: "memory_forget",
    description: "Delete or decay a memory when information is outdated or incorrect. Use sparingly - memories are useful historical context.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: {
          type: "string",
          description: "ID of memory to forget"
        },
        reason: {
          type: "string",
          description: "Why this memory should be forgotten (for audit trail)"
        },
        hard: {
          type: "boolean",
          default: false,
          description: "Hard delete (permanent) vs soft delete (recoverable)"
        }
      },
      required: ["memory_id"]
    }
  },
  {
    name: "memory_associate",
    description: "Link two memories with a relationship. Helps build knowledge graph for traversal and causal reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        source_id: {
          type: "string",
          description: "Source memory ID"
        },
        target_id: {
          type: "string",
          description: "Target memory ID"
        },
        relationship: {
          type: "string",
          description: "Type of relationship between memories (e.g., 'causes', 'solves', 'similar_to', 'part_of', 'related_to'). The server supports ~65 relationship types from its ontology."
        },
        strength: {
          type: "number",
          minimum: 0,
          maximum: 1,
          default: 0.8,
          description: "Strength of association (0-1)"
        }
      },
      required: ["source_id", "target_id", "relationship"]
    }
  },
  {
    name: "memory_briefing",
    description: "Get a session briefing summarizing recent activity and context. Returns workspace stats and recent memory content at configurable detail level. Use at session start or after compaction to regain context.",
    inputSchema: {
      type: "object",
      properties: {
        time_window_minutes: {
          type: "number",
          default: 60,
          description: "Time window in minutes for recent memories (default: 60)"
        },
        detail_level: {
          type: "string",
          enum: ["abstract", "overview", "full"],
          default: "abstract",
          description: "Level of memory detail: abstract (summaries), overview (more detail), full (complete content)"
        },
        limit: {
          type: "number",
          default: 10,
          description: "Maximum number of recent memories to include (max: 50)"
        },
        include_memories: {
          type: "boolean",
          default: true,
          description: "Whether to include recent memory content in the briefing"
        },
        include_contradictions: {
          type: "boolean",
          default: true,
          description: "Flag contradicting memories in briefing"
        }
      }
    }
  },
  // Advanced tools
  {
    name: "memory_statistics",
    description: "Get memory statistics and analytics for the workspace. Use to understand memory usage patterns.",
    inputSchema: {
      type: "object",
      properties: {
        include_breakdown: {
          type: "boolean",
          default: true,
          description: "Include breakdown by type/subtype"
        }
      }
    }
  },
  {
    name: "memory_graph_query",
    description: "Multi-hop graph traversal to find related memories. Use to discover causal chains or knowledge paths.",
    inputSchema: {
      type: "object",
      properties: {
        start_memory_id: {
          type: "string",
          description: "Starting memory ID"
        },
        relationship_types: {
          type: "array",
          items: { type: "string" },
          description: "Filter by relationship types (e.g., ['causes', 'triggers'])"
        },
        max_depth: {
          type: "integer",
          default: 3,
          minimum: 1,
          maximum: 5,
          description: "Maximum traversal depth"
        },
        direction: {
          type: "string",
          enum: ["outgoing", "incoming", "both"],
          default: "both",
          description: "Traversal direction"
        },
        max_paths: {
          type: "integer",
          default: 50,
          description: "Maximum paths to return"
        }
      },
      required: ["start_memory_id"]
    }
  },
  {
    name: "memory_audit",
    description: "Audit memories for contradictions and inconsistencies. Use to maintain knowledge base health.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: {
          type: "string",
          description: "Specific memory to audit (omit to audit entire workspace)"
        },
        auto_resolve: {
          type: "boolean",
          default: false,
          description: "Automatically mark newer contradicting memories as preferred"
        }
      }
    }
  },
  // TODO: memory_compress - commented out until compression logic is implemented
  // {
  //   name: "memory_compress",
  //   ...
  // }
];

/**
 * Session-aware tools for working memory management.
 * These are only available when session mode is enabled.
 */
export const SESSION_TOOLS = [
  {
    name: "memory_session_start",
    description: "Start a new session for working memory. Called automatically by SessionStart hook, but can be called manually. After context compaction, call memory_context_inspect to re-orient with existing sandbox state. Returns session info.",
    inputSchema: {
      type: "object",
      properties: {
        metadata: {
          type: "object",
          description: "Optional metadata to attach to the session"
        }
      }
    }
  },
  {
    name: "memory_session_end",
    description: "End the current session and optionally commit working memory to long-term storage. Called automatically by SessionEnd hook.",
    inputSchema: {
      type: "object",
      properties: {
        commit: {
          type: "boolean",
          default: true,
          description: "Whether to commit working memory to long-term storage (extracts memories)"
        },
        importance_threshold: {
          type: "number",
          minimum: 0,
          maximum: 1,
          default: 0.5,
          description: "Minimum importance for extracted memories"
        }
      }
    }
  },
  {
    name: "memory_session_commit",
    description: "Commit working memory to long-term storage WITHOUT ending the session. Use for checkpoints during long sessions or before potential interruptions. The session remains active after commit.",
    inputSchema: {
      type: "object",
      properties: {
        importance_threshold: {
          type: "number",
          minimum: 0,
          maximum: 1,
          default: 0.5,
          description: "Minimum importance for extracted memories"
        },
        clear_after_commit: {
          type: "boolean",
          default: false,
          description: "Clear working memory after commit (session stays active)"
        }
      }
    }
  },
  {
    name: "memory_session_status",
    description: "Get current session status including working memory summary. Use to check if a session is active.",
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
];

/**
 * Context Environment tools for server-side sandbox execution.
 * These replace the old key-value context tools (memory_context_set/get/list/delete)
 * with a powerful code execution environment backed by the Python API server.
 */
export const CONTEXT_ENVIRONMENT_TOOLS = [
  {
    name: "memory_context_exec",
    description: "Execute Python code in the server-side sandbox. Variables from prior exec calls are available. Use for data transformation, analysis, and computation on loaded memories.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Python code to execute in sandbox. Variables from prior exec calls are available."
        },
        result_var: {
          type: "string",
          description: "Optional: store the expression result in this variable name"
        },
        return_result: {
          type: "boolean",
          default: true,
          description: "Return execution output to caller. Set false for large intermediate results."
        },
        max_return_chars: {
          type: "integer",
          default: 10000,
          description: "Maximum characters to return. Output beyond this is truncated with a notice."
        }
      },
      required: ["code"]
    }
  },
  {
    name: "memory_context_inspect",
    description: "Inspect variables in the server-side sandbox. Omit variable name for an overview of all variables, or specify one for detailed inspection. IMPORTANT: Call this after context compaction or at the start of continued sessions to re-orient — sandbox state persists server-side even when your context resets.",
    inputSchema: {
      type: "object",
      properties: {
        variable: {
          type: "string",
          description: "Optional: inspect a specific variable in detail. Omit for overview of all variables."
        },
        preview_chars: {
          type: "integer",
          default: 200,
          description: "Characters to include in value previews"
        }
      }
    }
  },
  {
    name: "memory_context_load",
    description: "Load memories from the store into a sandbox variable via semantic search. The loaded memories become available for exec, query, and RLM operations.",
    inputSchema: {
      type: "object",
      properties: {
        var: {
          type: "string",
          description: "Variable name to store loaded memories"
        },
        query: {
          type: "string",
          description: "Semantic search query"
        },
        limit: {
          type: "integer",
          default: 50,
          description: "Maximum memories to load"
        },
        types: {
          type: "array",
          items: {
            type: "string",
            enum: ["episodic", "semantic", "procedural", "working"]
          },
          description: "Filter by memory types"
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Filter by tags"
        },
        min_relevance: {
          type: "number",
          minimum: 0,
          maximum: 1,
          description: "Minimum relevance score"
        },
        include_embeddings: {
          type: "boolean",
          default: false,
          description: "Include embedding vectors (large). Usually false."
        }
      },
      required: ["var", "query"]
    }
  },
  {
    name: "memory_context_inject",
    description: "Inject a value directly into the server-side sandbox as a named variable. Use to pass data from the client into the execution environment.",
    inputSchema: {
      type: "object",
      properties: {
        key: {
          type: "string",
          description: "Variable name in sandbox"
        },
        value: {
          type: "string",
          description: "Value to store (string, JSON string, or code snippet)"
        },
        parse_json: {
          type: "boolean",
          default: false,
          description: "Parse value as JSON before storing (creates dict/list, not string)"
        }
      },
      required: ["key", "value"]
    }
  },
  {
    name: "memory_context_query",
    description: "Ask the server-side LLM a question using sandbox variables as context. The LLM sees the variable data and answers your prompt.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "Question or instruction for the server LLM to answer using the variable data"
        },
        variables: {
          type: "array",
          items: { type: "string" },
          description: "Variable names to include as context for the LLM query"
        },
        max_context_chars: {
          type: "integer",
          description: "Max chars of variable data to send to LLM. Omit for server default."
        },
        result_var: {
          type: "string",
          description: "Optional: store LLM response in this variable"
        }
      },
      required: ["prompt", "variables"]
    }
  },
  {
    name: "memory_context_rlm",
    description: "Run a Recursive Language Model (RLM) loop: the server LLM iteratively reasons over sandbox data and memories to achieve a goal. Returns a synthesis of findings.",
    inputSchema: {
      type: "object",
      properties: {
        goal: {
          type: "string",
          description: "Natural language goal for the RLM loop"
        },
        memory_query: {
          type: "string",
          description: "Optional: semantic query to pre-load memories into sandbox before RLM loop starts"
        },
        memory_limit: {
          type: "integer",
          default: 100,
          description: "Max memories to pre-load"
        },
        max_iterations: {
          type: "integer",
          default: 10,
          minimum: 1,
          maximum: 50,
          description: "Maximum RLM loop iterations"
        },
        variables: {
          type: "array",
          items: { type: "string" },
          description: "Existing sandbox variables to include as context"
        },
        result_var: {
          type: "string",
          description: "Optional: store the final synthesis in this variable"
        },
        detail_level: {
          type: "string",
          enum: ["brief", "standard", "detailed"],
          default: "standard",
          description: "How much detail to include in the returned synthesis"
        }
      },
      required: ["goal"]
    }
  },
  {
    name: "memory_context_status",
    description: "Get the status of the server-side context environment including sandbox state, active variables, and resource usage. Use after context compaction to check if a sandbox environment still exists.",
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "memory_context_checkpoint",
    description: "Checkpoint the sandbox state for persistence. Fires persistence hooks so enterprise deployments can save sandbox state to durable storage. No-op for default in-memory deployments.",
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
];

/**
 * Tool profiles for different use cases.
 *
 * The "cc" (Claude Code) profile is the recommended default - it provides
 * the essential tools for agent memory without overwhelming the tool list.
 */

/** All tool names by category */
export const TOOL_NAMES = {
  // Core memory operations
  remember: "memory_remember",
  recall: "memory_recall",
  reflect: "memory_reflect",
  forget: "memory_forget",
  associate: "memory_associate",

  // Extended/advanced operations
  briefing: "memory_briefing",
  statistics: "memory_statistics",
  graphQuery: "memory_graph_query",
  audit: "memory_audit",

  // Session management
  sessionStart: "memory_session_start",
  sessionEnd: "memory_session_end",
  sessionCommit: "memory_session_commit",
  sessionStatus: "memory_session_status",

  // Context environment (server-side sandbox)
  contextExec: "memory_context_exec",
  contextInspect: "memory_context_inspect",
  contextLoad: "memory_context_load",
  contextInject: "memory_context_inject",
  contextQuery: "memory_context_query",
  contextRlm: "memory_context_rlm",
  contextStatus: "memory_context_status",
  contextCheckpoint: "memory_context_checkpoint",
} as const;

/**
 * Tool profile definitions.
 *
 * - "cc": Claude Code profile (default) - 9 essential tools for agent memory
 * - "full": All tools enabled - for power users and advanced use cases
 * - "minimal": Just remember/recall - absolute minimum
 */
export type ToolProfile = "cc" | "full" | "minimal";

export const TOOL_PROFILES: Record<ToolProfile, string[]> = {
  /**
   * Claude Code profile (DEFAULT)
   *
   * 17 tools optimized for the Claude Code use case:
   * - Core: remember, recall, reflect, forget
   * - Session: session_start, session_end, session_commit, session_status
   * - Utility: briefing
   * - Context Environment: exec, inspect, load, inject, query, rlm, status, checkpoint
   *
   * Excluded (available in "full" profile):
   * - associate: Graph building is advanced, rarely used without guidance
   * - statistics: Admin/debugging, not agent workflow
   * - graph_query: Power user feature
   * - audit: Server-side background task
   */
  cc: [
    TOOL_NAMES.remember,
    TOOL_NAMES.recall,
    TOOL_NAMES.reflect,
    TOOL_NAMES.forget,
    TOOL_NAMES.briefing,
    TOOL_NAMES.sessionStart,
    TOOL_NAMES.sessionEnd,
    TOOL_NAMES.sessionCommit,
    TOOL_NAMES.sessionStatus,
    // Context environment
    TOOL_NAMES.contextExec,
    TOOL_NAMES.contextInspect,
    TOOL_NAMES.contextLoad,
    TOOL_NAMES.contextInject,
    TOOL_NAMES.contextQuery,
    TOOL_NAMES.contextRlm,
    TOOL_NAMES.contextStatus,
    TOOL_NAMES.contextCheckpoint,
  ],

  /**
   * Full profile - all tools enabled
   * For power users, debugging, or advanced graph-based memory use cases.
   */
  full: [
    // Core
    TOOL_NAMES.remember,
    TOOL_NAMES.recall,
    TOOL_NAMES.reflect,
    TOOL_NAMES.forget,
    TOOL_NAMES.associate,
    // Extended
    TOOL_NAMES.briefing,
    TOOL_NAMES.statistics,
    TOOL_NAMES.graphQuery,
    TOOL_NAMES.audit,
    // Session
    TOOL_NAMES.sessionStart,
    TOOL_NAMES.sessionEnd,
    TOOL_NAMES.sessionCommit,
    TOOL_NAMES.sessionStatus,
    // Context environment
    TOOL_NAMES.contextExec,
    TOOL_NAMES.contextInspect,
    TOOL_NAMES.contextLoad,
    TOOL_NAMES.contextInject,
    TOOL_NAMES.contextQuery,
    TOOL_NAMES.contextRlm,
    TOOL_NAMES.contextStatus,
    TOOL_NAMES.contextCheckpoint,
  ],

  /**
   * Minimal profile - just the essentials
   * For testing or extremely constrained environments.
   */
  minimal: [
    TOOL_NAMES.remember,
    TOOL_NAMES.recall,
  ],
};

/** Default profile for MCP server */
export const DEFAULT_PROFILE: ToolProfile = "cc";

/** Tool definition shape */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: string;
    properties?: Record<string, unknown>;
    required?: string[];
  };
}

/**
 * Get tools for a given profile.
 */
export function getToolsForProfile(profile: ToolProfile): ToolDefinition[] {
  const allTools: ToolDefinition[] = [...TOOLS, ...SESSION_TOOLS, ...CONTEXT_ENVIRONMENT_TOOLS];
  const enabledNames = TOOL_PROFILES[profile];
  return allTools.filter(tool => enabledNames.includes(tool.name));
}

/**
 * Check if a tool is enabled for a given profile.
 */
export function isToolEnabled(toolName: string, profile: ToolProfile): boolean {
  return TOOL_PROFILES[profile].includes(toolName);
}

// Legacy exports for backwards compatibility
// TODO: Remove in next major version

/**
 * @deprecated Use getToolsForProfile("cc") or TOOL_PROFILES instead
 */
export const CORE_TOOLS = TOOLS.filter(tool =>
  ["memory_remember", "memory_recall", "memory_reflect", "memory_forget", "memory_associate"].includes(tool.name)
);

/**
 * @deprecated Use getToolsForProfile("full") or TOOL_PROFILES instead
 */
export const EXTENDED_TOOLS = TOOLS.filter(tool =>
  ["memory_briefing", "memory_statistics", "memory_graph_query", "memory_audit"].includes(tool.name)
);
