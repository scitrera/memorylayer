/**
 * Observation extraction for auto-capture hooks.
 * Extracts structured observation data from tool usage.
 *
 * Adapted from the CC plugin with OpenCode tool name mappings.
 * OpenCode uses lowercase tool names: bash, edit, write, read, glob, grep,
 * task, webfetch, websearch, multiedit, ls, apply_patch, codesearch, etc.
 */

import { createHash } from "crypto";

// Tools to skip (internal/noisy/self-referential)
const SKIP_TOOLS = new Set([
  "todo",
  "question",
  "plan",
  "skill",
  "batch",
]);

// Also skip any tool matching these prefixes (memory tools)
const SKIP_PREFIXES = [
  "memorylayer",
  "memory_",
  "mcp__memorylayer",
];

export function shouldSkipTool(toolName: string): boolean {
  if (SKIP_TOOLS.has(toolName)) return true;
  return SKIP_PREFIXES.some(prefix => toolName.startsWith(prefix));
}

// Observation types
export type ObservationType = "read" | "write" | "execute" | "search" | "other";

export interface ObservationData {
  type: ObservationType;
  title: string;
  toolName: string;
  filesRead: string[];
  filesModified: string[];
  facts: string[];
  concepts: string[];
  intent: string | null;
  contentHash: string;
  summary: string;
}

/**
 * Classify tool as read/write/execute/search/other.
 * Uses OpenCode's lowercase tool names.
 */
export function getObservationType(toolName: string): ObservationType {
  const readTools = ["read", "glob", "grep", "ls", "codesearch"];
  const writeTools = ["write", "edit", "multiedit", "apply_patch"];
  const executeTools = ["bash", "task"];
  const searchTools = ["websearch", "webfetch"];

  if (readTools.includes(toolName)) return "read";
  if (writeTools.includes(toolName)) return "write";
  if (executeTools.includes(toolName)) return "execute";
  if (searchTools.includes(toolName)) return "search";
  return "other";
}

/**
 * Extract file paths from tool args, classified as read or modified
 */
export function extractFilePaths(
  toolName: string,
  toolArgs: Record<string, unknown>
): { filesRead: string[]; filesModified: string[] } {
  const filesRead: string[] = [];
  const filesModified: string[] = [];

  try {
    const filePath = (toolArgs.file_path || toolArgs.path || toolArgs.file || "") as string;

    if (!filePath) {
      // Try to extract paths from bash commands
      if (toolName === "bash") {
        const command = (toolArgs.command || "") as string;
        const pathMatches = command.match(
          /(?:^|\s)([^\s]+\.(ts|js|py|json|yaml|yml|md|txt|go|rs|java|c|cpp|h))\b/gi
        );
        if (pathMatches) {
          filesRead.push(...pathMatches.map((p: string) => p.trim()));
        }
      }
      return { filesRead, filesModified };
    }

    const type = getObservationType(toolName);
    if (type === "write") {
      filesModified.push(filePath);
    } else if (type === "read") {
      filesRead.push(filePath);
    }
  } catch {
    // Ignore parse errors
  }

  return { filesRead, filesModified };
}

/**
 * Generate observation title from tool usage
 */
export function generateTitle(toolName: string, toolArgs: Record<string, unknown>): string {
  try {
    switch (toolName) {
      case "read":
        return `Read ${toolArgs.file_path || toolArgs.path || "file"}`;
      case "write":
        return `Write ${toolArgs.file_path || toolArgs.path || "file"}`;
      case "edit":
      case "multiedit":
        return `Edit ${toolArgs.file_path || toolArgs.path || "file"}`;
      case "apply_patch":
        return `Patch ${toolArgs.file_path || toolArgs.path || "file"}`;
      case "bash": {
        const cmd = (toolArgs.command || "") as string;
        return `Run: ${cmd.substring(0, 50)}${cmd.length > 50 ? "..." : ""}`;
      }
      case "glob":
        return `Find ${toolArgs.pattern || "files"}`;
      case "grep":
      case "codesearch":
        return `Search "${toolArgs.pattern || ""}"`;
      case "task":
        return `Task: ${toolArgs.description || "agent"}`;
      case "websearch":
        return `Search: ${toolArgs.query || ""}`;
      case "webfetch":
        return `Fetch: ${toolArgs.url || ""}`;
      default:
        return toolName;
    }
  } catch {
    return toolName;
  }
}

/**
 * Extract facts from tool args/output
 */
export function extractFacts(
  toolName: string,
  toolArgs: Record<string, unknown>,
  toolOutput?: string
): string[] {
  const facts: string[] = [];

  try {
    const filePath = (toolArgs.file_path || toolArgs.path || toolArgs.file || "") as string;

    switch (toolName) {
      case "read":
        if (filePath) facts.push(`File read: ${filePath}`);
        break;
      case "write":
        if (filePath) facts.push(`File created/updated: ${filePath}`);
        break;
      case "edit":
      case "multiedit":
      case "apply_patch":
        if (filePath) facts.push(`File modified: ${filePath}`);
        if (toolArgs.old_string) {
          facts.push(`Code replaced in ${filePath.split(/[/\\]/).pop() || "file"}`);
        }
        break;
      case "bash": {
        const cmd = (toolArgs.command || "") as string;
        facts.push(`Command executed: ${cmd.substring(0, 100)}`);
        if (toolOutput) {
          if (toolOutput.includes("passed") || toolOutput.includes("\u2713")) facts.push("Tests passed");
          if (toolOutput.includes("failed") || toolOutput.includes("\u2717")) facts.push("Tests failed");
          if (toolOutput.includes("error") || toolOutput.includes("Error")) facts.push("Errors encountered");
        }
        break;
      }
      case "glob":
        if (toolArgs.pattern) facts.push(`Pattern searched: ${toolArgs.pattern}`);
        break;
      case "grep":
      case "codesearch":
        if (toolArgs.pattern) facts.push(`Code pattern searched: ${toolArgs.pattern}`);
        if (toolArgs.path) facts.push(`Search scope: ${toolArgs.path}`);
        break;
      case "websearch":
        if (toolArgs.query) facts.push(`Web search: ${toolArgs.query}`);
        break;
      case "webfetch":
        if (toolArgs.url) facts.push(`URL fetched: ${toolArgs.url}`);
        break;
      case "task":
        if (toolArgs.description) facts.push(`Sub-task: ${toolArgs.description}`);
        break;
    }
  } catch {
    // Ignore parse errors
  }

  return facts;
}

/**
 * Extract concepts/topics from tool usage
 */
export function extractConcepts(toolName: string, toolArgs: Record<string, unknown>): string[] {
  const concepts: Set<string> = new Set();

  try {
    const filePath = (toolArgs.file_path || toolArgs.path || toolArgs.file || "") as string;

    // Extract concepts from file paths
    if (filePath) {
      const parts = filePath.split(/[/\\]/);
      for (const part of parts) {
        if (["src", "lib", "dist", "node_modules", ".", ".."].includes(part)) continue;
        if (part.includes(".")) {
          const ext = part.split(".").pop();
          const extMap: Record<string, string> = {
            ts: "typescript", tsx: "react", js: "javascript", jsx: "react",
            py: "python", rs: "rust", go: "golang", css: "styling", scss: "styling",
            html: "html", json: "configuration", yaml: "configuration", yml: "configuration",
            md: "documentation", test: "testing", spec: "testing", sql: "database",
          };
          if (ext && extMap[ext]) concepts.add(extMap[ext]);
        }
        const dirMap: Record<string, string> = {
          tests: "testing", __tests__: "testing", test: "testing", spec: "testing",
          hooks: "hooks", api: "api", auth: "authentication", db: "database",
          components: "components", pages: "pages", routes: "routing", utils: "utilities",
          services: "services", middleware: "middleware", models: "models", types: "types",
          cli: "cli", config: "configuration", migrations: "database", schemas: "schemas",
        };
        if (dirMap[part]) concepts.add(dirMap[part]);
      }
    }

    // Extract function/class names from edit tools
    if (toolName === "edit" || toolName === "multiedit") {
      const oldStr = (toolArgs.old_string || "") as string;
      const newStr = (toolArgs.new_string || "") as string;
      const combined = oldStr + "\n" + newStr;

      const funcMatches = combined.match(/(?:function|async function|const|let|var)\s+(\w{3,})/g);
      if (funcMatches) {
        for (const m of funcMatches.slice(0, 3)) {
          const name = m.replace(/(?:function|async function|const|let|var)\s+/, "");
          concepts.add(`fn:${name}`);
        }
      }

      const classMatches = combined.match(/class\s+(\w{3,})/g);
      if (classMatches) {
        for (const m of classMatches.slice(0, 2)) {
          concepts.add(`class:${m.replace("class ", "")}`);
        }
      }

      if (/\bimport\b/.test(combined)) concepts.add("pattern:import");
      if (/\bexport\b/.test(combined)) concepts.add("pattern:export");
      if (/\binterface\b/.test(combined)) concepts.add("pattern:interface");
      if (/\benum\b/.test(combined)) concepts.add("pattern:enum");
      if (/\btry\s*\{/.test(combined)) concepts.add("pattern:error-handling");
      if (/\basync\b/.test(combined)) concepts.add("pattern:async");
    }

    // Tool-based concepts
    switch (toolName) {
      case "bash": {
        const cmd = (toolArgs.command || "") as string;
        if (cmd.includes("test") || cmd.includes("vitest") || cmd.includes("jest")) concepts.add("testing");
        if (cmd.includes("build") || cmd.includes("tsc")) concepts.add("build");
        if (cmd.includes("git")) concepts.add("version-control");
        if (cmd.includes("npm") || cmd.includes("yarn") || cmd.includes("pnpm") || cmd.includes("bun")) concepts.add("package-management");
        if (cmd.includes("docker")) concepts.add("containerization");
        if (cmd.includes("lint") || cmd.includes("eslint") || cmd.includes("biome")) concepts.add("linting");
        break;
      }
      case "websearch":
        concepts.add("research");
        break;
      case "webfetch":
        concepts.add("web-content");
        break;
      case "task":
        concepts.add("delegation");
        break;
    }
  } catch {
    // Ignore parse errors
  }

  return Array.from(concepts);
}

/**
 * Detect intent from tool usage and user prompt
 */
export function detectIntent(
  toolName: string,
  toolArgs: Record<string, unknown>,
  prompt?: string
): string | null {
  const intentPatterns = {
    bugfix: /fix|bug|error|issue|broken|crash|repair/i,
    feature: /add|feature|implement|create|new|build/i,
    refactor: /refactor|clean|rename|reorganize|restructure/i,
    testing: /test|spec|coverage|verify/i,
    investigation: /find|search|investigate|debug|analyze|explore/i,
    documentation: /document|comment|readme|doc|explain/i,
  };

  if (prompt) {
    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(prompt)) {
        return intent;
      }
    }
  }

  try {
    const inputStr = JSON.stringify(toolArgs);
    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(inputStr)) {
        return intent;
      }
    }
  } catch {
    // Ignore parse errors
  }

  switch (toolName) {
    case "read":
    case "glob":
    case "grep":
    case "codesearch":
      return "investigation";
    case "write":
      return "feature";
    case "edit":
      return null; // Too ambiguous without context
    default:
      return null;
  }
}

/**
 * Compute content hash for deduplication
 */
export function computeContentHash(...parts: string[]): string {
  return createHash("sha256")
    .update(parts.join("|"))
    .digest("hex")
    .slice(0, 16);
}

/**
 * Truncate string to max length
 */
function truncate(str: string, maxLen: number = 500): string {
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen) + "...";
}

/**
 * Build observation from tool execution data
 */
export function buildObservation(
  toolName: string,
  toolArgs: Record<string, unknown>,
  toolOutput?: string,
  currentPrompt?: string
): ObservationData | null {
  const type = getObservationType(toolName);
  const title = generateTitle(toolName, toolArgs);
  const { filesRead, filesModified } = extractFilePaths(toolName, toolArgs);
  const facts = extractFacts(toolName, toolArgs, toolOutput);
  const concepts = extractConcepts(toolName, toolArgs);
  const intent = detectIntent(toolName, toolArgs, currentPrompt);

  const summary = buildSummary(toolName, toolArgs, toolOutput, filesRead, filesModified);

  const contentHash = computeContentHash(
    toolName,
    JSON.stringify(toolArgs),
    summary
  );

  // Skip if empty observation (no meaningful data extracted)
  if (filesRead.length === 0 && filesModified.length === 0 && facts.length === 0 && concepts.length === 0) {
    return null;
  }

  return {
    type,
    title,
    toolName,
    filesRead,
    filesModified,
    facts,
    concepts,
    intent,
    contentHash,
    summary,
  };
}

/**
 * Build human-readable summary
 */
function buildSummary(
  toolName: string,
  toolArgs: Record<string, unknown>,
  _toolOutput: string | undefined,
  filesRead: string[],
  filesModified: string[]
): string {
  const parts: string[] = [];

  if (filesRead.length > 0) {
    parts.push(`Read: ${filesRead.join(", ")}`);
  }
  if (filesModified.length > 0) {
    parts.push(`Modified: ${filesModified.join(", ")}`);
  }

  try {
    if (toolName === "bash") {
      const cmd = (toolArgs.command || "") as string;
      parts.push(`Command: ${truncate(cmd, 100)}`);
    } else if (toolName === "grep" || toolName === "codesearch") {
      const pattern = (toolArgs.pattern || "") as string;
      parts.push(`Pattern: ${pattern}`);
    } else if (toolName === "task") {
      const desc = (toolArgs.description || "") as string;
      parts.push(`Task: ${truncate(desc, 100)}`);
    }
  } catch {
    // Ignore
  }

  return parts.length > 0 ? parts.join("; ") : `Used ${toolName}`;
}
