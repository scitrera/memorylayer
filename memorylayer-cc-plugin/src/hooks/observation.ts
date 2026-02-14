/**
 * Observation extraction for auto-capture hooks.
 * Extracts structured observation data from tool usage.
 */

import { createHash } from 'crypto';
import type { HookInput } from './types.js';

// Tools to skip (internal/noisy/self-referential)
const SKIP_TOOLS = new Set([
  'TodoWrite', 'TodoRead',
  'AskFollowupQuestion', 'AskUserQuestion',
  'AttemptCompletion',
  'LS',  // Low signal
]);

// Also skip any tool matching these prefixes (memory tools)
const SKIP_PREFIXES = [
  'mcp__memorylayer',
  'mcp__memory',
  'memory_',
];

export function shouldSkipTool(toolName: string): boolean {
  if (SKIP_TOOLS.has(toolName)) return true;
  return SKIP_PREFIXES.some(prefix => toolName.startsWith(prefix));
}

// Observation types
export type ObservationType = 'read' | 'write' | 'execute' | 'search' | 'other';

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
 * Classify tool as read/write/execute/search/other
 */
export function getObservationType(toolName: string): ObservationType {
  const readTools = ['Read', 'Glob', 'Grep', 'LS'];
  const writeTools = ['Write', 'Edit', 'MultiEdit', 'NotebookEdit'];
  const executeTools = ['Bash', 'Task', 'Skill'];
  const searchTools = ['WebSearch', 'WebFetch'];

  if (readTools.includes(toolName)) return 'read';
  if (writeTools.includes(toolName)) return 'write';
  if (executeTools.includes(toolName)) return 'execute';
  if (searchTools.includes(toolName)) return 'search';
  return 'other';
}

/**
 * Extract file paths from tool input, classified as read or modified
 */
export function extractFilePaths(toolName: string, toolInput: unknown): { filesRead: string[]; filesModified: string[] } {
  const filesRead: string[] = [];
  const filesModified: string[] = [];

  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};
    const filePath = (input as any)?.file_path || (input as any)?.path || '';

    if (!filePath) {
      // Try to extract paths from bash commands
      if (toolName === 'Bash') {
        const command = (input as any)?.command || '';
        const pathMatches = command.match(/(?:^|\s)([^\s]+\.(ts|js|py|json|yaml|yml|md|txt|go|rs|java|c|cpp|h))\b/gi);
        if (pathMatches) {
          filesRead.push(...pathMatches.map((p: string) => p.trim()));
        }
      }
      return { filesRead, filesModified };
    }

    const type = getObservationType(toolName);
    if (type === 'write') {
      filesModified.push(filePath);
    } else if (type === 'read') {
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
export function generateTitle(toolName: string, toolInput: unknown): string {
  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};

    switch (toolName) {
      case 'Read':
        return `Read ${(input as any)?.file_path || (input as any)?.path || 'file'}`;
      case 'Write':
        return `Write ${(input as any)?.file_path || (input as any)?.path || 'file'}`;
      case 'Edit':
      case 'MultiEdit':
        return `Edit ${(input as any)?.file_path || (input as any)?.path || 'file'}`;
      case 'Bash': {
        const cmd = (input as any)?.command || '';
        return `Run: ${cmd.substring(0, 50)}${cmd.length > 50 ? '...' : ''}`;
      }
      case 'Glob':
        return `Find ${(input as any)?.pattern || 'files'}`;
      case 'Grep':
        return `Search "${(input as any)?.pattern || ''}"`;
      case 'Task':
        return `Task: ${(input as any)?.description || 'agent'}`;
      case 'WebSearch':
        return `Search: ${(input as any)?.query || ''}`;
      case 'WebFetch':
        return `Fetch: ${(input as any)?.url || ''}`;
      default:
        return `${toolName}`;
    }
  } catch {
    return toolName;
  }
}

/**
 * Extract facts from tool input/response
 */
export function extractFacts(toolName: string, toolInput: unknown, toolOutput?: string): string[] {
  const facts: string[] = [];

  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};
    const filePath = (input as any)?.file_path || (input as any)?.path || '';

    switch (toolName) {
      case 'Read':
        if (filePath) facts.push(`File read: ${filePath}`);
        break;
      case 'Write':
        if (filePath) facts.push(`File created/updated: ${filePath}`);
        break;
      case 'Edit':
      case 'MultiEdit':
        if (filePath) facts.push(`File modified: ${filePath}`);
        if ((input as any)?.old_string) {
          facts.push(`Code replaced in ${filePath.split(/[/\\]/).pop() || 'file'}`);
        }
        break;
      case 'Bash': {
        const cmd = (input as any)?.command || '';
        facts.push(`Command executed: ${cmd.substring(0, 100)}`);
        // Extract test results
        if (toolOutput) {
          if (toolOutput.includes('passed') || toolOutput.includes('\u2713')) facts.push('Tests passed');
          if (toolOutput.includes('failed') || toolOutput.includes('\u2717')) facts.push('Tests failed');
          if (toolOutput.includes('error') || toolOutput.includes('Error')) facts.push('Errors encountered');
        }
        break;
      }
      case 'Glob':
        if ((input as any)?.pattern) facts.push(`Pattern searched: ${(input as any).pattern}`);
        break;
      case 'Grep':
        if ((input as any)?.pattern) facts.push(`Code pattern searched: ${(input as any).pattern}`);
        if ((input as any)?.path) facts.push(`Search scope: ${(input as any).path}`);
        break;
      case 'WebSearch':
        if ((input as any)?.query) facts.push(`Web search: ${(input as any).query}`);
        break;
      case 'WebFetch':
        if ((input as any)?.url) facts.push(`URL fetched: ${(input as any).url}`);
        break;
      case 'Task':
        if ((input as any)?.description) facts.push(`Sub-task: ${(input as any).description}`);
        if ((input as any)?.subagent_type) facts.push(`Agent type: ${(input as any).subagent_type}`);
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
export function extractConcepts(toolName: string, toolInput: unknown): string[] {
  const concepts: Set<string> = new Set();

  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};
    const filePath = ((input as any)?.file_path || (input as any)?.path || '') as string;

    // Extract concepts from file paths
    if (filePath) {
      // Directory-based concepts
      const parts = filePath.split(/[/\\]/);
      for (const part of parts) {
        if (['src', 'lib', 'dist', 'node_modules', '.', '..'].includes(part)) continue;
        if (part.includes('.')) {
          // File extension concepts
          const ext = part.split('.').pop();
          const extMap: Record<string, string> = {
            ts: 'typescript', tsx: 'react', js: 'javascript', jsx: 'react',
            py: 'python', rs: 'rust', go: 'golang', css: 'styling', scss: 'styling',
            html: 'html', json: 'configuration', yaml: 'configuration', yml: 'configuration',
            md: 'documentation', test: 'testing', spec: 'testing', sql: 'database',
          };
          if (ext && extMap[ext]) concepts.add(extMap[ext]);
        }
        // Directory-based concepts
        const dirMap: Record<string, string> = {
          tests: 'testing', __tests__: 'testing', test: 'testing', spec: 'testing',
          hooks: 'hooks', api: 'api', auth: 'authentication', db: 'database',
          components: 'components', pages: 'pages', routes: 'routing', utils: 'utilities',
          services: 'services', middleware: 'middleware', models: 'models', types: 'types',
          cli: 'cli', config: 'configuration', migrations: 'database', schemas: 'schemas',
        };
        if (dirMap[part]) concepts.add(dirMap[part]);
      }
    }

    // Extract function/class names from Edit/MultiEdit for code-specific searchability
    if (toolName === 'Edit' || toolName === 'MultiEdit') {
      const oldStr = ((input as any)?.old_string || '') as string;
      const newStr = ((input as any)?.new_string || '') as string;
      const combined = oldStr + '\n' + newStr;

      // Extract function names
      const funcMatches = combined.match(/(?:function|async function|const|let|var)\s+(\w{3,})/g);
      if (funcMatches) {
        for (const m of funcMatches.slice(0, 3)) {
          const name = m.replace(/(?:function|async function|const|let|var)\s+/, '');
          concepts.add(`fn:${name}`);
        }
      }

      // Extract class names
      const classMatches = combined.match(/class\s+(\w{3,})/g);
      if (classMatches) {
        for (const m of classMatches.slice(0, 2)) {
          concepts.add(`class:${m.replace('class ', '')}`);
        }
      }

      // Extract patterns: import, export, interface, type, enum
      if (/\bimport\b/.test(combined)) concepts.add('pattern:import');
      if (/\bexport\b/.test(combined)) concepts.add('pattern:export');
      if (/\binterface\b/.test(combined)) concepts.add('pattern:interface');
      if (/\benum\b/.test(combined)) concepts.add('pattern:enum');
      if (/\btry\s*\{/.test(combined)) concepts.add('pattern:error-handling');
      if (/\basync\b/.test(combined)) concepts.add('pattern:async');
    }

    // Tool-based concepts
    switch (toolName) {
      case 'Bash': {
        const cmd = ((input as any)?.command || '') as string;
        if (cmd.includes('test') || cmd.includes('vitest') || cmd.includes('jest')) concepts.add('testing');
        if (cmd.includes('build') || cmd.includes('tsc')) concepts.add('build');
        if (cmd.includes('git')) concepts.add('version-control');
        if (cmd.includes('npm') || cmd.includes('yarn') || cmd.includes('pnpm')) concepts.add('package-management');
        if (cmd.includes('docker')) concepts.add('containerization');
        if (cmd.includes('lint') || cmd.includes('eslint')) concepts.add('linting');
        break;
      }
      case 'WebSearch':
        concepts.add('research');
        break;
      case 'WebFetch':
        concepts.add('web-content');
        break;
      case 'Task':
        concepts.add('delegation');
        if ((input as any)?.subagent_type) concepts.add((input as any).subagent_type as string);
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
export function detectIntent(toolName: string, toolInput: unknown, prompt?: string): string | null {
  const intentPatterns = {
    bugfix: /fix|bug|error|issue|broken|crash|repair/i,
    feature: /add|feature|implement|create|new|build/i,
    refactor: /refactor|clean|rename|reorganize|restructure/i,
    testing: /test|spec|coverage|verify/i,
    investigation: /find|search|investigate|debug|analyze|explore/i,
    documentation: /document|comment|readme|doc|explain/i,
  };

  // Check user prompt first (if available)
  if (prompt) {
    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(prompt)) {
        return intent;
      }
    }
  }

  // Check tool input
  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};
    const inputStr = JSON.stringify(input);

    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(inputStr)) {
        return intent;
      }
    }
  } catch {
    // Ignore parse errors
  }

  // Tool-based intent detection
  switch (toolName) {
    case 'Read':
    case 'Glob':
    case 'Grep':
      return 'investigation';
    case 'Write':
      return 'feature';
    case 'Edit':
      return null; // Too ambiguous without context
    default:
      return null;
  }
}

/**
 * Compute content hash for deduplication
 */
export function computeContentHash(...parts: string[]): string {
  return createHash('sha256')
    .update(parts.join('|'))
    .digest('hex')
    .slice(0, 16);
}

/**
 * Truncate string to max length
 */
export function truncate(str: string, maxLen: number = 500): string {
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen) + '...';
}

/**
 * Build observation from hook input
 */
export function buildObservation(input: HookInput, currentPrompt?: string): ObservationData | null {
  const toolName = input.tool_name;
  if (!toolName) return null;

  // Extract all components
  const type = getObservationType(toolName);
  const title = generateTitle(toolName, input.tool_input);
  const { filesRead, filesModified } = extractFilePaths(toolName, input.tool_input);
  const facts = extractFacts(toolName, input.tool_input, input.tool_output);
  const concepts = extractConcepts(toolName, input.tool_input);
  const intent = detectIntent(toolName, input.tool_input, currentPrompt);

  // Build summary
  const summary = buildSummary(toolName, input.tool_input, input.tool_output, filesRead, filesModified);

  // Compute content hash for dedup
  const contentHash = computeContentHash(
    toolName,
    JSON.stringify(input.tool_input || {}),
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
  toolInput: unknown,
  _toolOutput: string | undefined,
  filesRead: string[],
  filesModified: string[]
): string {
  const parts: string[] = [];

  if (filesRead.length > 0) {
    parts.push(`Read: ${filesRead.join(', ')}`);
  }
  if (filesModified.length > 0) {
    parts.push(`Modified: ${filesModified.join(', ')}`);
  }

  // Add tool-specific context
  try {
    const input = typeof toolInput === 'object' && toolInput !== null ? toolInput : {};

    if (toolName === 'Bash') {
      const cmd = (input as any)?.command || '';
      parts.push(`Command: ${truncate(cmd, 100)}`);
    } else if (toolName === 'Grep') {
      const pattern = (input as any)?.pattern || '';
      parts.push(`Pattern: ${pattern}`);
    } else if (toolName === 'Task') {
      const desc = (input as any)?.description || '';
      parts.push(`Task: ${truncate(desc, 100)}`);
    }
  } catch {
    // Ignore
  }

  return parts.length > 0 ? parts.join('; ') : `Used ${toolName}`;
}
