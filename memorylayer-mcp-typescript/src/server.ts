/**
 * MCP server implementation for MemoryLayer.ai
 */

import {Server} from "@modelcontextprotocol/sdk/server/index.js";
import {StdioServerTransport} from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
    Tool
} from "@modelcontextprotocol/sdk/types.js";
import {writeFileSync, mkdirSync, existsSync, renameSync} from "fs";
import {join} from "path";
import {tmpdir} from "os";

import {MemoryLayerClient} from "./client.js";
import {MCPToolHandlers} from "./handlers.js";
import {
    ToolProfile,
    DEFAULT_PROFILE,
    getToolsForProfile,
    isToolEnabled,
} from "./tools.js";
import {SessionManager} from "./session.js";

/**
 * Format an error for display, handling various error types.
 * Prevents "[object Object]" from appearing in error messages.
 */
function formatError(error: unknown): string {
    // Standard Error object - check for details (e.g., MemoryLayerError from SDK)
    if (error instanceof Error) {
        const errWithDetails = error as Error & { details?: unknown; statusCode?: number };
        if (errWithDetails.details) {
            const details = typeof errWithDetails.details === 'string'
                ? errWithDetails.details
                : JSON.stringify(errWithDetails.details);
            return `${error.message}: ${details}`;
        }
        return error.message;
    }

    // String error
    if (typeof error === 'string') {
        return error;
    }

    // Object with common error properties
    if (error && typeof error === 'object') {
        const errorObj = error as Record<string, unknown>;

        // Check for common error message properties
        if (typeof errorObj.message === 'string') {
            return errorObj.message;
        }
        if (typeof errorObj.error === 'string') {
            return errorObj.error;
        }
        if (typeof errorObj.detail === 'string') {
            return errorObj.detail;
        }

        // API error response with nested error object
        if (errorObj.error && typeof errorObj.error === 'object') {
            const nestedError = errorObj.error as Record<string, unknown>;
            if (typeof nestedError.message === 'string') {
                return nestedError.message;
            }
            if (typeof nestedError.detail === 'string') {
                return nestedError.detail;
            }
        }

        // Fallback: stringify the object
        try {
            return JSON.stringify(error);
        } catch {
            return 'Unknown error (could not serialize)';
        }
    }

    // Fallback for other types
    return String(error);
}

export interface MCPServerOptions {
    workspaceId?: string;

    /**
     * Tool profile to use. Determines which tools are exposed.
     *
     * - "cc" (default): Claude Code profile - 9 essential tools for agent memory
     * - "full": All 18 tools enabled - for power users
     * - "minimal": Just remember/recall - absolute minimum
     *
     * Can also be set via MEMORYLAYER_TOOL_PROFILE env var.
     */
    toolProfile?: ToolProfile;

    /** Auto-start session when MCP server connects (default: true).
     *  Can be disabled via MEMORYLAYER_AUTO_START_SESSION=false env var. */
    autoStartSession?: boolean;

    // Legacy options (deprecated, use toolProfile instead)
    /** @deprecated Use toolProfile: "full" instead */
    extendedTools?: boolean;
    /** @deprecated Session mode is always enabled; use toolProfile to control session tools */
    sessionMode?: boolean;
}

export class MCPServer {
    private server: Server;
    private handlers: MCPToolHandlers;
    private sessionManager: SessionManager;
    private toolProfile: ToolProfile;
    private autoStartSession: boolean;

    constructor(
        client: MemoryLayerClient,
        options: MCPServerOptions = {}
    ) {
        // Determine tool profile (env var > option > legacy options > default)
        const envProfile = process.env.MEMORYLAYER_TOOL_PROFILE as ToolProfile | undefined;
        if (envProfile && !["cc", "full", "minimal"].includes(envProfile)) {
            console.error(`Invalid MEMORYLAYER_TOOL_PROFILE: "${envProfile}", using default "${DEFAULT_PROFILE}"`);
        }
        const validEnvProfile = envProfile && ["cc", "full", "minimal"].includes(envProfile) ? envProfile : undefined;

        // Handle legacy options for backwards compatibility
        let legacyProfile: ToolProfile | undefined;
        if (options.extendedTools === true) {
            console.error("Warning: extendedTools option is deprecated, use toolProfile: 'full' instead");
            legacyProfile = "full";
        }

        this.toolProfile = validEnvProfile ?? options.toolProfile ?? legacyProfile ?? DEFAULT_PROFILE;

        // Session mode is always enabled (session tools controlled by profile)
        const sessionModeEnabled = true;

        // Auto-start session: enabled by default
        // Can be disabled via env var MEMORYLAYER_AUTO_START_SESSION=false
        const envAutoStart = process.env.MEMORYLAYER_AUTO_START_SESSION;
        const envAutoStartDisabled = envAutoStart?.toLowerCase() === 'false' || envAutoStart === '0';
        this.autoStartSession = options.autoStartSession ?? !envAutoStartDisabled;

        this.sessionManager = new SessionManager({enabled: sessionModeEnabled});

        this.server = new Server(
            {
                name: "memorylayer",
                version: require("../package.json").version,
            },
            {
                capabilities: {
                    tools: {}
                }
            }
        );

        this.handlers = new MCPToolHandlers(client, this.sessionManager);
        this.setupHandlers();

        console.error(`MemoryLayer MCP server initialized with profile: "${this.toolProfile}"`);
    }

    private setupHandlers(): void {
        // List tools based on the configured profile
        this.server.setRequestHandler(ListToolsRequestSchema, async () => {
            const profileTools = getToolsForProfile(this.toolProfile);

            const tools: Tool[] = profileTools.map(toolDef => ({
                name: toolDef.name,
                description: toolDef.description,
                inputSchema: toolDef.inputSchema as unknown as {
                    type: "object";
                    properties?: { [x: string]: object } | undefined;
                    required?: string[] | undefined;
                }
            }));

            return {tools};
        });

        // Handle tool invocations
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const {name, arguments: args} = request.params;

            try {
                // Check if tool is enabled for current profile
                if (!isToolEnabled(name, this.toolProfile)) {
                    const errorMsg = `Tool "${name}" is not available in profile "${this.toolProfile}". Use MEMORYLAYER_TOOL_PROFILE=full to enable all tools.`;
                    console.error(errorMsg);
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({error: errorMsg, profile: this.toolProfile})
                            }
                        ],
                        isError: true
                    };
                }

                // Route to appropriate handler
                const handlerMethodName = name.replace(/_([a-z])/g, (_, letter) =>
                    letter.toUpperCase()
                );
                const handlerMethod = `handle${handlerMethodName.charAt(0).toUpperCase()}${handlerMethodName.slice(1)}`;

                if (!(handlerMethod in this.handlers)) {
                    const errorMsg = `Unknown tool: ${name}`;
                    console.error(errorMsg);
                    return {
                        content: [
                            {
                                type: "text",
                                text: JSON.stringify({error: errorMsg})
                            }
                        ]
                    };
                }

                // Call the handler
                const handler = (this.handlers as unknown as Record<string, (args: Record<string, unknown>) => Promise<string>>)[handlerMethod];
                const result = await handler.call(this.handlers, args || {});

                return {
                    content: [
                        {
                            type: "text",
                            text: result
                        }
                    ]
                };
            } catch (error) {
                const errorMsg = formatError(error);
                console.error(`Tool execution failed for ${name}:`, errorMsg);

                return {
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify({
                                success: false,
                                error: errorMsg
                            }, null, 2)
                        }
                    ],
                    isError: true
                };
            }
        });
    }

    async run(): Promise<void> {
        console.error("Starting MemoryLayer MCP server on stdio transport");

        const transport = new StdioServerTransport();
        await this.server.connect(transport);

        console.error("MCP server running");

        // Auto-start session if enabled
        if (this.autoStartSession) {
            try {
                const result = await this.handlers.handleMemorySessionStart({});
                console.error("Session auto-started:", result);

                // Write session handoff file for hook to adopt
                const session = this.sessionManager.currentSession;
                if (session?.serverSessionId) {
                    this.writeSessionHandoff(session.serverSessionId, session.workspaceId);
                }
            } catch (error) {
                console.error("Failed to auto-start session:", error instanceof Error ? error.message : error);
            }
        }
    }

    /**
     * Write session handoff file for hook to adopt the MCP-created session
     */
    private writeSessionHandoff(sessionId: string, workspaceId: string): void {
        try {
            const handoffDir = join(tmpdir(), "memorylayer");
            const handoffFile = join(handoffDir, `session-${process.ppid}.json`);

            if (!existsSync(handoffDir)) {
                mkdirSync(handoffDir, {recursive: true});
            }

            const handoff = {
                sessionId,
                workspaceId,
                pid: process.pid,
                createdAt: new Date().toISOString()
            };

            // Write atomically: tmp file then rename
            const tmpFile = handoffFile + ".tmp";
            writeFileSync(tmpFile, JSON.stringify(handoff));
            renameSync(tmpFile, handoffFile);

            console.error(`Session handoff written: ${handoffFile}`);
        } catch (error) {
            console.error("Failed to write session handoff:", error instanceof Error ? error.message : error);
        }
    }

    getManifest(): Record<string, unknown> {
        const profileTools = getToolsForProfile(this.toolProfile);
        return {
            name: "memorylayer",
            version: require("../package.json").version,
            description: "MemoryLayer.ai - Memory infrastructure for LLM-powered agents",
            homepage: "https://memorylayer.ai",
            toolProfile: this.toolProfile,
            capabilities: {
                tools: profileTools.map(t => t.name)
            }
        };
    }
}

export async function createServer(
    client: MemoryLayerClient,
    options: MCPServerOptions = {}
): Promise<MCPServer> {
    return new MCPServer(client, options);
}
