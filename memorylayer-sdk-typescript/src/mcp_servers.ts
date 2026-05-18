import type { MemoryLayerClient } from "./client.js";
import type { AuthorityContext } from "./types.js";

// ------------------------------------------------------------------ //
// MCP Server types
// ------------------------------------------------------------------ //

export interface McpServer {
  id: string;
  tenant_id: string;
  workspace_id: string;
  user_id?: string;
  name: string;
  description?: string;
  transport: string;
  command?: string;
  args: string[];
  env: Record<string, string>;
  url?: string;
  headers: Record<string, string>;
  metadata: Record<string, unknown>;
  source_mode: string;
  manifest_hash: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface McpServerCreateOptions {
  name: string;
  transport: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  description?: string;
  metadata?: Record<string, unknown>;
  source_mode?: string;
  enabled?: boolean;
  workspaceId?: string;
  userId?: string;
  authority?: AuthorityContext;
}

export interface McpServerUpdateOptions {
  description?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  metadata?: Record<string, unknown>;
  source_mode?: string;
  enabled?: boolean;
  authority?: AuthorityContext;
}

export interface McpServerListOptions {
  workspaceId?: string;
  userId?: string;
  name?: string;
  transport?: string;
  enabled?: boolean;
  limit?: number;
  offset?: number;
  authority?: AuthorityContext;
}

export interface McpServerResolveOptions {
  name: string;
  workspaceId?: string;
  authority?: AuthorityContext;
}

// ------------------------------------------------------------------ //
// McpServersNamespace — attached as client.mcpServers
// ------------------------------------------------------------------ //

export class McpServersNamespace {
  private _client: MemoryLayerClient;
  private _authority?: AuthorityContext;
  private _workspaceId?: string;

  constructor(
    client: MemoryLayerClient,
    authority?: AuthorityContext,
    workspaceId?: string,
  ) {
    this._client = client;
    this._authority = authority;
    this._workspaceId = workspaceId;
  }

  /** Return a scoped proxy that injects OBO headers and workspace on every call. */
  withAuthority(
    authority: AuthorityContext,
    workspaceId?: string,
  ): McpServersNamespace {
    return new McpServersNamespace(
      this._client,
      authority,
      workspaceId ?? this._workspaceId,
    );
  }

  private _req<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T> {
    return (this._client as unknown as ClientInternal)._mcpServersRequest<T>(
      method,
      path,
      body,
      authority ?? this._authority,
    );
  }

  async list(options: McpServerListOptions = {}): Promise<McpServer[]> {
    const params = new URLSearchParams();
    const wsId = options.workspaceId ?? this._workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options.userId) params.set("user_id", options.userId);
    if (options.name) params.set("name", options.name);
    if (options.transport) params.set("transport", options.transport);
    if (options.enabled !== undefined) params.set("enabled", String(options.enabled));
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    const response = await this._req<{ mcp_servers: McpServer[] }>(
      "GET",
      `/v1/mcp-servers${query ? `?${query}` : ""}`,
      undefined,
      options.authority,
    );
    return response.mcp_servers;
  }

  async get(serverId: string, authority?: AuthorityContext): Promise<McpServer> {
    const response = await this._req<{ mcp_server: McpServer }>(
      "GET",
      `/v1/mcp-servers/${serverId}`,
      undefined,
      authority,
    );
    return response.mcp_server;
  }

  async create(options: McpServerCreateOptions): Promise<McpServer> {
    const body: Record<string, unknown> = {
      name: options.name,
      transport: options.transport,
      args: options.args ?? [],
      env: options.env ?? {},
      headers: options.headers ?? {},
      metadata: options.metadata ?? {},
      source_mode: options.source_mode ?? "server",
      enabled: options.enabled ?? true,
    };
    if (options.command !== undefined) body.command = options.command;
    if (options.url !== undefined) body.url = options.url;
    if (options.description !== undefined) body.description = options.description;
    if (options.workspaceId ?? this._workspaceId) body.workspace_id = options.workspaceId ?? this._workspaceId;
    if (options.userId) body.user_id = options.userId;
    const response = await this._req<{ mcp_server: McpServer }>(
      "POST",
      "/v1/mcp-servers",
      body,
      options.authority,
    );
    return response.mcp_server;
  }

  async update(serverId: string, options: McpServerUpdateOptions = {}): Promise<McpServer> {
    const body: Record<string, unknown> = {};
    if (options.description !== undefined) body.description = options.description;
    if (options.command !== undefined) body.command = options.command;
    if (options.args !== undefined) body.args = options.args;
    if (options.env !== undefined) body.env = options.env;
    if (options.url !== undefined) body.url = options.url;
    if (options.headers !== undefined) body.headers = options.headers;
    if (options.metadata !== undefined) body.metadata = options.metadata;
    if (options.source_mode !== undefined) body.source_mode = options.source_mode;
    if (options.enabled !== undefined) body.enabled = options.enabled;
    const response = await this._req<{ mcp_server: McpServer }>(
      "PATCH",
      `/v1/mcp-servers/${serverId}`,
      body,
      options.authority,
    );
    return response.mcp_server;
  }

  async delete(serverId: string, authority?: AuthorityContext): Promise<void> {
    await this._req<void>(
      "DELETE",
      `/v1/mcp-servers/${serverId}`,
      undefined,
      authority,
    );
  }

  async resolve(options: McpServerResolveOptions): Promise<McpServer | null> {
    const body: Record<string, unknown> = { name: options.name };
    const wsId = options.workspaceId ?? this._workspaceId;
    if (wsId) body.workspace_id = wsId;
    const response = await this._req<{ mcp_server: McpServer | null }>(
      "POST",
      "/v1/mcp-servers/resolve",
      body,
      options.authority,
    );
    return response.mcp_server;
  }
}

// Internal interface to access the private request method from MemoryLayerClient.
interface ClientInternal {
  _mcpServersRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T>;
}
