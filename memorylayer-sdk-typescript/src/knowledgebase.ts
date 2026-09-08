import type { MemoryLayerClient } from "./client.js";
import type {
  AuthorityContext,
  GraphAnalysis,
  GraphAnalysisResponse,
  GraphCommunity,
  KBArticle,
  KBArticleListOptions,
  KBArticleListResponse,
  KBGenerateOptions,
  Knowledgebase,
} from "./types.js";

// ------------------------------------------------------------------ //
// KnowledgebaseNamespace — attached as client.kb
//
// Routes under /v1/knowledgebase*. The knowledgebase surface synthesizes a
// graph analysis of a workspace into Obsidian-compatible Markdown articles.
// ------------------------------------------------------------------ //

export class KnowledgebaseNamespace {
  private _client: MemoryLayerClient;
  private _authority?: AuthorityContext;

  constructor(client: MemoryLayerClient, authority?: AuthorityContext) {
    this._client = client;
    this._authority = authority;
  }

  /** Return a scoped proxy that injects OBO headers on every call. */
  withAuthority(authority: AuthorityContext): KnowledgebaseNamespace {
    return new KnowledgebaseNamespace(this._client, authority);
  }

  private _req<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
  ): Promise<T> {
    return (this._client as unknown as ClientInternal)._kbRequest<T>(
      method,
      path,
      body,
      authority ?? this._authority,
    );
  }

  /**
   * Get knowledgebase metadata for the current workspace.
   * Does NOT trigger regeneration. 404 if none generated yet.
   */
  async get(authority?: AuthorityContext): Promise<Knowledgebase> {
    return this._req<Knowledgebase>("GET", "/v1/knowledgebase", undefined, authority);
  }

  /**
   * Trigger knowledgebase generation for a workspace.
   * Runs graph analysis, community labeling, entity deep-dives, and produces
   * Markdown articles. Returns the resulting KB metadata.
   */
  async generate(options: KBGenerateOptions = {}, authority?: AuthorityContext): Promise<Knowledgebase> {
    const body: Record<string, unknown> = {
      include_rpg: options.includeRpg ?? false,
      max_communities: options.maxCommunities ?? 50,
      max_god_nodes: options.maxGodNodes ?? 20,
      regenerate: options.regenerate ?? false,
    };
    if (options.workspaceId !== undefined) body.workspace_id = options.workspaceId;
    if (options.contextId !== undefined) body.context_id = options.contextId;
    return this._req<Knowledgebase>("POST", "/v1/knowledgebase/generate", body, authority);
  }

  /** List knowledgebase articles, optionally filtered by type. */
  async listArticles(options: KBArticleListOptions = {}, authority?: AuthorityContext): Promise<KBArticle[]> {
    const params = new URLSearchParams();
    if (options.articleType) params.set("article_type", options.articleType);
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    if (options.offset !== undefined) params.set("offset", String(options.offset));
    const query = params.toString();
    const response = await this._req<KBArticleListResponse>(
      "GET",
      `/v1/knowledgebase/articles${query ? `?${query}` : ""}`,
      undefined,
      authority,
    );
    return response.articles;
  }

  /**
   * Get a single article by ID. Article IDs: `index`, `community-{n}`,
   * `entity-{slug}`.
   */
  async getArticle(articleId: string, authority?: AuthorityContext): Promise<KBArticle> {
    return this._req<KBArticle>(
      "GET",
      `/v1/knowledgebase/articles/${articleId}`,
      undefined,
      authority,
    );
  }

  /**
   * Download the knowledgebase as an Obsidian-compatible vault zip.
   * Returns the raw bytes as an ArrayBuffer.
   */
  async exportVault(authority?: AuthorityContext): Promise<ArrayBuffer> {
    return (this._client as unknown as ClientInternal)._kbRequest<ArrayBuffer>(
      "GET",
      "/v1/knowledgebase/export",
      undefined,
      authority ?? this._authority,
      "arraybuffer",
    );
  }

  /**
   * Get a fresh graph analysis (snapshot + communities + centrality + bridges
   * + stats) for the current workspace.
   */
  async getGraphAnalysis(authority?: AuthorityContext): Promise<GraphAnalysis | null> {
    const response = await this._req<GraphAnalysisResponse>(
      "GET",
      "/v1/knowledgebase/graph",
      undefined,
      authority,
    );
    return response.analysis ?? null;
  }

  /** Get detailed information about a single community by ID. */
  async getCommunity(communityId: number, authority?: AuthorityContext): Promise<GraphCommunity> {
    return this._req<GraphCommunity>(
      "GET",
      `/v1/knowledgebase/graph/communities/${communityId}`,
      undefined,
      authority,
    );
  }
}

// Internal interface to access the private request method from MemoryLayerClient.
interface ClientInternal {
  _kbRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
    responseMode?: "json" | "text" | "arraybuffer",
  ): Promise<T>;
}
