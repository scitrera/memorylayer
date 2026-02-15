import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

interface ConnectionConfig {
  baseUrl: string;
  apiKey?: string;
  workspaceId?: string;
}

let clientInstance: MemoryLayerClient | null = null;
let currentConfig: ConnectionConfig | null = null;

export function getClient(config: ConnectionConfig): MemoryLayerClient {
  if (
    !clientInstance ||
    !currentConfig ||
    currentConfig.baseUrl !== config.baseUrl ||
    currentConfig.apiKey !== config.apiKey ||
    currentConfig.workspaceId !== config.workspaceId
  ) {
    clientInstance = new MemoryLayerClient({
      baseUrl: config.baseUrl,
      apiKey: config.apiKey,
      workspaceId: config.workspaceId,
      timeout: 30000,
    });
    currentConfig = { ...config };
  }
  return clientInstance;
}

export function resetClient(): void {
  clientInstance = null;
  currentConfig = null;
}
