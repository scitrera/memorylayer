/**
 * MemoryLayer client wired to the per-sandbox Aether sidecar relay.
 *
 * The sidecar listens on plaintext localhost (no credentials needed) and
 * relays upstream with the proper Aether identity injected. We dial it with
 * a vanilla `AetherClient` and wrap an `AetherFetchTransport` around the
 * MemoryLayer target topic. The MemoryLayer TS SDK's `fetch` injection
 * point (added alongside this plugin) then routes every HTTP-shaped call
 * through the Aether transport.
 */

import { AetherClient, AetherFetchTransport } from "@scitrera/aether-client";
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

import type { PluginConfig } from "./config.js";

export interface ConfiguredMemoryLayer {
  client: MemoryLayerClient;
  /** The underlying Aether client. Owned by the plugin; closed on shutdown. */
  aether: AetherClient;
}

export async function createMemoryLayerClient(
  config: PluginConfig,
): Promise<ConfiguredMemoryLayer> {
  const aether = new AetherClient({
    address: config.aetherAddress,
    // No credentials — sidecar terminates the relay locally and injects
    // identity upstream. See [[project-workclaw-platform-shift]].
  });
  await aether.connect();

  const transport = new AetherFetchTransport(
    aether,
    config.memorylayerTargetTopic,
    config.memorylayerTimeoutMs,
  );

  const client = new MemoryLayerClient({
    baseUrl: config.memorylayerBaseUrl,
    workspaceId: config.defaultWorkspaceId,
    timeout: config.memorylayerTimeoutMs,
    fetch: transport.fetch.bind(transport),
    defaultAuthority: config.grantId
      ? {
          grantId: config.grantId,
          subject: { type: config.principalType, id: config.principalId },
        }
      : undefined,
  });

  return { client, aether };
}
