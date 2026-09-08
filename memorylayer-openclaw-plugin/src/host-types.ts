/** Memory contracts derived from OpenClaw's public capability export. */
import type { MemoryPluginCapability } from "openclaw/plugin-sdk/memory-host-core";

export type MemoryPluginRuntime = NonNullable<MemoryPluginCapability["runtime"]>;
export type MemorySearchManager = NonNullable<
  Awaited<ReturnType<MemoryPluginRuntime["getMemorySearchManager"]>>["manager"]
>;
export type MemorySearchResult = Awaited<ReturnType<MemorySearchManager["search"]>>[number];
export type MemoryReadResult = Awaited<ReturnType<MemorySearchManager["readFile"]>>;
export type MemoryProviderStatus = ReturnType<MemorySearchManager["status"]>;
export type MemoryEmbeddingProbeResult = Awaited<ReturnType<MemorySearchManager["probeEmbeddingAvailability"]>>;
