/**
 * Basic usage example for MemoryLayer.ai TypeScript SDK
 */

import { MemoryLayerClient, MemoryType, MemorySubtype } from "@scitrera/memorylayer-sdk";

async function main() {
  // Initialize the client
  const client = new MemoryLayerClient({
    baseUrl: process.env.MEMORYLAYER_BASE_URL || "http://localhost:8080",
    apiKey: process.env.MEMORYLAYER_API_KEY,
    workspaceId: process.env.MEMORYLAYER_WORKSPACE_ID,
  });

  console.log("=== Basic MemoryLayer Usage ===\n");

  // 1. Store some memories
  console.log("1. Storing memories...");

  const memory1 = await client.remember(
    "User prefers dark mode in the application UI",
    {
      type: MemoryType.SEMANTIC,
      subtype: MemorySubtype.PREFERENCE,
      importance: 0.8,
      tags: ["ui", "preferences"],
      metadata: { category: "appearance" },
    }
  );
  console.log(`Stored memory: ${memory1.id}`);

  const memory2 = await client.remember(
    "Fixed authentication timeout issue by increasing retry limit to 3",
    {
      type: MemoryType.PROCEDURAL,
      subtype: MemorySubtype.FIX,
      importance: 0.9,
      tags: ["auth", "bug-fix"],
      metadata: { file: "src/auth.ts", severity: "high" },
    }
  );
  console.log(`Stored memory: ${memory2.id}`);

  const memory3 = await client.remember(
    "API rate limit is 1000 requests per hour",
    {
      type: MemoryType.SEMANTIC,
      importance: 0.7,
      tags: ["api", "limits"],
    }
  );
  console.log(`Stored memory: ${memory3.id}\n`);

  // 2. Recall memories
  console.log("2. Recalling memories...");

  const recallResult = await client.recall("What are the user preferences?", {
    limit: 5,
    minRelevance: 0.5,
  });

  console.log(`Found ${recallResult.memories.length} relevant memories:`);
  recallResult.memories.forEach((mem) => {
    console.log(`  - ${mem.content.substring(0, 60)}...`);
  });
  console.log(
    `Search took ${recallResult.search_latency_ms}ms using ${recallResult.mode_used} mode\n`
  );

  // 3. Update a memory
  console.log("3. Updating memory importance...");

  const updated = await client.updateMemory(memory2.id, {
    importance: 0.95,
    tags: ["auth", "bug-fix", "critical"],
  });
  console.log(
    `Updated memory ${updated.id} - new importance: ${updated.importance}\n`
  );

  // 4. Get a specific memory
  console.log("4. Retrieving specific memory...");

  const retrieved = await client.getMemory(memory1.id);
  console.log(`Retrieved: ${retrieved.content}`);
  console.log(`Access count: ${retrieved.access_count}\n`);

  // 5. Reflect on memories
  console.log("5. Reflecting on memories...");

  const reflection = await client.reflect(
    "What do we know about authentication?",
    {
      maxTokens: 500,
      includeSources: true,
    }
  );
  console.log(`Reflection: ${reflection.reflection}`);
  console.log(
    `Based on ${reflection.source_memories.length} source memories\n`
  );

  // 6. Search with filters
  console.log("6. Searching with filters...");

  const filteredResult = await client.recall("authentication", {
    tags: ["auth"],
    types: [MemoryType.PROCEDURAL],
    limit: 10,
  });
  console.log(
    `Found ${filteredResult.memories.length} procedural auth memories\n`
  );

  // 7. Apply decay
  console.log("7. Applying decay to memory...");

  const decayed = await client.decay(memory3.id, 0.1);
  console.log(
    `Memory ${decayed.id} decay factor: ${decayed.decay_factor.toFixed(3)}\n`
  );

  // 8. Clean up (soft delete)
  console.log("8. Archiving memory...");

  await client.forget(memory3.id);
  console.log(`Memory ${memory3.id} archived\n`);

  console.log("=== Demo Complete ===");
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
