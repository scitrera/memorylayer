/**
 * Example usage of the MemoryLayer client
 */

// @ts-ignore
import { MemoryLayerClient, MemoryType, RelationshipType } from "@scitrera/memorylayer-mcp-server";

async function main() {
  // Create client instance
  const client = new MemoryLayerClient({
    baseUrl: "http://localhost:61001",
    workspaceId: "example-workspace"
  });

  // 1. Store some memories
  console.log("1. Storing memories...");

  const problemMemory = await client.remember({
    content: "User encountered authentication timeout errors when API response takes >30s",
    type: MemoryType.EPISODIC,
    importance: 0.7,
    tags: ["authentication", "timeout", "problem"]
  });
  console.log(`Stored problem memory: ${problemMemory.id}`);

  const solutionMemory = await client.remember({
    content: "Increased authentication timeout to 60s and added retry logic with exponential backoff",
    type: MemoryType.PROCEDURAL,
    importance: 0.9,
    tags: ["authentication", "timeout", "solution"]
  });
  console.log(`Stored solution memory: ${solutionMemory.id}`);

  // 2. Create an association between problem and solution
  console.log("\n2. Creating association...");

  const association = await client.associate({
    source_id: solutionMemory.id,
    target_id: problemMemory.id,
    relationship: RelationshipType.SOLVES,
    strength: 0.95
  });
  console.log(`Created association: ${association.id}`);

  // 3. Recall memories about authentication
  console.log("\n3. Recalling memories about authentication...");

  const recallResult = await client.recall({
    query: "authentication timeout issues",
    limit: 5,
    min_relevance: 0.5
  });

  console.log(`Found ${recallResult.memories.length} memories:`);
  for (const memory of recallResult.memories) {
    console.log(`  - [${memory.type}] ${memory.content.substring(0, 80)}...`);
  }

  // 4. Reflect on authentication patterns
  console.log("\n4. Reflecting on authentication patterns...");

  const reflectResult = await client.reflect({
    query: "What have we learned about handling authentication timeouts?",
    max_tokens: 300,
    include_sources: true
  });

  console.log("Reflection:");
  console.log(reflectResult.reflection);
  console.log(`\nBased on ${reflectResult.source_memories.length} source memories`);

  // 5. Get workspace statistics
  console.log("\n5. Getting workspace statistics...");

  const stats = await client.getStatistics(true);
  console.log(`Total memories: ${stats.total_memories}`);
  console.log(`Total associations: ${stats.total_associations}`);

  if (stats.breakdown) {
    console.log("\nMemories by type:");
    for (const [type, count] of Object.entries(stats.breakdown.by_type)) {
      console.log(`  ${type}: ${count}`);
    }
  }

  // 6. Get a briefing
  console.log("\n6. Getting session briefing...");

  const briefing = await client.getBriefing(24, true);
  console.log(briefing.briefing);
}

main().catch(console.error);
