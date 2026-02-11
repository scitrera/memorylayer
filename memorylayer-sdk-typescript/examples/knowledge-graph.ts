/**
 * Knowledge graph example
 * Demonstrates building and traversing a knowledge graph with associations
 */

import {
  MemoryLayerClient,
  MemoryType,
  MemorySubtype,
  RelationshipType,
} from "@scitrera/memorylayer-sdk";

async function main() {
  const client = new MemoryLayerClient({
    baseUrl: process.env.MEMORYLAYER_BASE_URL || "http://localhost:8080",
    apiKey: process.env.MEMORYLAYER_API_KEY,
    workspaceId: process.env.MEMORYLAYER_WORKSPACE_ID,
  });

  console.log("=== Knowledge Graph Demo ===\n");

  // 1. Create a knowledge graph of related concepts
  console.log("1. Building knowledge graph...\n");

  // Core problem
  const authProblem = await client.remember(
    "Users experiencing authentication timeouts during login",
    {
      type: MemoryType.EPISODIC,
      subtype: MemorySubtype.PROBLEM,
      importance: 0.95,
      tags: ["auth", "bug", "production"],
    }
  );

  // Root cause
  const rootCause = await client.remember(
    "Database connection pool exhausted under high load",
    {
      type: MemoryType.SEMANTIC,
      subtype: MemorySubtype.ERROR,
      importance: 0.9,
      tags: ["database", "performance"],
    }
  );

  // Solutions
  const solution1 = await client.remember(
    "Increase connection pool size from 10 to 50",
    {
      type: MemoryType.PROCEDURAL,
      subtype: MemorySubtype.SOLUTION,
      importance: 0.8,
      tags: ["database", "configuration"],
    }
  );

  const solution2 = await client.remember(
    "Implement connection pooling with retry logic",
    {
      type: MemoryType.PROCEDURAL,
      subtype: MemorySubtype.SOLUTION,
      importance: 0.85,
      tags: ["database", "code-pattern"],
    }
  );

  const solution3 = await client.remember(
    "Add request queuing with rate limiting",
    {
      type: MemoryType.PROCEDURAL,
      subtype: MemorySubtype.SOLUTION,
      importance: 0.75,
      tags: ["performance", "architecture"],
    }
  );

  // Related context
  const context = await client.remember(
    "Production environment handles 10,000 concurrent users during peak hours",
    {
      type: MemoryType.SEMANTIC,
      importance: 0.7,
      tags: ["production", "metrics"],
    }
  );

  // Similar past issue
  const similarIssue = await client.remember(
    "Similar timeout issue occurred in staging 2 months ago",
    {
      type: MemoryType.EPISODIC,
      subtype: MemorySubtype.PROBLEM,
      importance: 0.6,
      tags: ["staging", "historical"],
    }
  );

  // 2. Create associations to build the graph
  console.log("2. Creating associations...\n");

  // Problem -> Root cause
  await client.associate(
    rootCause.id,
    authProblem.id,
    RelationshipType.CAUSES,
    0.95
  );

  // Solutions -> Problem
  await client.associate(
    solution1.id,
    authProblem.id,
    RelationshipType.SOLVES,
    0.85
  );
  await client.associate(
    solution2.id,
    authProblem.id,
    RelationshipType.SOLVES,
    0.9
  );
  await client.associate(
    solution3.id,
    authProblem.id,
    RelationshipType.SOLVES,
    0.7
  );

  // Solution dependencies
  await client.associate(
    solution2.id,
    solution1.id,
    RelationshipType.BUILDS_ON,
    0.8
  );

  // Solution alternatives
  await client.associate(
    solution3.id,
    solution2.id,
    RelationshipType.ALTERNATIVE_TO,
    0.6
  );

  // Context
  await client.associate(
    authProblem.id,
    context.id,
    RelationshipType.OCCURS_IN,
    0.8
  );

  // Historical connection
  await client.associate(
    authProblem.id,
    similarIssue.id,
    RelationshipType.SIMILAR_TO,
    0.7
  );

  console.log("Knowledge graph created!\n");

  // 3. Traverse the graph
  console.log("3. Traversing knowledge graph...\n");

  // Get all associations for the auth problem
  const associations = await client.getAssociations(authProblem.id, "both");
  console.log(
    `Auth problem has ${associations.length} direct associations:`
  );
  associations.forEach((assoc) => {
    console.log(
      `  - ${assoc.relationship} (strength: ${assoc.strength.toFixed(2)})`
    );
  });
  console.log();

  // 4. Recall with graph traversal
  console.log("4. Recalling with graph traversal...\n");

  // Shallow traversal
  const shallowResult = await client.recall("authentication timeout", {
    includeAssociations: true,
    traverseDepth: 1,
    limit: 20,
  });
  console.log(
    `Shallow search (depth 1): ${shallowResult.memories.length} memories`
  );

  // Deep traversal
  const deepResult = await client.recall("authentication timeout", {
    includeAssociations: true,
    traverseDepth: 3,
    limit: 20,
  });
  console.log(
    `Deep search (depth 3): ${deepResult.memories.length} memories\n`
  );

  // 5. Get solutions for the problem
  console.log("5. Finding all solutions...\n");

  const solutionAssocs = associations.filter(
    (a) => a.relationship === RelationshipType.SOLVES
  );

  console.log(`Found ${solutionAssocs.length} solutions:`);
  for (const assoc of solutionAssocs) {
    const solution = await client.getMemory(assoc.source_id);
    console.log(
      `  - ${solution.content} (strength: ${assoc.strength.toFixed(2)})`
    );
  }
  console.log();

  // 6. Reflect on the problem
  console.log("6. Reflecting on the issue...\n");

  const reflection = await client.reflect(
    "What do we know about authentication timeout issues and their solutions?",
    {
      maxTokens: 1000,
      depth: 3,
      includeSources: true,
    }
  );

  console.log("Reflection:");
  console.log(reflection.reflection);
  console.log(`\nSources: ${reflection.source_memories.length} memories\n`);

  // 7. Find related historical issues
  console.log("7. Finding related historical issues...\n");

  const historicalResult = await client.recall("timeout issues", {
    tags: ["historical", "staging", "production"],
    includeAssociations: true,
    traverseDepth: 2,
  });

  console.log(`Found ${historicalResult.memories.length} related issues:`);
  historicalResult.memories.forEach((mem) => {
    console.log(`  - ${mem.content}`);
    console.log(`    Tags: ${mem.tags.join(", ")}`);
  });

  console.log("\n=== Demo Complete ===");
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
