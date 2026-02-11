/**
 * AI Agent integration example
 * Shows how to use MemoryLayer with an AI agent for persistent memory
 */

import {
  MemoryLayerClient,
  MemoryType,
  MemorySubtype,
  RelationshipType,
} from "@scitrera/memorylayer-sdk";

interface Message {
  role: "user" | "assistant";
  content: string;
}

class AIAgentWithMemory {
  private client: MemoryLayerClient;
  private sessionId: string | null = null;
  private conversationHistory: Message[] = [];

  constructor(client: MemoryLayerClient) {
    this.client = client;
  }

  async initialize() {
    // Create a session for this conversation
    const session = await this.client.createSession(7200); // 2 hours
    this.sessionId = session.id;
    console.log(`Session created: ${this.sessionId}`);

    // Get briefing of recent activity
    const briefing = await this.client.getBriefing(168); // Last 7 days
    console.log("\n=== Recent Activity Briefing ===");
    console.log(briefing.recent_activity_summary);

    if (briefing.contradictions && briefing.contradictions.length > 0) {
      console.log("\n⚠️  Contradictions detected:");
      briefing.contradictions.forEach((c) => console.log(`  - ${c}`));
    }

    if (briefing.open_threads.length > 0) {
      console.log("\nOpen threads:");
      briefing.open_threads.forEach((t) => console.log(`  - ${t}`));
    }
    console.log();

    return briefing;
  }

  async processMessage(userMessage: string): Promise<string> {
    // Add user message to history
    this.conversationHistory.push({
      role: "user",
      content: userMessage,
    });

    // Store session context
    if (this.sessionId) {
      await this.client.setContext(
        this.sessionId,
        "last_message",
        userMessage
      );
    }

    // Recall relevant memories
    const recallResult = await this.client.recall(userMessage, {
      limit: 5,
      minRelevance: 0.6,
      context: this.conversationHistory,
      includeAssociations: true,
      traverseDepth: 1,
    });

    console.log(
      `Recalled ${recallResult.memories.length} relevant memories:`
    );
    recallResult.memories.forEach((mem, i) => {
      console.log(`  ${i + 1}. ${mem.content.substring(0, 80)}...`);
    });

    // In a real agent, you would:
    // 1. Pass recalled memories as context to your LLM
    // 2. Generate a response using the LLM
    // 3. Store important information from the conversation

    // Simulated response
    const response = `Based on ${recallResult.memories.length} memories, here's what I know about "${userMessage}"...`;

    this.conversationHistory.push({
      role: "assistant",
      content: response,
    });

    return response;
  }

  async rememberFact(fact: string, importance = 0.7) {
    const memory = await this.client.remember(fact, {
      type: MemoryType.SEMANTIC,
      importance,
      tags: ["conversation", "learned"],
    });
    console.log(`Remembered: ${fact}`);
    return memory;
  }

  async rememberSolution(
    problem: string,
    solution: string,
    codeExample?: string
  ) {
    // Store the problem
    const problemMemory = await this.client.remember(problem, {
      type: MemoryType.EPISODIC,
      subtype: MemorySubtype.PROBLEM,
      importance: 0.8,
      tags: ["problem", "conversation"],
    });

    // Store the solution
    const solutionMemory = await this.client.remember(solution, {
      type: MemoryType.PROCEDURAL,
      subtype: MemorySubtype.SOLUTION,
      importance: 0.9,
      tags: ["solution", "conversation"],
      metadata: codeExample ? { code: codeExample } : {},
    });

    // Create association
    await this.client.associate(
      solutionMemory.id,
      problemMemory.id,
      RelationshipType.SOLVES,
      0.95
    );

    console.log(`Remembered solution for: ${problem}`);

    return { problemMemory, solutionMemory };
  }

  async rememberPreference(preference: string) {
    const memory = await this.client.remember(preference, {
      type: MemoryType.SEMANTIC,
      subtype: MemorySubtype.PREFERENCE,
      importance: 0.85,
      tags: ["preference", "user"],
    });
    console.log(`Remembered preference: ${preference}`);
    return memory;
  }

  async reflect(query: string) {
    const reflection = await this.client.reflect(query, {
      maxTokens: 1000,
      depth: 2,
      includeSources: true,
    });

    console.log(`\n=== Reflection on: "${query}" ===`);
    console.log(reflection.reflection);
    console.log(
      `\nBased on ${reflection.source_memories.length} memories, ${reflection.tokens_processed} tokens processed`
    );

    return reflection;
  }

  async cleanup() {
    if (this.sessionId) {
      await this.client.deleteSession(this.sessionId);
      console.log(`\nSession ${this.sessionId} cleaned up`);
    }
  }
}

async function main() {
  const client = new MemoryLayerClient({
    baseUrl: process.env.MEMORYLAYER_BASE_URL || "http://localhost:8080",
    apiKey: process.env.MEMORYLAYER_API_KEY,
    workspaceId: process.env.MEMORYLAYER_WORKSPACE_ID,
  });

  const agent = new AIAgentWithMemory(client);

  console.log("=== AI Agent with Memory Demo ===\n");

  // Initialize agent
  await agent.initialize();

  // Simulate a conversation
  console.log("User: How do I implement retry logic?");
  await agent.processMessage("How do I implement retry logic?");

  // Agent learns a solution
  await agent.rememberSolution(
    "Need to implement retry logic for API calls",
    "Use exponential backoff with max retry limit of 3",
    `
async function fetchWithRetry(url, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fetch(url);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await sleep(1000 * Math.pow(2, i));
    }
  }
}
    `.trim()
  );

  // Learn a preference
  await agent.rememberPreference(
    "User prefers TypeScript over JavaScript for new projects"
  );

  // Learn a fact
  await agent.rememberFact(
    "The system uses PostgreSQL for primary database and Redis for caching"
  );

  // Later conversation
  console.log("\n--- Later in conversation ---\n");
  console.log("User: What was that retry pattern you mentioned?");
  await agent.processMessage("What was that retry pattern you mentioned?");

  // Reflect on learnings
  await agent.reflect("What have we discussed about code patterns?");

  // Cleanup
  await agent.cleanup();

  console.log("\n=== Demo Complete ===");
}

main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
