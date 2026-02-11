/**
 * Tests for MCP tool definitions
 */

import { describe, it, expect } from "vitest";
import { TOOLS, SESSION_TOOLS, CONTEXT_ENVIRONMENT_TOOLS, CORE_TOOLS, EXTENDED_TOOLS } from "../src/tools.js";

describe("Tool Definitions", () => {
  describe("TOOLS", () => {
    it("should export core memory tools", () => {
      const toolNames = TOOLS.map(t => t.name);

      expect(toolNames).toContain("memory_remember");
      expect(toolNames).toContain("memory_recall");
      expect(toolNames).toContain("memory_reflect");
      expect(toolNames).toContain("memory_forget");
      expect(toolNames).toContain("memory_associate");
      expect(toolNames).toContain("memory_briefing");
      expect(toolNames).toContain("memory_statistics");
      expect(toolNames).toContain("memory_graph_query");
      expect(toolNames).toContain("memory_audit");
    });

    it("should have valid input schemas", () => {
      for (const tool of TOOLS) {
        expect(tool.inputSchema).toBeDefined();
        expect(tool.inputSchema.type).toBe("object");
        expect(tool.inputSchema.properties).toBeDefined();
      }
    });

    it("should have descriptions for all tools", () => {
      for (const tool of TOOLS) {
        expect(tool.description).toBeDefined();
        expect(tool.description.length).toBeGreaterThan(10);
      }
    });
  });

  describe("SESSION_TOOLS", () => {
    it("should export session management tools", () => {
      const toolNames = SESSION_TOOLS.map(t => t.name);

      expect(toolNames).toContain("memory_session_start");
      expect(toolNames).toContain("memory_session_end");
      expect(toolNames).toContain("memory_session_commit");
      expect(toolNames).toContain("memory_session_status");
    });

    it("should have valid input schemas", () => {
      for (const tool of SESSION_TOOLS) {
        expect(tool.inputSchema).toBeDefined();
        expect(tool.inputSchema.type).toBe("object");
      }
    });

    it("should have descriptions for all tools", () => {
      for (const tool of SESSION_TOOLS) {
        expect(tool.description).toBeDefined();
        expect(tool.description.length).toBeGreaterThan(10);
      }
    });
  });

  describe("CONTEXT_ENVIRONMENT_TOOLS", () => {
    it("should export context environment tools", () => {
      const toolNames = CONTEXT_ENVIRONMENT_TOOLS.map(t => t.name);

      expect(toolNames).toContain("memory_context_exec");
      expect(toolNames).toContain("memory_context_inspect");
      expect(toolNames).toContain("memory_context_load");
      expect(toolNames).toContain("memory_context_inject");
      expect(toolNames).toContain("memory_context_query");
      expect(toolNames).toContain("memory_context_rlm");
      expect(toolNames).toContain("memory_context_status");
      expect(toolNames).toContain("memory_context_checkpoint");
    });

    it("should have valid input schemas", () => {
      for (const tool of CONTEXT_ENVIRONMENT_TOOLS) {
        expect(tool.inputSchema).toBeDefined();
        expect(tool.inputSchema.type).toBe("object");
      }
    });

    it("should have descriptions for all tools", () => {
      for (const tool of CONTEXT_ENVIRONMENT_TOOLS) {
        expect(tool.description).toBeDefined();
        expect(tool.description.length).toBeGreaterThan(10);
      }
    });
  });

  describe("memory_remember tool", () => {
    const tool = TOOLS.find(t => t.name === "memory_remember");

    it("should require content", () => {
      expect(tool?.inputSchema.required).toContain("content");
    });

    it("should have memory type enum", () => {
      const typeProperty = tool?.inputSchema.properties?.type;
      expect(typeProperty?.enum).toContain("episodic");
      expect(typeProperty?.enum).toContain("semantic");
      expect(typeProperty?.enum).toContain("procedural");
      expect(typeProperty?.enum).toContain("working");
    });

    it("should have importance range", () => {
      const importance = tool?.inputSchema.properties?.importance;
      expect(importance?.minimum).toBe(0);
      expect(importance?.maximum).toBe(1);
    });
  });

  describe("memory_recall tool", () => {
    const tool = TOOLS.find(t => t.name === "memory_recall");

    it("should require query", () => {
      expect(tool?.inputSchema.required).toContain("query");
    });

    it("should have limit constraints", () => {
      const limit = tool?.inputSchema.properties?.limit;
      expect(limit?.minimum).toBe(1);
      expect(limit?.maximum).toBe(100);
    });
  });

  describe("memory_associate tool", () => {
    const tool = TOOLS.find(t => t.name === "memory_associate");

    it("should require source_id, target_id, and relationship", () => {
      expect(tool?.inputSchema.required).toContain("source_id");
      expect(tool?.inputSchema.required).toContain("target_id");
      expect(tool?.inputSchema.required).toContain("relationship");
    });

    it("should have relationship description", () => {
      const relationship = tool?.inputSchema.properties?.relationship;
      expect(relationship?.type).toBe("string");
      expect(relationship?.description).toBeDefined();
    });
  });

  describe("memory_context_exec tool", () => {
    const tool = CONTEXT_ENVIRONMENT_TOOLS.find(t => t.name === "memory_context_exec");

    it("should require code", () => {
      expect(tool?.inputSchema.required).toContain("code");
    });

    it("should have return_result option", () => {
      const returnResult = tool?.inputSchema.properties?.return_result;
      expect(returnResult?.type).toBe("boolean");
    });
  });

  describe("memory_session_end tool", () => {
    const tool = SESSION_TOOLS.find(t => t.name === "memory_session_end");

    it("should have commit option", () => {
      expect(tool?.inputSchema.properties?.commit).toBeDefined();
      expect(tool?.inputSchema.properties?.commit?.type).toBe("boolean");
    });

    it("should have importance_threshold option", () => {
      const threshold = tool?.inputSchema.properties?.importance_threshold;
      expect(threshold?.minimum).toBe(0);
      expect(threshold?.maximum).toBe(1);
    });
  });

  describe("Legacy exports", () => {
    it("should export CORE_TOOLS as filtered subset of TOOLS", () => {
      const coreNames = CORE_TOOLS.map(t => t.name);
      expect(coreNames).toEqual([
        "memory_remember", "memory_recall", "memory_reflect",
        "memory_forget", "memory_associate"
      ]);
    });

    it("should export EXTENDED_TOOLS as filtered subset of TOOLS", () => {
      const extNames = EXTENDED_TOOLS.map(t => t.name);
      expect(extNames).toEqual([
        "memory_briefing", "memory_statistics",
        "memory_graph_query", "memory_audit"
      ]);
    });
  });
});
