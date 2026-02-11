/**
 * Tests for SessionManager
 */

import { describe, it, expect, beforeEach } from "vitest";
import { SessionManager } from "../src/session.js";

describe("SessionManager", () => {
  let manager: SessionManager;

  beforeEach(() => {
    manager = new SessionManager();
  });

  describe("constructor", () => {
    it("should be enabled by default", () => {
      expect(manager.isEnabled).toBe(true);
    });

    it("should respect enabled config", () => {
      const disabled = new SessionManager({ enabled: false });
      expect(disabled.isEnabled).toBe(false);
    });

    it("should have no active session initially", () => {
      expect(manager.hasActiveSession).toBe(false);
      expect(manager.currentSession).toBeNull();
    });
  });

  describe("startSession", () => {
    it("should create a new session", () => {
      const session = manager.startSession("test-workspace");

      expect(session).toBeDefined();
      expect(session.id).toMatch(/^local_\d+_[a-z0-9]+$/);
      expect(session.workspaceId).toBe("test-workspace");
      expect(session.committed).toBe(false);
      expect(session.workingMemory.size).toBe(0);
    });

    it("should set hasActiveSession to true", () => {
      manager.startSession("test-workspace");
      expect(manager.hasActiveSession).toBe(true);
    });

    it("should store server session ID if provided", () => {
      const session = manager.startSession("test-workspace", "server-123");
      expect(session.serverSessionId).toBe("server-123");
    });

    it("should replace existing session", () => {
      const first = manager.startSession("workspace-1");
      const second = manager.startSession("workspace-2");

      expect(manager.currentSession?.id).toBe(second.id);
      expect(manager.currentSession?.workspaceId).toBe("workspace-2");
    });
  });

  describe("endSession", () => {
    it("should return null if no active session", () => {
      const result = manager.endSession();
      expect(result).toBeNull();
    });

    it("should return the ended session", () => {
      const started = manager.startSession("test-workspace");
      const ended = manager.endSession();

      expect(ended?.id).toBe(started.id);
    });

    it("should clear the active session", () => {
      manager.startSession("test-workspace");
      manager.endSession();

      expect(manager.hasActiveSession).toBe(false);
      expect(manager.currentSession).toBeNull();
    });
  });

  describe("markCommitted", () => {
    it("should mark session as committed", () => {
      manager.startSession("test-workspace");
      manager.markCommitted();

      expect(manager.currentSession?.committed).toBe(true);
    });

    it("should do nothing if no active session", () => {
      // Should not throw
      manager.markCommitted();
    });
  });

  describe("working memory", () => {
    beforeEach(() => {
      manager.startSession("test-workspace");
    });

    describe("getAllWorkingMemory", () => {
      it("should return empty array for new session", () => {
        const entries = manager.getAllWorkingMemory();
        expect(entries).toEqual([]);
      });

      it("should return empty array if no session", () => {
        manager.endSession();
        const entries = manager.getAllWorkingMemory();
        expect(entries).toEqual([]);
      });
    });

    describe("clearWorkingMemory", () => {
      it("should not throw on empty working memory", () => {
        expect(() => manager.clearWorkingMemory()).not.toThrow();
        expect(manager.getAllWorkingMemory()).toHaveLength(0);
      });
    });
  });

  describe("getSessionSummary", () => {
    it("should return inactive status when no session", () => {
      const summary = manager.getSessionSummary();
      expect(summary.active).toBe(false);
    });

    it("should return session info when active", () => {
      manager.startSession("test-workspace", "server-123");

      const summary = manager.getSessionSummary();

      expect(summary.active).toBe(true);
      expect(summary.workspaceId).toBe("test-workspace");
      expect(summary.serverSessionId).toBe("server-123");
      expect(summary.workingMemoryCount).toBe(0);
      expect(summary.committed).toBe(false);
    });
  });

  describe("getTtlSeconds", () => {
    it("should return default TTL", () => {
      expect(manager.getTtlSeconds()).toBe(3600);
    });

    it("should return custom TTL", () => {
      const custom = new SessionManager({ ttlSeconds: 7200 });
      expect(custom.getTtlSeconds()).toBe(7200);
    });
  });
});
