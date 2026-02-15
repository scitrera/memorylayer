"use client";

import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useConnection } from "@/providers/connection-provider";

export default function SettingsPage() {
  const { connectionConfig, setConnectionConfig, isConnected, testConnection } =
    useConnection();

  const [baseUrl, setBaseUrl] = useState(connectionConfig.baseUrl);
  const [apiKey, setApiKey] = useState(connectionConfig.apiKey ?? "");
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setBaseUrl(connectionConfig.baseUrl);
    setApiKey(connectionConfig.apiKey ?? "");
  }, [connectionConfig]);

  async function handleTest() {
    setTesting(true);
    try {
      const ok = await testConnection();
      if (ok) {
        toast.success("Connection successful");
      } else {
        toast.error("Connection failed - server unreachable");
      }
    } catch {
      toast.error("Connection failed");
    } finally {
      setTesting(false);
    }
  }

  async function handleSave() {
    setConnectionConfig({
      baseUrl: baseUrl.trim() || "/api/ml",
      apiKey: apiKey.trim() || undefined,
      workspaceId: connectionConfig.workspaceId,
    });
    toast.success("Settings saved");

    // Auto-test connection after save
    setTesting(true);
    try {
      const ok = await testConnection();
      if (ok) {
        toast.success("Connection verified");
      } else {
        toast.error("Connection failed - server unreachable");
      }
    } catch {
      toast.error("Connection test failed");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure your MemoryLayer server connection
        </p>
      </div>

      <div className="space-y-4 rounded-lg border bg-card p-6">
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              isConnected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-sm font-medium">
            {isConnected ? "Connected" : "Disconnected"}
          </span>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="baseUrl"
            className="text-sm font-medium text-foreground"
          >
            Server URL
          </label>
          <input
            id="baseUrl"
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://localhost:61001"
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>

        <div className="space-y-2">
          <label
            htmlFor="apiKey"
            className="text-sm font-medium text-foreground"
          >
            API Key
          </label>
          <input
            id="apiKey"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Optional"
            className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>

        <div className="flex gap-3 pt-2">
          <button
            onClick={handleTest}
            disabled={testing}
            className="inline-flex h-9 items-center justify-center rounded-md border border-input bg-background px-4 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
          >
            {testing ? "Testing..." : "Test Connection"}
          </button>
          <button
            onClick={handleSave}
            className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
