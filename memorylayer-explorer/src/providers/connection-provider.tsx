"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";
import { getClient, resetClient } from "@/lib/client";

interface ConnectionConfig {
  baseUrl: string;
  apiKey?: string;
  workspaceId?: string;
}

interface ConnectionContextValue {
  connectionConfig: ConnectionConfig;
  setConnectionConfig: (config: ConnectionConfig) => void;
  client: MemoryLayerClient;
  isConnected: boolean;
  testConnection: () => Promise<boolean>;
}

const STORAGE_KEY = "memorylayer-connection";

const DEFAULT_CONFIG: ConnectionConfig = {
  baseUrl: "/api/ml",
  workspaceId: "_default",
};

const ConnectionContext = createContext<ConnectionContextValue | null>(null);

function loadConfig(): ConnectionConfig {
  if (typeof window === "undefined") return DEFAULT_CONFIG;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<ConnectionConfig>;
      return {
        baseUrl: parsed.baseUrl || DEFAULT_CONFIG.baseUrl,
        apiKey: parsed.apiKey,
        workspaceId: parsed.workspaceId || DEFAULT_CONFIG.workspaceId,
      };
    }
  } catch {
    // ignore parse errors
  }
  return DEFAULT_CONFIG;
}

function saveConfig(config: ConnectionConfig): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // ignore storage errors
  }
}

export function ConnectionProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<ConnectionConfig>(DEFAULT_CONFIG);
  const [isConnected, setIsConnected] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    setConfig(loadConfig());
  }, []);

  const client = getClient(config);

  const testConnection = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${config.baseUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      const connected = response.ok;
      setIsConnected(connected);
      return connected;
    } catch {
      setIsConnected(false);
      return false;
    }
  }, [config.baseUrl]);

  // Periodic health check
  useEffect(() => {
    testConnection();
    intervalRef.current = setInterval(testConnection, 30_000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [testConnection]);

  const updateConfig = useCallback(
    (newConfig: ConnectionConfig) => {
      setConfig(newConfig);
      saveConfig(newConfig);
      resetClient();
    },
    []
  );

  return (
    <ConnectionContext.Provider
      value={{
        connectionConfig: config,
        setConnectionConfig: updateConfig,
        client,
        isConnected,
        testConnection,
      }}
    >
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection(): ConnectionContextValue {
  const context = useContext(ConnectionContext);
  if (!context) {
    throw new Error("useConnection must be used within a ConnectionProvider");
  }
  return context;
}
