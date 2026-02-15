"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useConnection } from "@/providers/connection-provider";
import type { Association } from "@/types";

export function useAssociations(
  memoryId: string | undefined,
  direction: "outgoing" | "incoming" | "both" = "both",
) {
  const { client, isConnected } = useConnection();

  return useQuery<Association[]>({
    queryKey: ["memories", memoryId, "associations", direction],
    queryFn: () => client.getAssociations(memoryId!, direction),
    enabled: isConnected && !!memoryId,
  });
}

export function useCreateAssociation() {
  const { client } = useConnection();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      sourceId,
      targetId,
      relationship,
      strength = 0.5,
    }: {
      sourceId: string;
      targetId: string;
      relationship: string;
      strength?: number;
    }) => client.associate(sourceId, targetId, relationship, strength),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["memories", variables.sourceId, "associations"],
      });
      queryClient.invalidateQueries({
        queryKey: ["memories", variables.targetId, "associations"],
      });
      queryClient.invalidateQueries({ queryKey: ["graph"] });
    },
  });
}
