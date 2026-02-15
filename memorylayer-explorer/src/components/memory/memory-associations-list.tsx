"use client";

import Link from "next/link";
import type { Association } from "@/types";
import { RELATIONSHIP_TO_CATEGORY, RELATIONSHIP_CATEGORY_COLORS } from "@/lib/constants";
import { formatStrength } from "@/lib/format";
import { TimeAgo } from "@/components/shared/time-ago";

interface MemoryAssociationsListProps {
  associations: Association[];
  currentMemoryId: string;
}

export function MemoryAssociationsList({
  associations,
  currentMemoryId,
}: MemoryAssociationsListProps) {
  if (associations.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">No associations</div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">Direction</th>
            <th className="px-3 py-2 font-medium">Relationship</th>
            <th className="px-3 py-2 font-medium">Linked Memory</th>
            <th className="px-3 py-2 font-medium">Strength</th>
            <th className="px-3 py-2 font-medium">Created</th>
          </tr>
        </thead>
        <tbody>
          {associations.map((assoc) => {
            const isSource = assoc.source_id === currentMemoryId;
            const linkedId = isSource ? assoc.target_id : assoc.source_id;
            const category = RELATIONSHIP_TO_CATEGORY[assoc.relationship] ?? "context";
            const categoryColors = RELATIONSHIP_CATEGORY_COLORS[category];

            return (
              <tr key={assoc.id} className="border-b hover:bg-muted/50">
                <td className="px-3 py-2">
                  <span className="text-xs text-muted-foreground">
                    {isSource ? "outgoing" : "incoming"}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`inline-flex rounded-md px-2 py-0.5 text-xs font-medium ${
                      categoryColors?.bg ?? "bg-slate-50"
                    } ${categoryColors?.text ?? "text-slate-700"}`}
                  >
                    {assoc.relationship.replace(/_/g, " ")}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <Link
                    href={`/memories/${linkedId}`}
                    className="font-mono text-xs text-primary hover:underline"
                  >
                    {linkedId.slice(0, 12)}...
                  </Link>
                </td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 rounded-full bg-slate-200">
                      <div
                        className="h-1.5 rounded-full bg-primary"
                        style={{ width: `${assoc.strength * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {formatStrength(assoc.strength)}
                    </span>
                  </div>
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  <TimeAgo date={assoc.created_at} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
