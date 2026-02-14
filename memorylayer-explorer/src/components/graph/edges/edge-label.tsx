"use client";

interface EdgeLabelProps {
  relationship: string;
  strength: number;
}

export function EdgeLabel({ relationship, strength }: EdgeLabelProps) {
  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-md px-2 py-0.5 text-xs font-mono text-slate-600 shadow-sm border border-slate-100 pointer-events-auto whitespace-nowrap">
      {relationship.replace(/_/g, " ")} ({Math.round(strength * 100)}%)
    </div>
  );
}
