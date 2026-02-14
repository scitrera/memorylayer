import { type Node, type Edge } from "@xyflow/react";

const RING_RADIUS = 250;

export function applyRadialLayout(
  nodes: Node[],
  edges: Edge[],
  centerId?: string,
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) return { nodes, edges };

  const center = centerId ?? nodes[0].id;

  // Build adjacency list
  const adj = new Map<string, Set<string>>();
  for (const node of nodes) {
    adj.set(node.id, new Set());
  }
  for (const edge of edges) {
    const src = edge.source as string;
    const tgt = edge.target as string;
    adj.get(src)?.add(tgt);
    adj.get(tgt)?.add(src);
  }

  // BFS to compute depth
  const depth = new Map<string, number>();
  depth.set(center, 0);
  const queue = [center];
  let head = 0;

  while (head < queue.length) {
    const current = queue[head++];
    const d = depth.get(current)!;
    for (const neighbor of adj.get(current) ?? []) {
      if (!depth.has(neighbor)) {
        depth.set(neighbor, d + 1);
        queue.push(neighbor);
      }
    }
  }

  // Assign depth to disconnected nodes
  for (const node of nodes) {
    if (!depth.has(node.id)) {
      depth.set(node.id, 3);
    }
  }

  // Group nodes by depth
  const rings = new Map<number, string[]>();
  for (const [id, d] of depth) {
    if (id === center) continue;
    const list = rings.get(d) ?? [];
    list.push(id);
    rings.set(d, list);
  }

  // Position center
  const positions = new Map<string, { x: number; y: number }>();
  positions.set(center, { x: 0, y: 0 });

  // Position each ring
  for (const [d, ids] of rings) {
    const radius = d * RING_RADIUS;
    const angleStep = (2 * Math.PI) / ids.length;
    ids.forEach((id, i) => {
      positions.set(id, {
        x: Math.cos(angleStep * i - Math.PI / 2) * radius,
        y: Math.sin(angleStep * i - Math.PI / 2) * radius,
      });
    });
  }

  const layoutNodes = nodes.map((node) => {
    const pos = positions.get(node.id) ?? { x: 0, y: 0 };
    return { ...node, position: pos };
  });

  return { nodes: layoutNodes, edges };
}
