import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import { type Node, type Edge } from "@xyflow/react";

interface SimNode extends SimulationNodeDatum {
  id: string;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  strength: number;
}

export function applyForceLayout(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) return { nodes, edges };

  const simNodes: SimNode[] = nodes.map((n) => ({
    id: n.id,
    x: n.position.x || Math.random() * 500,
    y: n.position.y || Math.random() * 500,
  }));

  const nodeIndex = new Map(simNodes.map((n, i) => [n.id, i]));

  const simLinks: SimLink[] = edges
    .filter(
      (e) => nodeIndex.has(e.source as string) && nodeIndex.has(e.target as string),
    )
    .map((e) => ({
      source: nodeIndex.get(e.source as string)!,
      target: nodeIndex.get(e.target as string)!,
      strength: (e.data?.strength as number) ?? 0.5,
    }));

  const simulation = forceSimulation<SimNode>(simNodes)
    .force(
      "link",
      forceLink<SimNode, SimLink>(simLinks).distance(
        (d) => 100 + (1 - d.strength) * 200,
      ),
    )
    .force("charge", forceManyBody().strength(-300))
    .force("center", forceCenter(0, 0))
    .force("collide", forceCollide(60))
    .stop();

  for (let i = 0; i < 300; i++) {
    simulation.tick();
  }

  const layoutNodes = nodes.map((node, i) => ({
    ...node,
    position: {
      x: simNodes[i].x ?? 0,
      y: simNodes[i].y ?? 0,
    },
  }));

  return { nodes: layoutNodes, edges };
}
