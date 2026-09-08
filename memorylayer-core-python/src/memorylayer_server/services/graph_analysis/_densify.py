"""In-process, config-gated graph densification for KB community detection.

The base association graph (explicit PART_OF / contradiction / auto-assoc edges)
is sparse for document-derived memories, so Louvain returns near-singleton
communities. This module augments that graph with weighted edges from up to
three additional signals so related memories actually cluster:

  * cosine  — single-vector cosine kNN across ALL memories (universal baseline;
              pulls in non-page memories like chat/facts).
  * maxsim  — ColPali MaxSim kNN across memories backed by a page multivector
              (the reliable signal for document content).
  * entity  — co-occurrence: memories sharing a registry entity (representation-
              agnostic cross-type glue between page and non-page memories).

Each signal's raw edge weights are min-max normalized to ~[0,1] then scaled by a
per-signal weight, so signals on different scales (MaxSim vs cosine) can be
combined without one swamping the others. Merged weights land on the NetworkX
edge ``weight`` attribute, which Louvain honours.

This is IN-PROCESS by design for a first experiment (loads vectors into memory
and does O(N^2) MaxSim over pages). It is NOT built for very large workspaces —
scale-out (approximate kNN / DB-side MaxSim) is a later step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..embedding._maxsim import MultiVectorEmbedding, maxsim_score

# Edge = (node_a, node_b, weight)
Edge = tuple[str, str, float]


@dataclass
class DensifyConfig:
    """Resolved densification knobs (from Variables / env)."""

    enabled: bool = False
    cosine_enabled: bool = True
    cosine_threshold: float = 0.82
    cosine_k: int = 8
    cosine_weight: float = 1.0
    maxsim_enabled: bool = True
    maxsim_threshold: float = 0.0
    maxsim_k: int = 8
    maxsim_weight: float = 1.5
    entity_enabled: bool = True
    entity_weight: float = 1.0
    entity_max_group: int = 50

    @classmethod
    def from_variables(cls, v) -> "DensifyConfig":
        from ...config import (
            DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_K,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_K,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD,
            DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT,
            DEFAULT_MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED,
            MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED,
            MEMORYLAYER_KB_DENSIFY_COSINE_K,
            MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD,
            MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT,
            MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED,
            MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP,
            MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT,
            MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED,
            MEMORYLAYER_KB_DENSIFY_MAXSIM_K,
            MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD,
            MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT,
            MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED,
        )

        if v is None:
            return cls()
        return cls(
            enabled=bool(v.get(MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED, DEFAULT_MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED)),
            cosine_enabled=bool(v.get(MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED, DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED)),
            cosine_threshold=float(v.get(MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD, DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD)),
            cosine_k=int(v.get(MEMORYLAYER_KB_DENSIFY_COSINE_K, DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_K)),
            cosine_weight=float(v.get(MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT, DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT)),
            maxsim_enabled=bool(v.get(MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED, DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED)),
            maxsim_threshold=float(v.get(MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD, DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD)),
            maxsim_k=int(v.get(MEMORYLAYER_KB_DENSIFY_MAXSIM_K, DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_K)),
            maxsim_weight=float(v.get(MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT, DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT)),
            entity_enabled=bool(v.get(MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED, DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED)),
            entity_weight=float(v.get(MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT, DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT)),
            entity_max_group=int(v.get(MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP, DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP)),
        )


# ------------------------------------------------------------------ #
# Pure edge builders (no I/O — unit-testable)
# ------------------------------------------------------------------ #

def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _dedupe(edges: Iterable[Edge], reduce: str = "max") -> list[Edge]:
    """Collapse duplicate undirected edges. ``reduce`` = 'max' or 'sum'."""
    acc: dict[tuple[str, str], float] = {}
    for a, b, w in edges:
        if a == b:
            continue
        key = _canonical(a, b)
        if key not in acc:
            acc[key] = w
        elif reduce == "sum":
            acc[key] += w
        else:
            acc[key] = max(acc[key], w)
    return [(a, b, w) for (a, b), w in acc.items()]


def _normalize_weight(edges: list[Edge], weight: float) -> list[Edge]:
    """Min-max normalize edge weights to [0,1], then scale by ``weight``."""
    if not edges:
        return []
    ws = [w for _, _, w in edges]
    lo, hi = min(ws), max(ws)
    rng = hi - lo
    out: list[Edge] = []
    for a, b, w in edges:
        nw = (w - lo) / rng if rng > 1e-9 else 1.0
        out.append((a, b, nw * weight))
    return out


def cosine_knn_edges(vectors: dict[str, list[float]], k: int, threshold: float) -> list[Edge]:
    """Undirected cosine-kNN edges over single-vector embeddings."""
    ids = [i for i, vec in vectors.items() if vec]
    if len(ids) < 2:
        return []
    m = np.asarray([vectors[i] for i in ids], dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mn = m / norms
    sims = mn @ mn.T
    np.fill_diagonal(sims, -1.0)
    kk = min(k, len(ids) - 1)
    edges: list[Edge] = []
    for i in range(len(ids)):
        row = sims[i]
        nbr = np.argpartition(row, -kk)[-kk:]
        for j in nbr:
            s = float(row[j])
            if s >= threshold:
                edges.append((ids[i], ids[int(j)], s))
    return _dedupe(edges, reduce="max")


def maxsim_knn_edges(
    multivectors: dict[str, list[list[float]]],
    k: int,
    threshold: float,
) -> list[Edge]:
    """MaxSim-kNN edges over MEMORY multivectors (ColPali late interaction).

    Document-derived memories carry a multivector directly (``Memory.multivector``,
    populated by the ingest pipeline), so we score memory-to-memory MaxSim without
    any page indirection. O(n^2) over the memories that have a multivector — fine
    at the experiment scale, not for very large workspaces.
    """
    ids = [i for i, mv in multivectors.items() if mv]
    if len(ids) < 2:
        return []
    mv = {i: MultiVectorEmbedding(vectors=multivectors[i]) for i in ids}
    edges: list[Edge] = []
    for i in range(len(ids)):
        scored = [
            (maxsim_score(mv[ids[i]], mv[ids[j]]), ids[j])
            for j in range(len(ids)) if j != i
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, nj in scored[:k]:
            if s >= threshold:
                edges.append((ids[i], nj, float(s)))
    return _dedupe(edges, reduce="max")


def entity_cooccurrence_edges(
    entity_members: dict[str, list[str]],
    node_set: set[str],
    max_group: int,
) -> list[Edge]:
    """Edges between memories sharing an entity. Edge weight accumulates over the
    number of shared entities (summed, then normalized by the caller). Entities
    with > ``max_group`` in-graph members are skipped (would over-connect).
    """
    edges: list[Edge] = []
    for members in entity_members.values():
        ms = [m for m in dict.fromkeys(members) if m in node_set]
        if len(ms) < 2 or len(ms) > max_group:
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                edges.append((ms[i], ms[j], 1.0))
    return _dedupe(edges, reduce="sum")


def _merge_into_graph(g, edges: list[Edge], signal: str) -> int:
    """Add/accumulate weighted edges into the graph. Returns edges added (new)."""
    added = 0
    for a, b, w in edges:
        if not g.has_node(a) or not g.has_node(b):
            continue
        if g.has_edge(a, b):
            data = g[a][b]
            data["weight"] = float(data.get("weight", 1.0)) + w
            signals = set(data.get("densify_signals", ()))
            signals.add(signal)
            data["densify_signals"] = tuple(sorted(signals))
        else:
            g.add_edge(a, b, weight=w, relationship_type=f"densify:{signal}",
                       strength=w, densify_signals=(signal,))
            added += 1
    return added


# ------------------------------------------------------------------ #
# Async orchestrator (I/O)
# ------------------------------------------------------------------ #

async def densify_graph(g, workspace_id: str, storage, cfg: DensifyConfig, logger) -> dict:
    """Augment ``g`` in place with weighted edges from the enabled signals.

    Returns a small summary dict for logging/telemetry.
    """
    node_ids = list(g.nodes())
    node_set = set(node_ids)
    base_edges = g.number_of_edges()

    # Ensure base (association) edges carry a numeric `weight` so Louvain weighs
    # them alongside the densified edges (they use `strength` today).
    for _a, _b, data in g.edges(data=True):
        if "weight" not in data:
            data["weight"] = float(data.get("strength", 1.0) or 1.0)

    summary = {"base_edges": base_edges, "cosine": 0, "maxsim": 0, "entity": 0}

    # 1. Cosine kNN (all memories with a single vector)
    if cfg.cosine_enabled:
        try:
            vectors = await storage.get_embeddings_batch(workspace_id, node_ids)
            edges = _normalize_weight(
                cosine_knn_edges(vectors or {}, cfg.cosine_k, cfg.cosine_threshold),
                cfg.cosine_weight,
            )
            summary["cosine"] = _merge_into_graph(g, edges, "cosine")
        except Exception as e:
            logger.warning("densify: cosine signal failed for %s: %s", workspace_id, e)

    # 2. MaxSim kNN (memories carrying a ColPali multivector)
    if cfg.maxsim_enabled:
        try:
            multivectors = await storage.get_multivectors_batch(workspace_id, node_ids)
            edges = _normalize_weight(
                maxsim_knn_edges(multivectors or {}, cfg.maxsim_k, cfg.maxsim_threshold),
                cfg.maxsim_weight,
            )
            summary["maxsim"] = _merge_into_graph(g, edges, "maxsim")
        except Exception as e:
            logger.warning("densify: maxsim signal failed for %s: %s", workspace_id, e)

    # 3. Entity co-occurrence. list_workspace_entity_members returns member rows
    # ({entity_id, memory_id, ...}); group them into entity -> [memory_id].
    if cfg.entity_enabled:
        try:
            member_rows = await storage.list_workspace_entity_members(workspace_id)
            grouped: dict[str, list[str]] = defaultdict(list)
            for row in member_rows or []:
                eid, mid = row.get("entity_id"), row.get("memory_id")
                if eid and mid:
                    grouped[eid].append(mid)
            edges = _normalize_weight(
                entity_cooccurrence_edges(grouped, node_set, cfg.entity_max_group),
                cfg.entity_weight,
            )
            summary["entity"] = _merge_into_graph(g, edges, "entity")
        except Exception as e:
            logger.warning("densify: entity signal failed for %s: %s", workspace_id, e)

    summary["total_edges"] = g.number_of_edges()
    g.graph["densify"] = summary  # retrievable by the densify_preview tuning tool
    logger.info(
        "densify workspace=%s: base=%d +cosine=%d +maxsim=%d +entity=%d -> total=%d edges",
        workspace_id, base_edges, summary["cosine"], summary["maxsim"],
        summary["entity"], summary["total_edges"],
    )
    return summary
