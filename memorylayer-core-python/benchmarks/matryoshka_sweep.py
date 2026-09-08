#!/usr/bin/env python3
"""Offline matryoshka dimension sweep over already-ingested vectors.

Answers "how far can the memory embedder be truncated (matryoshka) before recall
degrades, and does the entity-anchor arm compensate?" — WITHOUT re-ingesting at
each dimension. Because matryoshka embeddings are nested, the first N dims of a
stored D-dim vector are (after L2-renorm) the N-dim embedding. So this pulls the
stored vectors ONCE from a PostgreSQL store, truncates+renormalizes to each dim
in numpy, and computes pure-vector and entity-anchor (speaker-restricted) recall@k
by brute force. Seconds, no GPU, no re-embed.

It self-validates: at the full stored dim, pure-vector and +entity-anchor recall
should match the live `run_benchmark.py` pipeline within ~0.01 (methodology differs
only by overfetch/boosts).

Env (all have defaults; the defaults assume a local eval stack — point them at
your own Postgres store and embedding endpoint):
  MATRYOSHKA_PG_URL       Postgres store with ingested memories (metadata.eval_key,
                          metadata.speaker, embedding).
  MATRYOSHKA_QRELS        qrels.json (eval harness format). Default: LoCoMo cache.
  MATRYOSHKA_EMB_URL      OpenAI-compatible /v1 base for query embeddings.
  MATRYOSHKA_EMB_MODEL    embedding model id.
  MATRYOSHKA_DIMS         comma list of dims to sweep (must be <= stored dim).

Requires: numpy, asyncpg, openai. Run with the enterprise venv (has asyncpg).
"""

import os
import json
import asyncio
from collections import defaultdict

import numpy as np
import asyncpg
from openai import AsyncOpenAI

from memorylayer_server.services.memory.entities import extract_query_entities

PG_URL = os.environ.get("MATRYOSHKA_PG_URL", "postgresql://memorylayer:memorylayer@localhost:5432/memorylayer")
QRELS = os.environ.get("MATRYOSHKA_QRELS", "benchmarks/.cache/locomo/qrels.json")
EMB_URL = os.environ.get("MATRYOSHKA_EMB_URL", "http://localhost:8002/v1")
EMB_MODEL = os.environ.get("MATRYOSHKA_EMB_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
DIMS = [int(x) for x in os.environ.get("MATRYOSHKA_DIMS", "64,128,256,384,512,768,1024,1536,1920").split(",")]
K = 5
RRF_K = 60  # DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_RRF_K
ENTITY_POOL = 200  # _ENTITY_ANCHOR_POOL


def _l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def _rrf_fuse(vec_rank, ent_rank, k=RRF_K):
    score = {}
    for r, i in enumerate(vec_rank):
        score[i] = score.get(i, 0) + 1.0 / (k + r + 1)
    for r, i in enumerate(ent_rank):
        score[i] = score.get(i, 0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]


async def main():
    conn = await asyncpg.connect(PG_URL)
    rows = await conn.fetch(
        "SELECT metadata->>'eval_key' k, metadata->>'speaker' sp, embedding::text e "
        "FROM memories WHERE embedding IS NOT NULL AND metadata ? 'eval_key'"
    )
    await conn.close()
    keys = [r["k"] for r in rows]
    speakers = [r["sp"] for r in rows]
    M = np.array([np.fromstring(r["e"].strip("[]"), sep=",", dtype=np.float32) for r in rows])
    print(f"loaded {M.shape[0]} memories, stored dim={M.shape[1]}")

    qrels = json.load(open(QRELS))["queries"]
    queries = [q["query"] for q in qrels]
    relevants = [set(q["relevant"]) for q in qrels]
    qents = [set(extract_query_entities(q)) for q in queries]

    cli = AsyncOpenAI(api_key="x", base_url=EMB_URL)
    QE = []
    stored_dim = M.shape[1]
    for i in range(0, len(queries), 64):
        resp = await cli.embeddings.create(model=EMB_MODEL, input=queries[i : i + 64], dimensions=stored_dim)
        QE.extend([d.embedding for d in resp.data])
    Q = np.array(QE, dtype=np.float32)

    sp2idx = defaultdict(list)
    for i, s in enumerate(speakers):
        if s:
            sp2idx[s].append(i)

    print(f"\n{'dim':>6}{'vector':>10}{'+anchor':>10}{'Δanchor':>10}")
    results = []
    for d in DIMS:
        if d > stored_dim:
            continue
        Md = _l2(M[:, :d])
        Qd = _l2(Q[:, :d])
        sims = Qd @ Md.T
        vec_r, anc_r = [], []
        for qi in range(len(queries)):
            order = np.argsort(-sims[qi])
            vtop = [keys[i] for i in order[:K]]
            vec_r.append(_recall_at_k(vtop, relevants[qi], K))
            cand = []
            for ent in qents[qi]:
                cand.extend(sp2idx.get(ent, []))
            cand = list(dict.fromkeys(cand))
            if cand:
                cand_sorted = sorted(cand, key=lambda i: -sims[qi, i])[:ENTITY_POOL]
                fused = _rrf_fuse(list(order[:60]), cand_sorted, RRF_K)
                atop = [keys[i] for i in fused[:K]]
            else:
                atop = vtop
            anc_r.append(_recall_at_k(atop, relevants[qi], K))
        vr, ar = float(np.mean(vec_r)), float(np.mean(anc_r))
        results.append({"dim": d, "vector": round(vr, 3), "anchor": round(ar, 3)})
        print(f"{d:>6}{vr:>10.3f}{ar:>10.3f}{ar - vr:>+10.3f}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
