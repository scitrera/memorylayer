# Deterministic Memory Evaluation — 2026-09-02

## Decision

Enable bounded typed-relation recall by default. Keep the enrichment policy at
`generative` and keep the omitted-request recall budget at `0`.

The relation gate passed by a useful margin: structured relations improved
relation-query Recall@5 by 19.8 absolute points, exceeded the 10-point gate,
left the generic control slice unchanged, and added 2.9 ms to warm p95 recall
latency when traversal fired. Enabling the arm with no relation edges changed no
rankings and added no p95 latency. The feature remains intent-gated, bounded,
caller-disableable, and a capability-checked no-op on storage providers without
typed-relation support.

The available evidence does not justify changing the other compatibility
defaults. The strongest prior LoCoMo QA configuration depends on generative fact
decomposition, and this run did not evaluate an equivalent deterministic
replacement during ingestion. A global non-zero recall budget would also change
the response shape for long memories; LoCoMo's short turns did not exercise that
case.

## Configuration and provenance

- Code revision: `ac5f456bade6ae7bfc2c5cf04e7e6e0cf214540e` plus the dirty
  deterministic-memory implementation under evaluation.
- Embeddings: `Qwen/Qwen3-VL-Embedding-2B`, 1,960 dimensions, through the
  production embedding endpoint.
- Retrieval: MemoryLayer's in-process SQLite path, vector ranking, no reranker,
  Recall@5 with a retrieval limit of 20.
- Query embeddings were warmed before arm timing. The warmup cost is retained in
  each JSON artifact.
- Every evaluated arm recorded zero generative calls.

Artifacts:

- `benchmarks/.cache/deterministic-memory-locomo-qwen1960.json`
- `benchmarks/.cache/deterministic-memory-relations-qwen1960.json`
- Runner: `benchmarks/run_deterministic_memory_eval.py`

## LoCoMo retrieval compatibility

The corpus contains 5,882 turns and the query set contains 200 questions,
balanced across multi-hop, open-domain, single-hop, and temporal categories.
Both arms used the exact same corpus, embeddings, and query order.

| Arm | Recall@5 | MRR | p50 | p95 | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compatibility (`generative`, relations off, no request budget) | 54.87% | 0.450 | 168.9 ms | 253.5 ms | 0 |
| Deterministic + relations + 2,048-token budget | 54.87% | 0.450 | 167.9 ms | 265.3 ms | 0 |

Recall and MRR were identical in every query-type slice. Pairwise inspection of
all 200 per-query rows found no ranking changes. Mean latency was 1.1 ms lower in
the treatment arm; its 11.7 ms p95 increase was not accompanied by a p50 or mean
increase. The budget produced zero truncations, omissions, or violations because
the returned LoCoMo turns fit within 2,048 tokens.

This is a retrieval compatibility evaluation, not the retrieve-answer-judge
LoCoMo score. The most recent complete same-environment MemoryLayer QA artifact
remains 78.0% at top 20 and 80.5% at top 30 over 400 questions using
`gemini-flash-latest` as answerer and judge.

## gbrain world-v1 typed relations

The imported gbrain suite contains 240 pages and 145 relation questions: 40
employment, 39 investment, 16 advisory, and 50 meeting-attendance questions.
Forty generic page-retrieval questions form a separate negative-control slice.

| Arm | Recall@5 | MRR | p95 | Relation paths |
| --- | ---: | ---: | ---: | ---: |
| Vector baseline | 48.10% | 0.359 | 35.9 ms | 0 |
| Relation arm, no edges | 48.10% | 0.359 | 35.8 ms | 0 |
| Content-only patterns | 48.10% | 0.359 | 39.9 ms | 0 |
| Trusted structured relations | 67.93% | 0.674 | 38.8 ms | 382 |
| Trusted structured relations, evidence qrels | 98.62% | 0.635 | 40.1 ms | 382 |

The structured arm wrote 261 resolved edges with no rejection or unresolved
endpoint. It consumed the suite's sealed `_facts` as trusted explicit relation
input, so it measures the storage/routing/retrieval ceiling rather than
autonomous extraction quality. Content-only extraction wrote no edges from this
corpus: its Markdown links and phrasing do not match the deliberately narrow
high-precision patterns. Structured ingest adapters are therefore the practical
route to realizing the measured gain.

The ordinary gbrain target qrel for attendance points to attendee profile pages,
while MemoryLayer intentionally returns the source memory that proves the edge.
Attendance target-page Recall@5 consequently falls from 15% to 7%, but its
evidence-page recall is 100%. Advisory, investment, and employment target recall
reach 100% with structured relations.

Generic controls remain 97.5% Recall@5 in both arms with 0/40 ranking changes.
The explicit generic, missing-seed, and wrong-relation-type safety cases all
return zero paths; only the missing seed is marked unresolved.

## Mem0 comparison status

No new full same-environment Mem0 number is reported. The preserved native Mem0
Qdrant index contains only 57 extracted memories across LoCoMo conversations
0–2 and does not retain source-turn qrel identifiers. It cannot support a valid
retrieval Recall@k calculation, and treating it as a completed QA corpus would be
misleading. This matches the earlier measured blocker: Mem0's serial
extract/search/reconcile write loop decelerated to roughly 4–5 turns per minute,
making a 5,882-turn ingest a multi-hour run.

For context only, the prior complete same-environment MemoryLayer QA result is
78.0% at top 20 / 80.5% at top 30. Mem0's published 92.5% uses a different hosted
pipeline and judge and is not an apples-to-apples comparator.

## Follow-up gates

1. Add structured relation mappings to ingestion adapters and measure extraction
   coverage separately from retrieval quality.
2. Re-run end-to-end LoCoMo QA after relations exist in the actual ingested
   corpus; raw-turn retrieval alone cannot measure their answer utility.
3. Do not switch the policy default to `adaptive` until fact-decomposition QA is
   at most one point below the generative baseline and cost dashboards confirm
   zero unlabeled calls.
4. Do not set a global recall-token default until long-memory and context-pack
   suites demonstrate both hard-budget compliance and preserved answer quality.
5. For a real same-environment Mem0 number, checkpoint one completed native
   ingest or explicitly label an `infer=False` batch run as a raw-store baseline.
