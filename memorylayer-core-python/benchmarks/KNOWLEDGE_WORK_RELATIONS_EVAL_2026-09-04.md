# General knowledge-work relation evaluation — 2026-09-04

## Outcome

Metadata-first relation acquisition passed its initial deterministic contract and
improved evidence retrieval over an already strong semantic baseline. This is
evidence that the relational recall default can contribute beyond the
meeting/venture-capital distribution in gbrain `world-v1`.

The implementation remains opt-in at ingestion: only the reserved
`metadata.knowledge_work` profile creates these edges. Relational recall remains
enabled by default because generic queries did not enter the relation path.

## Setup

- Dataset: `benchmarks/datasets/knowledge_work_relations_v1.json`
- Dataset SHA-256: `a15a6729b9d1af89f0cf2351042ec019f2d538465da35aa06b7ff4b47c7a2c7d`
- Records: 20, including five generic controls
- Gold relation assertions: 48
- Relation queries: 30 across 11 query families
- Generic control queries: 5
- Domains: software delivery, operations, incident management, architecture,
  business operations, product research, policy/compliance, management,
  organizational work, and analytics
- Embeddings: production embed-server endpoint, Qwen3-VL-Embedding-2B,
  1,960 dimensions
- Ranking: vector retrieval, no reranker, limit 10, scored at R@5
- Generation policy: deterministic; no LLM relation extraction
- Comparison: the same ingested corpus and query order with `include_relations`
  disabled versus enabled

The benchmark passes no `RememberInput.relations`. All stored edges must be
acquired by the production remember pipeline from the source metadata profile.

## Results

### Acquisition and persistence

| Measure | Result |
| --- | ---: |
| Expected assertions | 48 |
| Acquired assertions | 48 |
| Precision | 1.000 |
| Recall | 1.000 |
| Persisted edges/evidence | 48 |
| Unresolved | 0 |
| Rejected | 0 |
| Duplicate | 0 |
| Canonical entities | 61 |

This is profile-conformance precision/recall against hand-authored structured
metadata. It does not measure noisy connector-field normalization or relation
extraction from prose.

### Relational questions

| Measure | Vector baseline | Metadata relations | Delta |
| --- | ---: | ---: | ---: |
| R@5 | 0.9667 | 1.0000 | +0.0333 |
| MRR | 0.9000 | 1.0000 | +0.1000 |
| nDCG@5 | 0.9131 | 1.0000 | +0.0869 |
| p95 latency | 11.196 ms | 14.056 ms | +2.860 ms |

All 30 relation queries produced a path, yielding 34 bounded paths in total.
Five queries improved in pairwise MRR, 25 tied, and none regressed. The weighted
relation RRF arm makes exact, evidence-backed paths outrank a vector-only result
instead of allowing the two rank-one signals to tie on a random memory-id
tie-break.

### Generic controls

| Measure | Vector baseline | Relations enabled | Delta |
| --- | ---: | ---: | ---: |
| R@5 | 1.0000 | 1.0000 | 0 |
| MRR | 1.0000 | 1.0000 | 0 |
| nDCG@5 | 1.0000 | 1.0000 | 0 |
| Rankings changed | 0 | 0 | 0 |
| Relation paths | 0 | 0 | 0 |

The control slice is small, so it is a regression sentinel rather than proof of
broad intent-classifier specificity.

### Existing gbrain regression gate

The production-Qwen `world-v1` relation suite was rerun after increasing the
exact-relation RRF weight. Its trusted-structured arm retained R@5 of 0.679
against the 0.481 baseline (+0.198) and improved MRR to 0.686 (+0.327). Generic
controls remained R@5 0.975 and MRR 0.946 with zero changed rankings. The
no-edge arm also retained identical retrieval quality. Thus the ranking fix did
not worsen the meeting/VC regression surface or the known attendance
evidence-versus-target-profile mismatch.

## Interpretation

The gain is smaller than the gbrain trusted-structured ceiling (+19.8 R@5
points) because production embeddings already retrieve most of this compact
corpus. The remaining improvement is nevertheless useful: metadata relations
close the final recall gap and move every relevant evidence memory to rank one,
with no generic ranking changes and about 2.9 ms p95 traversal overhead.

The benchmark validates the highest-confidence acquisition path—source-system
metadata—not autonomous prose extraction. The next external-validity gate should
use held-out exports from multiple connector types with independently labeled
edges, missing fields, stale assignments, aliases, and deliberately incorrect
metadata. An end-to-end answer/judge run should follow once those fixtures exist.

## Reproduce

```bash
python benchmarks/run_knowledge_work_relations_eval.py \
  --embedding-provider embed_server \
  --embed-server-url http://127.0.0.1:8002 \
  --dimensions 1960 \
  --json-out benchmarks/.cache/knowledge-work-relations-v1-qwen1960.json
```

The JSON artifact records per-query rankings, paths, latencies, configuration,
dataset hash, acquisition errors, and the source revision.
