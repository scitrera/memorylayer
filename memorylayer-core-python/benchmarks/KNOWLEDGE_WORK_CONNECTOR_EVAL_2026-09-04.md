# Messy knowledge-work connector evaluation — 2026-09-04

## Outcome

The connector boundary normalized and persisted all 57 hand-authored relations
without a false positive, miss, rejection, or unresolved endpoint. Relational
recall improved the production-embedding retrieval arm and did not change any
generic-control ranking.

The first hash-embedding run also found a genuine query-routing defect:
`What superseded Data Retention Policy 2024?` recognized the relation but did
not preserve the full seed phrase. The checked-in result is from the rerun after
fixing and regression-testing the `superseded`/`replaced` forms.

## Fixture

- Dataset: `benchmarks/datasets/knowledge_work_connectors_v1.json`
- SHA-256: `5c29e91a339fa426658deb3957519c280b9d9569203b5c3af4851dca6af61972`
- 19 records: 14 connector-shaped relation records and 5 generic controls
- 57 gold relations
- 29 questions: 24 relation questions and 5 controls
- Domains include delivery, operations, incidents, architecture, governance,
  organizational work, collaboration, analytics, quality, and vendor work
- Noise includes nested records, camelCase identities, source IDs, aliases,
  duplicate names, malformed identities, and empty/sentinel values

All records pass through `normalize_connector_metadata` before the existing
metadata relation parser and normal remember pipeline. No gold relation is
passed through `RememberInput.relations`.

## Acquisition and persistence

| Metric | Result |
| --- | ---: |
| Expected / predicted / true-positive relations | 57 / 57 / 57 |
| Precision / recall | 1.000 / 1.000 |
| False positives / false negatives | 0 / 0 |
| Persisted resolved relations | 57 |
| Unresolved / rejected / duplicate | 0 / 0 / 0 |
| Canonical entities | 70 |

Three deliberately malformed values were isolated as warnings: one identity
without a display name and two empty/sentinel dependencies. Valid fields in the
same source record were retained.

## Retrieval with production Qwen embeddings

Endpoint configuration: `embed_server`, 1960 dimensions.

| Relation questions (24) | Vector baseline | Metadata relations | Delta |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.9583 | 1.0000 | +0.0417 |
| MRR | 0.9375 | 1.0000 | +0.0625 |
| nDCG@5 | 0.9430 | 1.0000 | +0.0570 |
| p95 latency | 13.594 ms | 15.507 ms | +1.913 ms |

All 24 relation questions fired the graph arm and returned 26 evidence paths.
Two rankings improved, 22 tied, and none regressed. The five generic controls
remained at 1.0 Recall@5/MRR/nDCG with zero ranking changes. Control latency is
not interpreted because five sequential samples are too noisy for a latency
claim.

## Offline hash result

The dependency-free hash run also reached 1.0 Recall@5/MRR/nDCG on relation
questions, improving five rankings with 19 ties and no regressions. Relation
p95 increased by 0.556 ms. Generic-control rankings were unchanged.

## Interpretation and rollout

This result supports automatic normalization only when a request explicitly
declares a connector envelope, with relational recall enabled. It does not
support mining arbitrary metadata or unstructured text heuristically. The
fixture is intentionally difficult at the schema boundary but remains curated;
the next evidence tier is a replay of sampled, redacted production connector
payloads and repeated latency trials under representative concurrency.

The generation-policy default remains independent: this benchmark evaluates
acquisition, persistence, query routing, and evidence retrieval, not generated
answer quality.

## Reproduce

```bash
python benchmarks/run_knowledge_work_relations_eval.py \
  --dataset benchmarks/datasets/knowledge_work_connectors_v1.json \
  --embedding-provider embed_server \
  --embed-server-url http://127.0.0.1:8002 \
  --dimensions 1960 \
  --json-out benchmarks/.cache/knowledge-work-connectors-v1-qwen1960.json
```
