# General knowledge-work relations

MemoryLayer's knowledge-work profile is a small interoperability layer for
metadata that professional tools already know with high confidence. It is not a
replacement for the base ontology and it does not require RDF. Connectors place
the profile under `memory.metadata.knowledge_work`; the normal remember pipeline
creates typed canonical entities and evidence-backed relations without an LLM.

## Ontology review

The profile deliberately reuses the stable common denominator of established
vocabularies rather than copying any one domain model:

- [W3C PROV-O](https://www.w3.org/TR/prov-o/) separates entities, activities,
  and agents and supplies attribution, generation, use, derivation, and
  responsibility. It is the basis for artifact provenance, evidence linkage,
  and keeping an assertion's source.
- [Dublin Core Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/)
  supplies widely used resource metadata: creator, contributor, `isPartOf`,
  references, source, requires, and replaces. It is the basis for authorship,
  contribution, containment, references, dependencies, and supersession.
- [Schema.org Action](https://schema.org/Action) models an agent acting on an
  object with participants, instruments, and results. [Schema.org Role](https://schema.org/Role)
  shows why a relationship may need role and time qualifiers. These inform the
  active-voice actor edges and the qualified source profile.
- [W3C Organization Ontology](https://www.w3.org/TR/vocab-org/) models
  organizational membership as either a direct relation or a qualified
  membership with role and duration. It is the basis for `member_of`; role and
  validity remain profile qualifiers rather than being flattened into new verbs.
- [ActivityStreams 2.0 Core](https://www.w3.org/TR/activitystreams-core/) uses a
  general actor/object/target/result/instrument activity shape. It supports a
  connector-neutral event representation, but its large verb vocabulary is not
  imported into MemoryLayer's retrieval ontology.
- [P-Plan](https://www.opmw.org/model/p-plan/) extends PROV-O with plans and
  steps. It motivates distinct `project` and `work_item` entities and explicit
  workflow dependencies.
- [OSLC Change Management](https://docs.oasis-open.org/oslc-domains/cm/v3.0/cs01/part2-change-mgt-vocab/cm-v3.0-cs01-part2-change-mgt-vocab.html)
  covers tasks, review tasks, change requests, authorization, parentage, and
  affected resources. It informs responsibility, review/approval, containment,
  and impact relations.
- [W3C Time Ontology](https://www.w3.org/TR/owl-time/) provides the richer model
  to use when connector validity intervals eventually need first-class temporal
  querying. Today those values remain on the source assertion metadata.

No reviewed vocabulary directly captures the complete everyday retrieval
contract: “who owns this work?”, “what decision affected this system?”, and
“which evidence supports that decision?”. MemoryLayer therefore adopts a small
application profile whose terms have explicit mappings to those standards.

## Core profile

The typed entities added for metadata-first acquisition are:

| Entity | Intended use |
| --- | --- |
| `artifact` | Documents, datasets, models, designs, messages, and other outputs |
| `work_item` | Tasks, issues, requirements, deliverables, and corrective actions |
| `decision` | Named choices, approvals, and policy decisions |
| `topic` | Subjects used to organize knowledge resources |

They complement the existing `person`, `org`, `project`, `event`, `place`, and
`concept` types. They are metadata-addressable but are not added to the default
NER label set, so enabling this profile does not broaden probabilistic text
extraction.

The profile normalizes passive aliases onto active canonical edges:

| Profile field | Canonical edge |
| --- | --- |
| `owner`, `owners`, `owned_by` | agent `owns` subject |
| `assignee`, `assignees`, `assigned_to`, `responsible` | agent `responsible_for` subject |
| `author`, `authors`, `creator`, `creators`, `created_by` | agent `authored` subject |
| `contributor`, `contributors` | agent `contributed_to` subject |
| `reviewer`, `reviewers`, `reviewed_by` | agent `reviewed` subject |
| `approver`, `approvers`, `approved_by` | agent `approved` subject |
| `decision_maker`, `decision_makers`, `decided_by` | agent `decided` subject |
| `project`, `organization`, `part_of` | subject `part_of` container |
| `member_of` | subject `member_of` organization |
| `depends_on`, `blocks`, `blocked_by` | normalized workflow edge |
| `references`, `supersedes`, `based_on` | normalized provenance edge |
| `affects`, `supports`, `supported_by`, `contradicts` | normalized impact/evidence edge |
| `about`, `topics` | subject `about` topic |

`relations` is an escape hatch for a registered ontology term that lacks a
shorthand. Either endpoint may be the literal `$subject`.

## Wire example

```json
{
  "content": "Delivery record PM-204 is ready for the next review.",
  "metadata": {
    "source": "project_connector",
    "knowledge_work": {
      "subject": {
        "name": "Phoenix Migration",
        "type": "project",
        "aliases": ["PM-204"],
        "external_ids": {"project_system": "204"}
      },
      "owner": {"name": "Alice Chen", "type": "person"},
      "organization": {"name": "Platform Group", "type": "org"},
      "depends_on": {"name": "Security Sign-off", "type": "work_item"},
      "about": "Customer Identity",
      "valid_from": "2026-08-01",
      "role": "accountable"
    }
  }
}
```

The profile is preserved verbatim on the memory, so `valid_from`, `role`, source
record revisions, connector-specific status, and similar qualifiers remain
available through relation evidence. The compact graph contains only the
retrieval-stable edge. Entity aliases and external IDs are also carried into
entity-registry provenance.

Malformed entries are isolated: valid assertions from the same profile are
still written, while rejected and unresolved counts appear in
`memory.relation_write_result`. Replaying the same memory and assertion remains
idempotent through the existing relation-evidence uniqueness contract.

## Connector normalization

`services.ingest.normalize_connector_metadata` is the common deterministic
boundary for connector-shaped records. It recognizes an explicit
`connector_type` or `source_kind`, including records nested below
`connector_record`, and maps an allowlist of source fields into the core
profile. It handles common snake/camel-case identity objects (`displayName`,
`accountId`, `emailAddress`, `login`), arrays, aliases, source record IDs, and
sentinel values such as `unassigned`. It does not inspect free text.

The boundary is used by:

- single and batch `POST /v1/memories` requests that declare a connector;
- `email_to_remember_input`, which adds message authorship and registry-backed
  `sent_to` assertions while retaining its existing explicit relation output;
- document processing, which carries the normalized profile and normalization
  audit record onto page-derived memories without copying the entire connector
  payload or credentials; and
- `connector_to_remember_input`, the pure adapter for additional connectors.

Caller-authored profile values are authoritative. The normalizer may fill a
missing field from source metadata, but never overwrites an existing profile
field. A malformed explicit profile is preserved and reported rather than
silently replaced. Subject-only records do not activate relation acquisition,
and arbitrary metadata without a connector declaration is left untouched.

`metadata.knowledge_work_normalization` records the normalizer version,
connector type, inferred record kind, mapped fields, and isolated warnings.
These diagnostics are bounded and contain no free-text inference output.

On conditional delete, relation evidence is deactivated by the storage layer.
Restoring a versioned metadata-first memory deterministically replays its stored
profile, reactivating the same idempotent evidence rows on SQLite and PostgreSQL.

## Benchmark contract

`benchmarks/datasets/knowledge_work_relations_v1.json` is checked in and hand
authored. It covers software delivery, operations, incident management,
architecture, policy/compliance, research, management, organizational work, and
analytics. It includes generic non-relational controls.

Run it with local lexical embeddings:

```bash
python benchmarks/run_knowledge_work_relations_eval.py \
  --json-out benchmarks/.cache/knowledge-work-relations-v1-hash.json
```

Or with the production embedding service:

```bash
python benchmarks/run_knowledge_work_relations_eval.py \
  --embedding-provider embed_server \
  --embed-server-url http://127.0.0.1:8002 \
  --dimensions 1960 \
  --json-out benchmarks/.cache/knowledge-work-relations-v1-qwen1960.json
```

The acquisition score is independent of retrieval: predicted triples are
compared directly with hand-authored triples. Retrieval then compares the same
stored corpus with relation recall off and on. This is an evidence-retrieval
benchmark, not an answer-generation benchmark, and its curated fixture should be
supplemented with connector-derived held-out data before using it as a broad
production-quality claim.

`benchmarks/datasets/knowledge_work_connectors_v1.json` is the harder boundary
suite. Its records resemble GitHub, Jira, Linear, Google Drive, Confluence,
ServiceNow, Notion, Slack, Teams, email, and model-registry payloads. It includes
nested identities, vendor keys, aliases, duplicate fields, missing display
names, empty/sentinel values, explicit-profile precedence, and non-connector
controls. Run it through the same evaluator with:

```bash
python benchmarks/run_knowledge_work_relations_eval.py \
  --dataset benchmarks/datasets/knowledge_work_connectors_v1.json \
  --json-out benchmarks/.cache/knowledge-work-connectors-v1-hash.json
```
