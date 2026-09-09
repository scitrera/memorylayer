# Changelog

All notable changes to the MemoryLayer OSS packages are documented here.

Packages in this repo share one version line (see `versions.yaml`), with two
deliberate exceptions: `memorylayer-embed-server` (container image only) and
`memorylayer-explorer` (a Next.js app, unpublished) are versioned independently.

## [0.2.0] — 2026-09-08

The largest release so far: a new Go SDK, two new agent-host plugins, an entity
and perspective layer under recall, and a retrieval-quality gate in CI.

### Added

**New packages**
- `memorylayer-sdk-go` — Go client SDK mirroring the Python/TypeScript SDKs
  (memories, recall, documents, skills, MCP registries, threads, sessions,
  workspaces, entities, knowledgebase, datasets, tokens). Includes a nested
  `aether/` module providing an Aether-relay transport, and bound-namespace
  on-behalf-of proxies for Skills / McpServers / Kb.
- `memorylayer-openclaw-plugin` — backs OpenClaw's first-class memory capability.
  Message persistence, search management, and opt-in skills sync
  (`MEMORYLAYER_SKILLS_SYNC=1`) that materializes MemoryLayer skills into
  OpenClaw's watched directory. Tracked and CI-tested; not yet published to npm.
- `memorylayer-codex-plugin` — MemoryLayer for Codex via MCP, with recall and
  durable-capture guidance skills.
- `memorylayer-server-rpg` — Repository Planning Graph services, included in the
  server container and released as a separate Python plugin on PyPI. Installing
  it brings in the matching server version and automatically enables `/v1/rpg`.

**Deterministic memory and context**
- Generation policies, confidence reporting, bounded typed-relation recall, and
  structured knowledge-work connector normalization.
- Durable raw session checkpoints, budgeted context packs, and signed context
  delta cursors, with SDK and MCP support.
- Pre-compaction transcript capture for the Claude Code and OpenCode plugins.

**Entity registry**
- Canonical entity nodes with aliases and accretion, alias-aware resolution, and
  a seeded catalog of canonical entities.
- Batch dedup via union-find clustering with representative merge; self-merge and
  double-merge guards.
- PROV-O-flavored provenance/lineage reads and a unified `build_provenance()`.
- Entity relatedness through member co-occurrence, plus a bulk `cooccurrence_map`
  for whole-graph relatedness.
- External-KB entity-linking seam with a Wikidata enrichment provider.
- Entity types became a first-class ontology dimension; backfill from existing
  memories, registry wipe, and force-re-extract for a clean rebuild.

**Perspective and representation**
- `get_representation(observer, subject)` assembles what one party understands
  about another, anchored on `observer_id` / `subject_id`.
- User-scope self-representation: preferences that follow a user across
  workspaces, with a preference-vs-episodic auto-classifier for routing.
- New `/v1/representation` endpoints.

**Retrieval**
- Query-intent classification (rule-based, no LLM) driving soft channel selection.
- MMR result diversification, structured cue anchors with a cue-entity link, a
  fact retrieval channel, and a bounded graph-traversal channel — each flag-gated.
- Entity-anchored recall enabled by default.
- Temporal support: `event_time` on memories, time filtering/ordering,
  `get_timeline`, and temporal neighbors.
- Alias and backlink boosting with full-text index reconciliation.
- `RecallMode.AGENTIC` added for enterprise agentic recall.

**Evaluation**
- `memorylayer-eval` CLI with a retrieval harness, metrics (recall@k, precision@k,
  MRR, nDCG), and a `gate` subcommand that fails the build when ranking quality
  regresses. Wired into CI as a required step; a synthetic fixture corpus ships in
  the wheel.
- Benchmark converters for public datasets (HotpotQA, LoCoMo, LongMemEval,
  GammaCorpus), a corpus-subset builder, and a source downloader.

**Knowledge base**
- Incremental rendering: community matching, content-hash skip, and GC.
- Community summaries generated in one call at a large token budget, grounded with
  member-id citations and a bounded context.
- Alignment with Open Knowledge Format (OKF) v0.1; trust/provenance YAML
  frontmatter; `contradicted_by` trust signal; numbered member references.
- Citation lint and a grounding-coverage metric.
- Background generation via a `kb_update` task with a dirty-watermark skip.

**Graph**
- `GraphQueryService` seam with an OSS relational implementation.
- `entity_neighborhood` and workspace entity enumeration; `fragments_for_memory`.
- Communities-only graph option, node labels, and member citation ids for UI wiring.

**Chat and ingestion**
- Thread `ownership` (user- vs workspace-scoped) with cross-workspace user-thread
  APIs and filters; sub-threads via `parent_thread`; per-thread idle policy with
  archive/hide lifecycle.
- Email as a first-class ingestion source (`/v1/ingest`), riding the same
  decompose + enrich + entity-accretion pipeline as chat and documents.
- Image-aware (multimodal) fact decomposition.

**Knowledgebase**
- Corpus coverage: the share of graph memories that reached an article, reported on
  the `Knowledgebase` model and the index article, alongside counts of what
  `min_community_size` and `max_communities` dropped. A KB covering a third of a
  workspace previously looked exactly like one covering all of it.
- A post-generation link-integrity sweep records how many wikilinks resolve, which
  catches the case construction cannot see: an article reused verbatim on a
  content-hash hit whose target was since reclaimed by GC.
- OKF front matter on god-node entity articles, which were the only pages in an
  exported vault carrying none.
- A validated mermaid topology diagram on community articles, showing which
  clusters a community bridges to. Diagrams that fail validation are omitted
  rather than emitted, since a broken fence renders as an error block.

**Skills, LLM, and platform**
- Skills: inline bundle files on create, addenda, `_global` skills in workspace
  listings, recursive skill-folder parsing, and per-file delete.
- LLM: pluggable API key store, a Fireworks provider, env-tunable completion-token
  caps, and attribution.
- Aether: configurable terminator `header_mode`, bounded worker shutdown, capped
  pool-task concurrency, and periodic in-flight task-count logging.

### Changed

- **Defaults are now "batteries included": `pip install memorylayer-server && memorylayer serve`
  works with no API key and no peer container.** Previously a zero-config server could not
  store a single memory — the default embedding provider pointed at a
  `memorylayer-embed-server` peer on `localhost:61051`, so the first `POST /v1/memories`
  failed with an opaque 500. Specifically:
  - `MEMORYLAYER_EMBEDDING_PROVIDER` now defaults to `hash` (lexical, dependency-free,
    offline) instead of `embed_server`. It is a *working* default, not a good embedder:
    the provider logs a startup warning naming the upgrade paths. The published Docker
    image pins `MEMORYLAYER_EMBEDDING_PROVIDER=embed_server`, so it keeps its GPU-backed
    behavior and cannot silently fall back to lexical matching.
  - **`memorylayer-embed-server` now defaults its single-vector provider to
    `sentence_transformers`** (`all-MiniLM-L6-v2`, 384-d, CPU) instead of
    `vllm_subprocess` (`Qwen3-VL-Embedding-2B`, 2048-d, GPU). Running the embed server
    no longer requires a GPU, a CUDA toolchain, or a 2B-parameter download. Install with
    the new `local` extra: `pip install "memorylayer-embed-server[local]"`. **GPU
    deployments must now set `MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_subprocess`
    explicitly** — and note this changes the vector dimension, which is a property of
    stored data (see below).
  - `DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER` moved 1024 → **384** to match that model.
    Both READMEs gained an "Embedding dimensions" section: the SQLite schema is
    dimension-agnostic (`embedding` is a plain `BLOB`, no migration needed), but mixing
    widths in one workspace makes `sqlite-vec` return **no results at all** for the whole
    query, and makes the pure-Python fallback silently drop the older memories. Changing
    the model — even to another 384-d one — means re-embedding.
  - `MEMORYLAYER_SESSION_SERVICE` now defaults to `persistent` instead of `in-memory`, so
    sessions and token-budget extraction state survive a restart. Set `in-memory`
    explicitly for ephemeral deployments.
  - The `embed_server` provider now probes its peer at startup and logs an actionable
    error when unreachable, instead of failing later as a generic 500 per write.
  - A standalone server no longer logs an alarming `AETHER_API_KEY is required` warning.
    When no `AETHER_*` variable is set at all, Aether is reported as not configured and
    stays inactive; fail-fast credential checking is unchanged whenever Aether *is*
    configured.
  - "LLM provider not configured" is no longer warned three times per stored memory. The
    registry already announces it once at startup, so these expected skips are now debug
    level; genuine LLM failures still warn.
- **`GET /v1/memories` and `POST /v1/memories/recall` no longer return raw embedding
  vectors by default.** They dominated the payload (~8 KB per memory at 1024 dimensions,
  so ~80 KB for a default recall) and callers rarely use them. Pass
  `include_embeddings=true` to get them back.
- **CI is now generated from `versions.yaml`** by `scitrera-repo-tools`; the
  hand-written `ci.yml` and `release.yml` were replaced by per-lane workflows
  (`test-python`, `test-npm`, `publish-python`, `publish-npm`, `version-check`).
- Authorization: `documents` write/delete now require READWRITE rather than
  MANAGE, matching every other content resource. A workspace writer can manage
  that workspace's documents.
- Auto-association batches and deterministically gates relationship
  classification, cutting LLM calls per stored memory.
- Backlink salience defaults to off, and `similar_to` edges are excluded from
  in-degree.
- Knowledge-base articles use numbered member references instead of raw ids.
- `Entity.entity_type` widened to `str` so ontology domain types round-trip.

### Fixed

- npm and Python distributions exclude local agent state, credentials, and data
  files even when built from a developer checkout. Package artifacts include
  their license text.
- LangChain declares its legacy API dependency and handles the SDK's public
  exception types. Both Python adapters are now covered by release CI.
- TypeScript SDK browser bundling no longer tries to resolve Node filesystem
  modules; filesystem-only skill helpers report a clear unsupported-runtime error.
- RPG endpoints preserve authorization denials as HTTP 403 responses.
- Updated vulnerable npm dependencies and `python-multipart`; updated the nested
  Go transport to Aether API/SDK v0.2.3. Both the dependency-free core Go module
  and the nested Aether module require Go 1.25.14 and release at v0.2.0.

- Chat threads owned by a user are isolated per on-behalf-of user (cross-user
  leak); thread storage is owner-scoped.
- Caller-asserted LLM identity headers no longer reach third-party providers.
  `X-Scitrera-Source` / `X-Scitrera-Tenant` (and the per-task `X-Scitrera-Task-Id`)
  were stamped on **every** OpenAI-compatible provider with no check on where it
  pointed, so a deployment that set `MEMORYLAYER_TENANT_ID` and ran a profile
  against `api.openai.com` or Fireworks sent its tenant identifier to that vendor
  on every call. Stamping is now gated on `MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS`,
  an allowlist of first-party hosts matched against a profile's `base_url`.
  **This is empty by default, so no profile is stamped unless you configure it** —
  set it to your gateway's host(s) if you rely on these headers. A profile with no
  `base_url` (i.e. pointed at the vendor's own endpoint) is never stamped.
- Knowledgebase wikilinks resolve. Entity articles are filed under an id carrying a
  disambiguation suffix, but cross-links were built from a bare slug, so **every**
  entity-to-entity link — in the index's Central Nodes list and in each article's
  Connections list — pointed at a file that did not exist. Links are now resolved
  against the set of articles a run actually produces, and a reference with no
  article renders as plain text instead of a dangling link. The index also shows
  real entity titles rather than truncated uuids.
- `recall()` reranks the overfetch pool before truncating, so it always returns at
  most `limit` results.
- Registry-recall gained a query-relevance floor as a safety valve against
  member-dump flooding.
- Service and agent principals never own a memory.
- `create_association` is idempotent; graph analysis runs CPU-bound compute off the
  event loop and uses a single-pass `analyze()` with bulk edge fetch.
- `/livez` and `/v1/health` are exempt from rate limiting.
- Skill folder parsing no longer double-wraps frontmatter metadata and excludes
  `__pycache__`/`*.pyc`/`*.pyo`.
- Multi-extension routing in the embed server no longer 404s silently.

### Notes for packagers

- `release-manual.yml` is gone; every workflow is now generated from
  `versions.yaml`. To re-publish after a partially successful release, re-dispatch
  `publish-python` or `publish-npm` — both skip versions already on the registry,
  so a re-run publishes exactly what is missing without needing to identify which
  package failed. Docker images come from `build-docker`.

- `memorylayer-sdk-go` is consumed by tag. Publishing it means pushing
  per-module tags (`memorylayer-sdk-go/v0.2.0`, and
  `memorylayer-sdk-go/aether/v0.2.0` for the nested module) — there is no
  artifact upload step.
- The Go SDK is not yet covered by a generated CI lane; see the note in
  `versions.yaml` under `ci:` for why and for the manual verification commands.
- `ruff` lint and format checks remain disabled in CI pending a tree-wide
  autofix pass; see the `ci.python.lint` comment in `versions.yaml`.

## [0.1.22] and earlier

See the git history. Release notes start with 0.2.0.
